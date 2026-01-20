"""
系统诊断脚本

收集运行环境、网络、视频源质量等信息，辅助问题排查
"""
import os
import sys
import cv2
import time
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger

# 诊断结果
diagnosis = {
    "timestamp": datetime.now().isoformat(),
    "environment": {},
    "hardware": {},
    "network": {},
    "video_sources": [],
    "issues": [],
    "recommendations": []
}


def check_environment():
    """检查运行环境"""
    print("\n" + "=" * 60)
    print("1. 运行环境检查")
    print("=" * 60)

    env = diagnosis["environment"]

    env["python_version"] = sys.version
    env["platform"] = platform.platform()
    env["machine"] = platform.machine()
    env["opencv_version"] = cv2.__version__

    print(f"  Python: {sys.version.split()[0]}")
    print(f"  平台: {platform.platform()}")
    print(f"  OpenCV: {cv2.__version__}")

    # 检查关键依赖
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
        print(f"  NumPy: {np.__version__}")
    except ImportError:
        diagnosis["issues"].append("NumPy 未安装")

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
    except ImportError:
        env["torch_version"] = "未安装"
        print("  PyTorch: 未安装")

    # 检查推理引擎
    try:
        import onnxruntime as ort
        env["onnxruntime_version"] = ort.__version__
        env["onnxruntime_providers"] = ort.get_available_providers()
        print(f"  ONNX Runtime: {ort.__version__}")
        print(f"  可用 Provider: {ort.get_available_providers()}")
    except ImportError:
        pass

    try:
        from openvino.runtime import Core
        env["openvino_available"] = True
        print("  OpenVINO: 可用")
    except ImportError:
        env["openvino_available"] = False


def check_hardware():
    """检查硬件性能"""
    print("\n" + "=" * 60)
    print("2. 硬件性能检查")
    print("=" * 60)

    hw = diagnosis["hardware"]

    # CPU 信息
    hw["cpu_count"] = os.cpu_count()
    print(f"  CPU 核心数: {os.cpu_count()}")

    # 内存信息
    try:
        import psutil
        mem = psutil.virtual_memory()
        hw["memory_total_gb"] = round(mem.total / (1024**3), 1)
        hw["memory_available_gb"] = round(mem.available / (1024**3), 1)
        hw["memory_percent"] = mem.percent
        hw["cpu_percent"] = psutil.cpu_percent(interval=1)

        print(f"  内存: {hw['memory_available_gb']:.1f}GB / {hw['memory_total_gb']:.1f}GB ({mem.percent}%)")
        print(f"  CPU 占用: {hw['cpu_percent']}%")

        if mem.percent > 85:
            diagnosis["issues"].append(f"内存占用过高: {mem.percent}%")
            diagnosis["recommendations"].append("考虑关闭其他程序或增加内存")

        if hw['cpu_percent'] > 80:
            diagnosis["issues"].append(f"CPU 占用过高: {hw['cpu_percent']}%")
            diagnosis["recommendations"].append("考虑减少同时监控的摄像头数量或升级硬件")

        # 磁盘信息
        try:
            config_path = Path(__file__).parent / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            image_path = config.get("RTSP_MONITOR_CONFIG", {}).get("image_save_path", "")

            if image_path:
                disk = psutil.disk_usage(image_path)
                hw["disk_total_gb"] = round(disk.total / (1024**3), 1)
                hw["disk_free_gb"] = round(disk.free / (1024**3), 1)
                hw["disk_percent"] = disk.percent

                print(f"  磁盘({image_path[:20]}...): {hw['disk_free_gb']:.1f}GB 可用 / {hw['disk_total_gb']:.1f}GB ({disk.percent}%)")

                if disk.percent > 90:
                    diagnosis["issues"].append(f"磁盘空间不足: {disk.percent}% 已使用")
                    diagnosis["recommendations"].append("清理磁盘空间或更换更大容量的存储")

                if hw['disk_free_gb'] < 10:
                    diagnosis["issues"].append(f"磁盘剩余空间过低: {hw['disk_free_gb']:.1f}GB")
        except Exception:
            pass

        # 磁盘写入速度测试
        try:
            test_file = Path(__file__).parent / ".disk_test_temp"
            test_data = b"x" * (1024 * 1024)  # 1MB

            start = time.time()
            with open(test_file, 'wb') as f:
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())
            write_time = time.time() - start

            test_file.unlink()  # 删除测试文件

            write_speed = 1 / write_time  # MB/s
            hw["disk_write_speed_mbps"] = round(write_speed, 1)
            print(f"  磁盘写入速度: {write_speed:.1f} MB/s")

            if write_speed < 10:
                diagnosis["issues"].append(f"磁盘写入速度较慢: {write_speed:.1f} MB/s")
                diagnosis["recommendations"].append("磁盘I/O性能不足可能导致丢帧，考虑使用SSD")

        except Exception as e:
            print(f"  磁盘写入测试失败: {e}")

    except ImportError:
        print("  (安装 psutil 可获取更详细的硬件信息: pip install psutil)")
        hw["psutil_available"] = False


def extract_ip_from_rtsp(rtsp_url):
    """从 RTSP URL 中提取 IP 地址"""
    import re
    match = re.search(r'rtsp://[^:]*:?[^@]*@?(\d+\.\d+\.\d+\.\d+)', rtsp_url)
    if match:
        return match.group(1)
    match = re.search(r'rtsp://(\d+\.\d+\.\d+\.\d+)', rtsp_url)
    if match:
        return match.group(1)
    return None


def check_network(hosts=None):
    """检查网络连通性"""
    print("\n" + "=" * 60)
    print("3. 网络检查")
    print("=" * 60)

    net = diagnosis["network"]

    # 默认测试目标
    if hosts is None:
        hosts = ["114.114.114.114"]

        # 从数据库获取摄像头 IP 进行测试
        try:
            cameras = get_cameras_from_db()
            camera_ips = set()
            for cam in cameras[:5]:  # 最多测试5个摄像头IP
                ip = extract_ip_from_rtsp(cam.get('rtsp_url', ''))
                if ip:
                    camera_ips.add(ip)
            if camera_ips:
                hosts.extend(list(camera_ips))
                print(f"  将测试 {len(camera_ips)} 个摄像头IP的网络连通性")
        except Exception:
            pass

    for host in hosts:
        try:
            # Windows 和 Linux 的 ping 参数不同
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "3", "-w", "1000", host]
            else:
                cmd = ["ping", "-c", "3", "-W", "1", host]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout

            # 解析延迟
            if "平均" in output or "Average" in output or "avg" in output:
                # 尝试提取平均延迟
                import re
                match = re.search(r'(\d+\.?\d*)ms', output)
                if match:
                    latency = float(match.group(1))
                    net[host] = {"status": "ok", "latency_ms": latency}
                    print(f"  {host}: {latency:.1f}ms")

                    if latency > 100:
                        diagnosis["issues"].append(f"网络延迟较高: {host} = {latency}ms")
                else:
                    net[host] = {"status": "ok", "latency_ms": "unknown"}
                    print(f"  {host}: 连通")
            elif result.returncode == 0:
                net[host] = {"status": "ok"}
                print(f"  {host}: 连通")
            else:
                net[host] = {"status": "failed"}
                print(f"  {host}: 不通")
                diagnosis["issues"].append(f"无法连接: {host}")

        except subprocess.TimeoutExpired:
            net[host] = {"status": "timeout"}
            print(f"  {host}: 超时")
            diagnosis["issues"].append(f"Ping 超时: {host}")
        except Exception as e:
            net[host] = {"status": "error", "error": str(e)}
            print(f"  {host}: 错误 - {e}")


def check_video_source(url_or_id, name="Camera"):
    """检查单个视频源"""
    import numpy as np
    print(f"\n  检查: {name}")

    result = {
        "name": name,
        "url": str(url_or_id),
        "status": "unknown"
    }

    try:
        # 连接
        start_time = time.time()
        if isinstance(url_or_id, int):
            cap = cv2.VideoCapture(url_or_id)
        else:
            # RTSP 流，强制 TCP
            if 'rtsp://' in str(url_or_id).lower():
                url = url_or_id
                if '?' not in url:
                    url += '?rtsp_transport=tcp'
                elif 'rtsp_transport' not in url:
                    url += '&rtsp_transport=tcp'
                cap = cv2.VideoCapture(url)
            else:
                cap = cv2.VideoCapture(url_or_id)

        connect_time = time.time() - start_time

        if not cap.isOpened():
            result["status"] = "failed"
            result["error"] = "无法打开视频流"
            print(f"    状态: 连接失败")
            diagnosis["issues"].append(f"视频源连接失败: {name}")
            return result

        result["connect_time_sec"] = round(connect_time, 2)
        print(f"    连接耗时: {connect_time:.2f}秒")

        if connect_time > 5:
            diagnosis["issues"].append(f"连接耗时过长: {name} = {connect_time:.1f}秒")
            diagnosis["recommendations"].append(f"检查 {name} 的网络连接质量")

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        result["width"] = width
        result["height"] = height
        result["fps"] = fps

        print(f"    分辨率: {width}x{height}")
        print(f"    帧率: {fps:.1f} fps")

        if width < 640 or height < 480:
            diagnosis["issues"].append(f"分辨率较低: {name} = {width}x{height}")
            diagnosis["recommendations"].append(f"考虑提高 {name} 的分辨率设置")

        # 测试读取帧（测5秒，检测实际帧率和稳定性）
        test_duration = 5.0  # 测试时长（秒）
        frames_read = 0
        frames_failed = 0
        first_frame_checked = False
        nominal_fps = fps if fps > 0 else 25  # 标称帧率，用于防止读缓冲区过快

        start_time = time.time()
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= test_duration:
                break

            ret, frame = cap.read()

            if ret and frame is not None:
                frames_read += 1

                # 检查帧质量（第一帧）
                if not first_frame_checked:
                    first_frame_checked = True
                    # 亮度检测
                    frame_mean = float(np.mean(frame))
                    frame_std = float(np.std(frame))
                    result["frame_mean"] = frame_mean
                    result["frame_std"] = frame_std

                    if frame_mean < 10:
                        diagnosis["issues"].append(f"画面过暗: {name} (亮度={frame_mean:.1f})")
                    elif frame_mean > 245:
                        diagnosis["issues"].append(f"画面过亮: {name} (亮度={frame_mean:.1f})")

                # 防止读缓冲区过快：如果读取速度超过标称帧率的1.5倍，适当让出CPU
                if elapsed > 0 and frames_read / elapsed > nominal_fps * 1.5:
                    time.sleep(0.01)
            else:
                frames_failed += 1

        elapsed_time = time.time() - start_time

        # 统计结果
        if frames_read > 0:
            actual_fps = frames_read / elapsed_time
            result["actual_fps"] = round(actual_fps, 1)
            result["frames_read"] = frames_read
            result["frames_failed"] = frames_failed
            result["test_duration_sec"] = round(elapsed_time, 1)
            result["frame_loss_rate"] = round(frames_failed / (frames_read + frames_failed) * 100, 1) if (frames_read + frames_failed) > 0 else 0
            result["status"] = "ok"

            print(f"    实际帧率: {actual_fps:.1f} fps ({frames_read}帧/{elapsed_time:.1f}秒)")
            print(f"    丢帧: {frames_failed} 次")

            if actual_fps < 5:
                diagnosis["issues"].append(f"实际帧率过低: {name} = {actual_fps:.1f}fps")
                diagnosis["recommendations"].append(f"检查 {name} 的网络带宽或摄像头性能")

            if result['frame_loss_rate'] > 10:
                diagnosis["issues"].append(f"丢帧率过高: {name} = {result['frame_loss_rate']:.1f}%")
                diagnosis["recommendations"].append(f"检查 {name} 的网络稳定性，考虑使用TCP传输")
        else:
            result["status"] = "failed"
            result["error"] = "无法读取帧"
            print(f"    状态: 无法读取帧")
            diagnosis["issues"].append(f"无法读取帧: {name}")

        cap.release()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"    错误: {e}")
        diagnosis["issues"].append(f"视频源检查异常: {name} - {e}")

    diagnosis["video_sources"].append(result)
    return result


def get_cameras_from_db():
    """从数据库 app_roomip 表读取摄像头列表"""
    cameras = []

    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        db_config = config.get("DATABASE_CONFIG", {})

        import pymysql
        conn = pymysql.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 3306),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', ''),
            charset=db_config.get('charset', 'utf8mb4'),
            connect_timeout=10
        )

        cursor = conn.cursor()
        # 查询 app_roomip 表，获取摄像头信息
        cursor.execute("""
            SELECT id, fjmc, rtspssl, gnslx
            FROM app_roomip
            WHERE rtspssl IS NOT NULL AND rtspssl != ''
            ORDER BY id
        """)

        for row in cursor.fetchall():
            cameras.append({
                'id': row[0],
                'name': row[1] or f"摄像头{row[0]}",
                'rtsp_url': row[2],
                'area': row[3] or ''
            })

        cursor.close()
        conn.close()

    except ImportError:
        print("  pymysql 未安装，无法连接数据库")
    except Exception as e:
        print(f"  数据库连接失败: {e}")

    return cameras


def check_video_sources_from_db():
    """从数据库读取并检查所有视频源"""
    print("\n" + "=" * 60)
    print("4. 数据库视频源检查 (app_roomip)")
    print("=" * 60)

    cameras = get_cameras_from_db()

    if not cameras:
        print("  未找到摄像头配置或数据库连接失败")
        print("  可使用 --rtsp 参数手动测试: python diagnose.py --rtsp 'rtsp://...'")
        return

    print(f"  从数据库读取到 {len(cameras)} 个摄像头\n")

    for cam in cameras:
        print(f"  [{cam['id']}] {cam['name']} ({cam['area']})")
        print(f"      URL: {cam['rtsp_url'][:50]}..." if len(cam['rtsp_url']) > 50 else f"      URL: {cam['rtsp_url']}")
        check_video_source(cam['rtsp_url'], f"{cam['name']}(ID:{cam['id']})")
        print()


def check_video_sources():
    """检查所有配置的视频源"""
    print("\n" + "=" * 60)
    print("4. 视频源检查")
    print("=" * 60)

    # 加载配置
    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        rtsp_config = config.get("RTSP_MONITOR_CONFIG", {})

        # 检查 USB 摄像头
        if rtsp_config.get("enable_usb_camera", False):
            usb_id = rtsp_config.get("usb_camera_id", 0)
            usb_name = rtsp_config.get("usb_camera_name", "USB摄像头")
            check_video_source(usb_id, usb_name)

    except Exception as e:
        print(f"  加载配置失败: {e}")

    # 提示
    print("\n  提示: 使用 --rtsp 参数可测试指定的 RTSP 地址")
    print("    python diagnose.py --rtsp 'rtsp://...'")


def check_log_errors():
    """检查日志中的错误"""
    print("\n" + "=" * 60)
    print("5. 日志错误检查")
    print("=" * 60)

    log_dir = Path(__file__).parent / "logs"
    if not log_dir.exists():
        print("  日志目录不存在")
        return

    # 查找最新的日志文件
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        print("  没有找到日志文件")
        return

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"  检查日志: {latest_log.name}")

    # 统计错误类型
    error_counts = {}
    warning_counts = {}

    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-1000:]  # 只看最后1000行

        for line in lines:
            if '| ERROR |' in line or 'ERROR' in line:
                # 提取错误类型
                if '连接失败' in line or '连接异常' in line:
                    error_counts['连接失败'] = error_counts.get('连接失败', 0) + 1
                elif '读取帧失败' in line or '获取帧失败' in line:
                    error_counts['读取帧失败'] = error_counts.get('读取帧失败', 0) + 1
                elif '超时' in line or 'timeout' in line.lower():
                    error_counts['超时'] = error_counts.get('超时', 0) + 1
                elif '重连' in line or 'reconnect' in line.lower():
                    error_counts['重连'] = error_counts.get('重连', 0) + 1
                else:
                    error_counts['其他错误'] = error_counts.get('其他错误', 0) + 1

            elif '| WARNING |' in line or 'WARNING' in line:
                if '断流' in line or 'stale' in line.lower():
                    warning_counts['断流'] = warning_counts.get('断流', 0) + 1
                elif '队列已满' in line:
                    warning_counts['队列满'] = warning_counts.get('队列满', 0) + 1

        if error_counts:
            print("  最近错误统计:")
            for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"    - {err_type}: {count} 次")
                if count > 10:
                    diagnosis["issues"].append(f"频繁出现 {err_type}: {count} 次")

        if warning_counts:
            print("  最近警告统计:")
            for warn_type, count in sorted(warning_counts.items(), key=lambda x: -x[1]):
                print(f"    - {warn_type}: {count} 次")

        if not error_counts and not warning_counts:
            print("  最近1000行日志中没有发现错误或警告")

    except Exception as e:
        print(f"  读取日志失败: {e}")


def print_summary(output_dir=None):
    """打印诊断总结"""
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    if diagnosis["issues"]:
        print("\n发现的问题:")
        for i, issue in enumerate(diagnosis["issues"], 1):
            print(f"  {i}. {issue}")
    else:
        print("\n未发现明显问题")

    if diagnosis["recommendations"]:
        print("\n建议:")
        for i, rec in enumerate(diagnosis["recommendations"], 1):
            print(f"  {i}. {rec}")

    # 保存诊断报告
    if output_dir:
        report_dir = Path(output_dir)
    else:
        report_dir = Path(__file__).parent

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n详细报告已保存: {report_path}")
    return report_path


def main():
    global diagnosis

    # 重置诊断结果
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment": {},
        "hardware": {},
        "network": {},
        "video_sources": [],
        "issues": [],
        "recommendations": []
    }

    print("=" * 60)
    print("系统诊断工具")
    print("=" * 60)

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='系统诊断工具')
    parser.add_argument('--rtsp', type=str, help='测试指定的 RTSP 地址')
    parser.add_argument('--ping', type=str, nargs='+', help='Ping 测试的目标地址')
    parser.add_argument('--quick', action='store_true', help='快速检查（跳过视频源）')
    args = parser.parse_args()

    check_environment()
    check_hardware()

    if args.ping:
        check_network(args.ping)
    else:
        check_network()

    if args.rtsp:
        print("\n" + "=" * 60)
        print("4. 指定视频源检查")
        print("=" * 60)
        check_video_source(args.rtsp, "指定RTSP")
    elif not args.quick:
        # 默认从数据库读取所有摄像头并测试
        check_video_sources_from_db()

    check_log_errors()
    print_summary()


def run_diagnosis(output_dir=None, quick=False):
    """
    运行一次完整诊断（供外部调用）

    Args:
        output_dir: 诊断报告输出目录，默认为脚本所在目录
        quick: 是否快速模式（跳过视频源检查）

    Returns:
        诊断报告文件路径
    """
    global diagnosis

    # 重置诊断结果
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "environment": {},
        "hardware": {},
        "network": {},
        "video_sources": [],
        "issues": [],
        "recommendations": []
    }

    logger.info("开始运行系统诊断...")

    check_environment()
    check_hardware()
    check_network()

    if not quick:
        check_video_sources_from_db()

    check_log_errors()

    report_path = print_summary(output_dir)
    logger.info(f"诊断完成，报告已保存: {report_path}")

    return report_path


class DiagnoseScheduler:
    """诊断调度器，用于定时运行诊断"""

    def __init__(self, interval_minutes=30, output_dir=None):
        """
        Args:
            interval_minutes: 诊断间隔（分钟）
            output_dir: 诊断报告输出目录
        """
        self.interval_minutes = interval_minutes
        self.output_dir = output_dir
        self._running = False
        self._thread = None

    def start(self):
        """启动定时诊断"""
        import threading

        if self._running:
            logger.warning("诊断调度器已在运行")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"诊断调度器已启动，间隔: {self.interval_minutes} 分钟")

    def stop(self):
        """停止定时诊断"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("诊断调度器已停止")

    def _run_loop(self):
        """定时运行循环"""
        # 启动后先运行一次
        try:
            run_diagnosis(output_dir=self.output_dir, quick=False)
        except Exception as e:
            logger.error(f"首次诊断运行失败: {e}")

        # 定时运行
        interval_seconds = self.interval_minutes * 60
        while self._running:
            # 分段睡眠，以便能及时响应停止信号
            for _ in range(interval_seconds):
                if not self._running:
                    break
                time.sleep(1)

            if self._running:
                try:
                    run_diagnosis(output_dir=self.output_dir, quick=False)
                except Exception as e:
                    logger.error(f"定时诊断运行失败: {e}")


if __name__ == "__main__":
    main()
