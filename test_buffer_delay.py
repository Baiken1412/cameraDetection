"""
缓冲区延迟测试脚本
验证：不清空缓冲区时，延迟是否会无限增长
"""
import cv2
import time
from datetime import datetime

def test_buffer_accumulation(rtsp_url: str, duration_seconds: int = 60):
    """
    测试缓冲区延迟累积

    Args:
        rtsp_url: RTSP流地址（或摄像头ID）
        duration_seconds: 测试时长（秒）
    """
    print(f"=== 缓冲区延迟测试 ===")
    print(f"RTSP URL: {rtsp_url}")
    print(f"测试时长: {duration_seconds}秒")
    print(f"策略: 不清空缓冲区，持续读取")
    print()

    # 打开视频流
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("❌ 无法打开视频流")
        return

    # 尝试设置缓冲区为1（实际效果取决于后端）
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"缓冲区设置: {buffer_size}")
    except:
        print("无法设置缓冲区大小")

    print()
    print("开始测试...")
    print("格式: [时间] 延迟估算 | 说明")
    print("-" * 60)

    start_time = time.time()
    frame_count = 0
    last_log_time = start_time

    # 记录延迟历史
    delay_history = []

    while time.time() - start_time < duration_seconds:
        # 读取帧（不清空缓冲区）
        read_start = time.time()
        ret, frame = cap.read()
        read_duration = time.time() - read_start

        if not ret:
            print(f"[{time.time()-start_time:.1f}s] ❌ 读取失败")
            break

        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time

        # 每5秒输出一次状态
        if current_time - last_log_time >= 5.0:
            # 估算延迟：基于读取耗时（粗略估算）
            # 实际延迟 ≈ 缓冲区积压帧数 / fps
            fps = frame_count / elapsed if elapsed > 0 else 0

            # 如果读取很快（<0.01s），说明缓冲区有积压
            # 如果读取很慢（>0.05s），说明在等待新帧
            estimated_delay = "未知"
            if read_duration < 0.01:
                estimated_delay = "缓冲区有积压（读取很快）"
            elif read_duration > 0.04:  # 25fps = 40ms/帧
                estimated_delay = "缓冲区几乎为空（等待新帧）"
            else:
                estimated_delay = "正常（实时读取）"

            print(
                f"[{elapsed:6.1f}s] "
                f"已读取: {frame_count:4d}帧 | "
                f"FPS: {fps:5.1f} | "
                f"读取耗时: {read_duration*1000:5.2f}ms | "
                f"{estimated_delay}"
            )

            delay_history.append({
                'time': elapsed,
                'read_duration': read_duration,
                'fps': fps
            })

            last_log_time = current_time

        # 每10秒模拟一次耗时操作（YOLO检测）
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            if frame_count % 250 == 0:  # 避免重复触发
                print(f"\n[{elapsed:6.1f}s] 🔍 模拟YOLO检测（耗时500ms）...")
                time.sleep(0.5)  # 模拟YOLO耗时
                print(f"[{elapsed:6.1f}s] ✓ YOLO检测完成\n")

    cap.release()

    print("-" * 60)
    print("\n=== 测试总结 ===")
    print(f"总帧数: {frame_count}")
    print(f"平均FPS: {frame_count / duration_seconds:.1f}")
    print()
    print("延迟分析:")

    # 分析读取耗时的变化趋势
    if len(delay_history) > 2:
        first_read = delay_history[0]['read_duration']
        last_read = delay_history[-1]['read_duration']

        print(f"  初始读取耗时: {first_read*1000:.2f}ms")
        print(f"  最终读取耗时: {last_read*1000:.2f}ms")
        print(f"  变化: {(last_read - first_read)*1000:+.2f}ms")
        print()

        if abs(last_read - first_read) < 0.01:
            print("✓ 结论: 延迟稳定，未无限增长")
        elif last_read > first_read:
            print("⚠️  结论: 延迟有所增加（但应该会稳定）")
        else:
            print("✓ 结论: 延迟减少或稳定")


if __name__ == "__main__":
    import sys

    # 使用示例
    # 1. USB摄像头: python test_buffer_delay.py 0
    # 2. RTSP流: python test_buffer_delay.py rtsp://user:pass@ip/stream

    if len(sys.argv) > 1:
        url = sys.argv[1]
        # 如果是数字，转换为int（USB摄像头）
        try:
            url = int(url)
        except:
            pass
    else:
        print("用法:")
        print("  USB摄像头: python test_buffer_delay.py 0")
        print("  RTSP流: python test_buffer_delay.py rtsp://...")
        print()
        print("使用默认USB摄像头(0)进行测试...")
        url = 0

    # 运行测试（60秒）
    test_buffer_accumulation(url, duration_seconds=60)
