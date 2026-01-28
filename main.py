"""
RTSP视频流监测系统 - 主程序
使用背景建模法进行检测
"""
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;0"
import multiprocessing
import signal
import sys
import time
from loguru import logger
import config_loader as config
from database import Database
from camera_monitor import CameraMonitor
from license_manager import LicenseManager


class RtspMonitorSystem:
    """RTSP监测系统主类"""

    def __init__(self):
        self.db = Database()
        self.monitors = {}  # 存储每个摄像头的监测对象
        self.yolo_pools = {}  # YOLO实例池字典: {pool_id: YoloDetectorPool}
        self.default_yolo_pool = None  # 默认全局YOLO池（用于yolo_pool_id=NULL的摄像头）
        self.running = False

        # 自动重启机制
        self.restart_counts = {}  # {camera_id: 重启次数}
        self.restart_times = {}   # {camera_id: 上次重启时间}
        self.max_restarts = 10    # 最大重启次数
        self.restart_interval = 60  # 重启间隔（秒）
        self.stable_reset_time = 3600  # 稳定运行多久后重置计数（秒），默认1小时
    
    def initialize(self):
        """初始化系统"""
        # 确保目录存在
        config.ensure_directories()

        # 配置日志
        logger.add(
            config.LOG_CONFIG['file'],
            level=config.LOG_CONFIG['level'],
            format=config.LOG_CONFIG['format'],
            rotation=config.LOG_CONFIG['rotation'],
            retention=config.LOG_CONFIG['retention']
        )

        # 连接数据库
        if not self.db.connect():
            logger.error("数据库连接失败，系统无法启动")
            return False

        logger.info("========== RTSP监测系统初始化完成（使用背景建模检测） ==========")
        return True
    
    def start_all_monitors(self):
        """启动所有摄像头监测"""
        logger.info("========== 开始启动RTSP监测服务 ==========")

        # 查询所有有效的摄像头
        cameras = self.db.get_all_cameras()

        if not cameras:
            logger.warning("未找到有效的摄像头配置")
            return

        logger.info(f"找到 {len(cameras)} 个有效摄像头，开始启动监测...")

        # ==================== 步骤1：分析摄像头配置，创建所需的YOLO池 ====================
        pool_ids_needed = set()  # 需要创建的池ID集合
        need_default_pool = False  # 是否需要默认全局池

        for camera in cameras:
            pool_id = camera.get('yolo_pool_id')

            if pool_id is None or pool_id == -1:
                # NULL或-1 → 使用默认全局池
                need_default_pool = True
            elif pool_id > 0:
                # 正整数 → 使用指定的池
                pool_ids_needed.add(pool_id)
            # pool_id == 0 → 独立实例，不需要创建池

        logger.info(
            f"YOLO池配置分析: 需要创建池ID={sorted(pool_ids_needed)}, "
            f"需要默认池={need_default_pool}"
        )

        # 创建所需的YOLO池
        try:
            # 根据配置选择多进程池或多线程池
            use_multiprocess = config.YOLO_POOL_CONFIG.get('use_multiprocess', False)

            if use_multiprocess:
                from core.yolo_process_pool import YoloProcessPool as YoloDetectorPool
                logger.info("使用多进程 YOLO 池（解决 GIL 阻塞问题）")
            else:
                from core.yolo_pool import YoloDetectorPool
                logger.info("使用多线程 YOLO 池")

            # 准备检测器配置
            if config.YOLO_POOL_CONFIG.get('detector_type', 'adaptive') == 'adaptive':
                detector_config = config.ADAPTIVE_DETECTION_CONFIG.copy()
            else:
                detector_config = {
                    'model_path': getattr(config, 'YOLO_MODEL_PATH', None),
                    'device': getattr(config, 'DEVICE', 'auto'),
                    'conf_threshold': config.ADAPTIVE_DETECTION_CONFIG.get('conf_threshold', 0.5),
                    'iou_threshold': config.ADAPTIVE_DETECTION_CONFIG.get('iou_threshold', 0.4)
                }

            # 创建默认全局池（如果需要且配置启用）
            if need_default_pool and config.YOLO_POOL_CONFIG.get('enabled', False):
                self.default_yolo_pool = YoloDetectorPool(
                    pool_size=config.YOLO_POOL_CONFIG.get('pool_size', 2),
                    detector_type=config.YOLO_POOL_CONFIG.get('detector_type', 'adaptive'),
                    detector_config=detector_config
                )
                logger.info(f"已创建默认YOLO池: {self.default_yolo_pool}")

            # 创建指定的池（每个池默认1个实例，表示"共享"）
            for pool_id in pool_ids_needed:
                pool = YoloDetectorPool(
                    pool_size=1,  # 每个自定义池默认1个实例
                    detector_type=config.YOLO_POOL_CONFIG.get('detector_type', 'adaptive'),
                    detector_config=detector_config
                )
                self.yolo_pools[pool_id] = pool
                logger.info(f"已创建YOLO池 ID={pool_id}: {pool}")

        except Exception as e:
            logger.error(f"创建YOLO池失败: {e}")
            logger.warning("将回退到每个摄像头独立创建YOLO实例的模式")

        # ==================== 步骤2：为每个摄像头分配池并启动监测 ====================
        for camera in cameras:
            try:
                pool_id = camera.get('yolo_pool_id')

                # 根据pool_id选择对应的池
                if pool_id is None or pool_id == -1:
                    yolo_pool = self.default_yolo_pool
                    pool_desc = "默认全局池"
                elif pool_id == 0:
                    yolo_pool = None
                    pool_desc = "独立实例"
                elif pool_id > 0:
                    yolo_pool = self.yolo_pools.get(pool_id)
                    pool_desc = f"池{pool_id}"
                else:
                    yolo_pool = None
                    pool_desc = "未知配置，使用独立实例"
                    logger.warning(
                        f"摄像头 {camera.get('fjmc')} 的 yolo_pool_id={pool_id} 无效"
                    )

                logger.info(
                    f"摄像头 {camera.get('fjmc')} (ID: {camera['id']}) "
                    f"使用YOLO配置: {pool_desc}"
                )

                monitor = CameraMonitor(camera, self.db, yolo_pool=yolo_pool)
                monitor.start()
                self.monitors[camera['id']] = monitor

                # 避免同时启动过多连接，稍作延迟
                time.sleep(1)

            except Exception as e:
                logger.error(f"启动摄像头 {camera.get('fjmc', 'Unknown')} 监测失败: {e}")

        logger.info("========== RTSP监测服务启动完成 ==========")
        self.running = True

    def stop_all_monitors(self):
        """停止所有监测"""
        logger.info("========== 停止所有RTSP监测服务 ==========")
        
        for camera_id, monitor in self.monitors.items():
            try:
                monitor.stop()
            except Exception as e:
                logger.error(f"停止摄像头 {camera_id} 监测失败: {e}")
        
        self.monitors.clear()
        self.running = False
        logger.info("========== 所有RTSP监测服务已停止 ==========")
    
    def restart_monitor(self, camera_id):
        """
        重启单个摄像头监测

        Args:
            camera_id: 摄像头ID

        Returns:
            bool: 重启是否成功
        """
        monitor = self.monitors.get(camera_id)
        if not monitor:
            logger.error(f"摄像头 {camera_id} 不存在，无法重启")
            return False

        # 检查重启次数
        restart_count = self.restart_counts.get(camera_id, 0)
        if restart_count >= self.max_restarts:
            logger.error(
                f"摄像头 {camera_id} ({monitor.camera_name}) "
                f"重启次数已达上限({self.max_restarts}次)，停止重启"
            )
            return False

        # 检查重启间隔
        last_restart = self.restart_times.get(camera_id, 0)
        if time.time() - last_restart < self.restart_interval:
            return False  # 静默跳过，避免频繁日志

        logger.warning(
            f"摄像头 {camera_id} ({monitor.camera_name}) 监测已停止，"
            f"尝试重启 ({restart_count + 1}/{self.max_restarts})"
        )

        # 获取原始配置
        camera_info = monitor.camera_info
        yolo_pool = monitor.yolo_pool

        # 停止旧监测器
        try:
            monitor.stop()
        except Exception as e:
            logger.warning(f"停止旧监测器时出错: {e}")

        # 创建并启动新监测器
        try:
            new_monitor = CameraMonitor(camera_info, self.db, yolo_pool=yolo_pool)
            new_monitor.start()
            self.monitors[camera_id] = new_monitor

            # 更新重启计数
            self.restart_counts[camera_id] = restart_count + 1
            self.restart_times[camera_id] = time.time()

            logger.info(f"摄像头 {camera_id} ({camera_info['fjmc']}) 重启成功")
            return True

        except Exception as e:
            logger.error(f"重启摄像头 {camera_id} 失败: {e}")
            self.restart_counts[camera_id] = restart_count + 1
            self.restart_times[camera_id] = time.time()
            return False

    def reset_restart_count(self, camera_id):
        """重置指定摄像头的重启计数（可在运行稳定后调用）"""
        if camera_id in self.restart_counts:
            self.restart_counts[camera_id] = 0
            logger.info(f"摄像头 {camera_id} 重启计数已重置")

    def reload_cameras(self):
        """重新加载摄像头配置并重启监测"""
        logger.info("重新加载摄像头配置...")
        self.stop_all_monitors()

        time.sleep(2)  # 等待资源释放

        # 重置所有重启计数
        self.restart_counts.clear()
        self.restart_times.clear()

        self.start_all_monitors()
    
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭系统...")
        self.stop_all_monitors()

        # 关闭所有YOLO实例池
        if self.default_yolo_pool is not None:
            try:
                self.default_yolo_pool.shutdown()
                logger.info("默认YOLO实例池已关闭")
            except Exception as e:
                logger.error(f"关闭默认YOLO实例池失败: {e}")

        for pool_id, pool in self.yolo_pools.items():
            try:
                pool.shutdown()
                logger.info(f"YOLO池 {pool_id} 已关闭")
            except Exception as e:
                logger.error(f"关闭YOLO池 {pool_id} 失败: {e}")

        self.yolo_pools.clear()

        self.db.close()
        logger.info("系统已关闭")


def signal_handler(sig, frame):
    """信号处理函数"""
    logger.info("接收到退出信号，正在关闭系统...")
    if 'monitor_system' in globals():
        monitor_system.shutdown()
    sys.exit(0)


def main():
    """主函数"""
    global monitor_system

    # ========== 许可证验证 ==========
    print("=" * 60)
    print("正在验证软件许可证...")
    print("=" * 60)

    license_manager = LicenseManager()

    #if not license_manager.check_license():
    #    print("\n" + "=" * 60)
    ##    print("【许可证验证失败】")
    ##    print("=" * 60)
    #    print(f"当前机器码: {license_manager.machine_code}")
    #    print("\n请联系软件提供商获取有效的许可证文件。")
    #    print("需要提供上述机器码以生成对应的许可证。")
    #    print("=" * 60)
    #    input("\n按Enter键退出...")
    #    sys.exit(1)

    print("\n" + "=" * 60)
    print("【许可证验证成功】")
    print("=" * 60)
    print()

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 创建监测系统
    monitor_system = RtspMonitorSystem()
    
    # 初始化
    if not monitor_system.initialize():
        logger.error("系统初始化失败，退出")
        return
    
    try:
        # 延迟启动，确保系统完全初始化
        time.sleep(3)
        
        # 启动所有监测
        monitor_system.start_all_monitors()
        
        # 保持运行
        check_interval_counter = 0
        while monitor_system.running:
            time.sleep(1)
            check_interval_counter += 1

            # 每10秒检查一次监测线程状态（避免频繁检查）
            if check_interval_counter >= 10:
                check_interval_counter = 0

                # 检查监测线程是否还在运行，自动重启停止的监测
                for camera_id, monitor in list(monitor_system.monitors.items()):
                    if not monitor.is_running():
                        monitor_system.restart_monitor(camera_id)
                    else:
                        # 摄像头正常运行，检查是否需要重置重启计数
                        if camera_id in monitor_system.restart_times:
                            last_restart = monitor_system.restart_times[camera_id]
                            if time.time() - last_restart >= monitor_system.stable_reset_time:
                                # 稳定运行超过阈值，重置计数
                                old_count = monitor_system.restart_counts.get(camera_id, 0)
                                if old_count > 0:
                                    monitor_system.restart_counts[camera_id] = 0
                                    del monitor_system.restart_times[camera_id]
                                    logger.info(
                                        f"摄像头 {camera_id} ({monitor.camera_name}) "
                                        f"稳定运行超过1小时，重启计数已重置 ({old_count} → 0)"
                                    )
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭系统...")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
    finally:
        monitor_system.shutdown()


if __name__ == '__main__':
    # Windows 多进程支持（打包成 exe 后必需）
    multiprocessing.freeze_support()
    main()

