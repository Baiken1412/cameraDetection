"""
RTSP视频流监测系统 - 主程序
使用背景建模法进行检测
"""
import signal
import sys
import time
from loguru import logger
import config
from database import Database
from camera_monitor import CameraMonitor
from license_manager import LicenseManager


class RtspMonitorSystem:
    """RTSP监测系统主类"""
    
    def __init__(self):
        self.db = Database()
        self.monitors = {}  # 存储每个摄像头的监测对象
        self.running = False
    
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

        # 统计启动情况
        enabled_count = 0
        disabled_count = 0

        # 为每个摄像头启动监测
        for camera in cameras:
            try:
                # 检查 sblx 字段，只有当 sblx=1 时才进行视频检测和截取
                sblx = camera.get('sblx')

                # sblx 可能是整数1或字符串'1'
                if sblx != 1 and sblx != '1':
                    logger.info(f"摄像头 {camera.get('fjmc', 'Unknown')} (ID: {camera.get('id')}) sblx={sblx}，跳过视频检测")
                    disabled_count += 1
                    continue

                logger.info(f"启动摄像头 {camera.get('fjmc', 'Unknown')} (ID: {camera.get('id')}) 监测 (sblx=1)")
                monitor = CameraMonitor(camera, self.db)
                monitor.start()
                self.monitors[camera['id']] = monitor
                enabled_count += 1

                # 避免同时启动过多连接，稍作延迟
                time.sleep(1)

            except Exception as e:
                logger.error(f"启动摄像头 {camera.get('fjmc', 'Unknown')} 监测失败: {e}")

        logger.info(f"========== RTSP监测服务启动完成 (已启动: {enabled_count}, 已跳过: {disabled_count}) ==========")
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
    
    def reload_cameras(self):
        """重新加载摄像头配置并重启监测"""
        logger.info("重新加载摄像头配置...")
        self.stop_all_monitors()
        
        time.sleep(2)  # 等待资源释放
        
        self.start_all_monitors()
    
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭系统...")
        self.stop_all_monitors()
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
    # global monitor_system

    # # ========== 许可证验证 ==========
    # print("=" * 60)
    # print("正在验证软件许可证...")
    # print("=" * 60)

    # license_manager = LicenseManager()

    # if not license_manager.check_license():
    #     print("\n" + "=" * 60)
    #     print("【许可证验证失败】")
    #     print("=" * 60)
    #     print(f"当前机器码: {license_manager.machine_code}")
    #     print("\n请联系软件提供商获取有效的许可证文件。")
    #     print("需要提供上述机器码以生成对应的许可证。")
    #     print("=" * 60)
    #     input("\n按Enter键退出...")
    #     sys.exit(1)

    # print("\n" + "=" * 60)
    # print("【许可证验证成功】")
    # print("=" * 60)
    # print()

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
        while monitor_system.running:
            time.sleep(1)
            
            # 检查监测线程是否还在运行
            for camera_id, monitor in list(monitor_system.monitors.items()):
                if not monitor.is_running():
                    logger.warning(f"摄像头 {camera_id} 监测已停止")
                    # 可以选择重新启动或移除
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭系统...")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
    finally:
        monitor_system.shutdown()


if __name__ == '__main__':
    main()

