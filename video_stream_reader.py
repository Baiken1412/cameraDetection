"""
视频流读取器 - 独立线程持续读取最新帧
解决多摄像头场景下的缓冲区积压问题

核心思想：
1. 独立线程持续读取视频流，丢弃旧帧，只保留最新帧
2. 主线程调用read()时，立即返回最新帧（无需等待）
3. 彻底解决缓冲区积压导致的时间错位问题
"""
import cv2
import threading
import time
from datetime import datetime
from typing import Optional, Tuple
from loguru import logger
import numpy as np


class VideoStreamReader:
    """
    视频流读取器（独立线程模式）

    特性：
    - 独立线程持续读取，确保缓冲区始终为空
    - 主线程调用read()时，返回最新帧（无延迟）
    - 支持OpenCV和PyAV两种模式
    - 线程安全（使用锁保护共享数据）
    """

    def __init__(self, cap_or_url, camera_name: str = "Unknown", use_pyav: bool = False):
        """
        初始化视频流读取器

        Args:
            cap_or_url: cv2.VideoCapture对象 或 RTSP URL字符串
            camera_name: 摄像头名称（用于日志）
            use_pyav: 是否使用PyAV（暂不支持，预留）
        """
        self.camera_name = camera_name
        self.use_pyav = use_pyav

        # 如果传入的是字符串URL，创建VideoCapture
        if isinstance(cap_or_url, str):
            self.cap = cv2.VideoCapture(cap_or_url)
            logger.info(f"[{self.camera_name}] VideoStreamReader: 从URL创建VideoCapture")
        elif isinstance(cap_or_url, int):
            # USB摄像头（设备ID）
            self.cap = cv2.VideoCapture(cap_or_url)
            logger.info(f"[{self.camera_name}] VideoStreamReader: 从设备ID {cap_or_url} 创建VideoCapture")
        else:
            # 传入的是已创建的VideoCapture对象
            self.cap = cap_or_url
            logger.info(f"[{self.camera_name}] VideoStreamReader: 使用已有VideoCapture对象")

        # 线程安全锁
        self.lock = threading.Lock()

        # 最新帧及其时间戳
        self.latest_frame = None
        self.latest_timestamp = None

        # 运行状态
        self.running = True
        self.started = False

        # 性能统计
        self.total_read_count = 0  # 总读取帧数
        self.total_drop_count = 0  # 总丢弃帧数（被覆盖的帧）
        self.last_stats_log_time = time.time()

        # 启动独立读取线程
        self.thread = threading.Thread(
            target=self._update_loop,
            name=f"VideoStreamReader-{self.camera_name}",
            daemon=True
        )
        self.thread.start()

        # 等待第一帧
        self._wait_for_first_frame()

        logger.info(
            f"✓ [{self.camera_name}] VideoStreamReader 已启动 - "
            f"独立线程将持续读取最新帧，彻底解决缓冲区积压"
        )

    def _wait_for_first_frame(self, timeout: float = 5.0):
        """等待第一帧可用"""
        start_time = time.time()
        while self.latest_frame is None:
            if time.time() - start_time > timeout:
                logger.warning(f"[{self.camera_name}] VideoStreamReader: 等待首帧超时（{timeout}秒）")
                break
            time.sleep(0.01)

        if self.latest_frame is not None:
            logger.debug(f"[{self.camera_name}] VideoStreamReader: 首帧已就绪")

    def _update_loop(self):
        """
        独立线程的更新循环

        任务：疯狂读取视频流，只保留最新帧，丢弃所有旧帧
        """
        logger.debug(f"[{self.camera_name}] VideoStreamReader 更新线程已启动")

        self.started = True
        consecutive_failures = 0
        max_consecutive_failures = 10  # 连续失败10次则停止

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(
                        f"[{self.camera_name}] VideoStreamReader: VideoCapture未打开，线程休眠中..."
                    )
                    time.sleep(1)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"[{self.camera_name}] VideoStreamReader: 连续失败{consecutive_failures}次，停止线程"
                        )
                        break
                    continue

                # 读取帧
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # 重置失败计数
                    consecutive_failures = 0

                    # 更新最新帧（覆盖旧帧）
                    with self.lock:
                        # 如果已有旧帧，计为丢弃
                        if self.latest_frame is not None:
                            self.total_drop_count += 1

                        # 保存最新帧和时间戳
                        self.latest_frame = frame
                        self.latest_timestamp = datetime.now()
                        self.total_read_count += 1

                    # 定期输出性能统计（每30秒）
                    current_time = time.time()
                    if current_time - self.last_stats_log_time >= 30:
                        with self.lock:
                            drop_rate = (self.total_drop_count / self.total_read_count * 100) if self.total_read_count > 0 else 0
                            logger.info(
                                f"[{self.camera_name}] VideoStreamReader 统计 - "
                                f"总读取: {self.total_read_count}帧, "
                                f"总丢弃: {self.total_drop_count}帧 ({drop_rate:.1f}%), "
                                f"说明主线程使用了 {100-drop_rate:.1f}% 的帧"
                            )
                            # 重置统计
                            self.total_read_count = 0
                            self.total_drop_count = 0
                            self.last_stats_log_time = current_time
                    time.sleep(0.015)
                else:
                    # 读取失败
                    consecutive_failures += 1
                    logger.debug(
                        f"[{self.camera_name}] VideoStreamReader: 读取帧失败 "
                        f"(连续失败{consecutive_failures}次)"
                    )

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"[{self.camera_name}] VideoStreamReader: "
                            f"连续失败{consecutive_failures}次，停止线程"
                        )
                        break

                    # 失败后短暂休眠
                    time.sleep(0.1)

                # 极短休眠，让出CPU（避免100%占用）
                # 但不能太长，否则会积压缓冲区
                time.sleep(0.001)  # 1ms

            except Exception as e:
                logger.error(f"[{self.camera_name}] VideoStreamReader 更新线程异常: {e}", exc_info=True)
                consecutive_failures += 1
                time.sleep(0.1)

        self.running = False
        logger.info(f"[{self.camera_name}] VideoStreamReader 更新线程已停止")

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        """
        读取最新帧（主线程调用接口）

        Returns:
            Tuple[bool, frame, timestamp]: (是否成功, 帧数据, 时间戳)
        """
        with self.lock:
            if self.latest_frame is not None:
                # 返回最新帧的副本（避免主线程修改影响共享数据）
                return True, self.latest_frame.copy(), self.latest_timestamp
            else:
                return False, None, None

    def is_opened(self) -> bool:
        """检查视频流是否打开"""
        return self.cap is not None and self.cap.isOpened() and self.started

    def release(self):
        """释放资源"""
        logger.info(f"[{self.camera_name}] VideoStreamReader: 正在停止...")

        # 停止更新线程
        self.running = False

        # 等待线程结束（最多5秒）
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning(
                    f"[{self.camera_name}] VideoStreamReader: 更新线程未能在5秒内停止"
                )

        # 释放VideoCapture
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        # 清空最新帧
        with self.lock:
            self.latest_frame = None
            self.latest_timestamp = None

        logger.info(f"[{self.camera_name}] VideoStreamReader: 已释放")

    def __del__(self):
        """析构函数"""
        self.release()
