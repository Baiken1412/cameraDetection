"""
输入整形层 - 把任意节奏的RTSP流变成稳定输出

核心思想：
1. 拉流线程：尽可能快地拉取帧，放入环形缓冲区
2. 节拍器：以固定频率向下游输出帧
3. 解耦上下游：无论源如何抖动，下游看到的永远是稳定节奏

适用场景：
- RTSP源推流不稳定（突发、卡顿、帧率波动）
- 需要稳定的检测负载
- 多路摄像头资源规划
"""
import cv2
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from loguru import logger
import numpy as np


class InputShaper:
    """
    输入整形层

    功能：
    - 独立线程持续拉流，放入环形缓冲区
    - 以固定节拍向下游输出帧
    - 吸收上游抖动，输出稳定节奏

    配置参数：
    - target_fps: 目标输出帧率（默认2fps，即每0.5秒输出一帧）
    - buffer_size: 环形缓冲区大小（默认30帧）
    - drop_strategy: 丢帧策略 "drop_old"（默认，丢旧保新）
    - empty_behavior: 缓冲区空时行为 "skip"（返回None）或 "repeat"（重复最后一帧）
    """

    def __init__(
        self,
        cap_or_url,
        camera_name: str = "Unknown",
        target_fps: float = 2.0,
        buffer_size: int = 30,
        drop_strategy: str = "drop_old",
        empty_behavior: str = "skip"
    ):
        """
        初始化输入整形层

        Args:
            cap_or_url: cv2.VideoCapture对象、RTSP URL字符串、或USB设备ID
            camera_name: 摄像头名称（用于日志）
            target_fps: 目标输出帧率（Hz）
            buffer_size: 缓冲区大小（帧数）
            drop_strategy: 丢帧策略（"drop_old" 丢旧帧）
            empty_behavior: 缓冲区空时行为（"skip" 跳过 / "repeat" 重复）
        """
        self.camera_name = camera_name
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps if target_fps > 0 else 0.5
        self.buffer_size = buffer_size
        self.drop_strategy = drop_strategy
        self.empty_behavior = empty_behavior

        # 创建或使用 VideoCapture
        if isinstance(cap_or_url, str):
            self.cap = cv2.VideoCapture(cap_or_url)
            self._owns_cap = True
            logger.info(f"[{self.camera_name}] InputShaper: 从URL创建VideoCapture")
        elif isinstance(cap_or_url, int):
            self.cap = cv2.VideoCapture(cap_or_url)
            self._owns_cap = True
            logger.info(f"[{self.camera_name}] InputShaper: 从设备ID {cap_or_url} 创建VideoCapture")
        else:
            self.cap = cap_or_url
            self._owns_cap = False
            logger.info(f"[{self.camera_name}] InputShaper: 使用已有VideoCapture对象")

        # 环形缓冲区：存储 (frame, timestamp)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        # 最后一帧（用于 repeat 模式）
        self.last_frame = None
        self.last_timestamp = None

        # 节拍控制
        self.last_output_time = 0

        # 运行状态
        self.running = True
        self.started = False

        # 统计信息
        self.stats = {
            'frames_pulled': 0,        # 拉取的帧数
            'frames_output': 0,        # 输出的帧数
            'frames_dropped': 0,       # 因缓冲区满而丢弃的帧数
            'buffer_overflows': 0,     # 缓冲区溢出次数
            'empty_reads': 0,          # 缓冲区空时的读取次数
            'repeat_frames': 0,        # 重复帧次数
        }
        self.stats_lock = threading.Lock()
        self.last_stats_log_time = time.time()

        # 启动拉流线程
        self.pull_thread = threading.Thread(
            target=self._pull_loop,
            name=f"InputShaper-Pull-{self.camera_name}",
            daemon=True
        )
        self.pull_thread.start()

        # 等待首帧
        self._wait_for_first_frame()

        logger.info(
            f"[{self.camera_name}] InputShaper 已启动 - "
            f"目标帧率: {target_fps}fps, 缓冲区: {buffer_size}帧, "
            f"丢帧策略: {drop_strategy}, 空缓冲行为: {empty_behavior}"
        )

    def _wait_for_first_frame(self, timeout: float = 5.0):
        """等待第一帧进入缓冲区"""
        start_time = time.time()
        while len(self.buffer) == 0:
            if time.time() - start_time > timeout:
                logger.warning(f"[{self.camera_name}] InputShaper: 等待首帧超时（{timeout}秒）")
                break
            time.sleep(0.01)

        if len(self.buffer) > 0:
            self.started = True
            logger.debug(f"[{self.camera_name}] InputShaper: 首帧已就绪")

    def _pull_loop(self):
        """
        拉流线程：尽可能快地拉取帧，放入缓冲区
        """
        logger.debug(f"[{self.camera_name}] InputShaper 拉流线程已启动")

        consecutive_failures = 0
        max_consecutive_failures = 10

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(f"[{self.camera_name}] InputShaper: VideoCapture未打开")
                    time.sleep(1)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"[{self.camera_name}] InputShaper: 连续失败{consecutive_failures}次，停止线程")
                        break
                    continue

                # 读取帧
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    consecutive_failures = 0
                    timestamp = datetime.now()

                    with self.buffer_lock:
                        # 检查缓冲区是否已满
                        if len(self.buffer) >= self.buffer_size:
                            with self.stats_lock:
                                self.stats['buffer_overflows'] += 1
                                self.stats['frames_dropped'] += 1
                            # deque 自动丢弃最旧的帧（maxlen机制）

                        # 添加到缓冲区
                        self.buffer.append((frame, timestamp))

                        # 更新最后一帧（用于 repeat 模式）
                        self.last_frame = frame
                        self.last_timestamp = timestamp

                    with self.stats_lock:
                        self.stats['frames_pulled'] += 1

                    # 定期输出统计（每30秒）
                    self._log_stats_periodically()

                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"[{self.camera_name}] InputShaper: 连续失败{consecutive_failures}次，停止线程")
                        break
                    time.sleep(0.001)

                # 极短休眠，避免CPU 100%
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"[{self.camera_name}] InputShaper 拉流线程异常: {e}", exc_info=True)
                consecutive_failures += 1
                time.sleep(0.1)

        self.running = False
        logger.info(f"[{self.camera_name}] InputShaper 拉流线程已停止")

    def _log_stats_periodically(self, interval: float = 30.0):
        """定期输出统计信息"""
        current_time = time.time()
        if current_time - self.last_stats_log_time >= interval:
            with self.stats_lock:
                stats_copy = self.stats.copy()
                # 重置统计
                for key in self.stats:
                    self.stats[key] = 0

            with self.buffer_lock:
                buffer_level = len(self.buffer)

            logger.info(
                f"[{self.camera_name}] InputShaper 统计 - "
                f"拉取: {stats_copy['frames_pulled']}帧, "
                f"输出: {stats_copy['frames_output']}帧, "
                f"丢弃: {stats_copy['frames_dropped']}帧, "
                f"溢出: {stats_copy['buffer_overflows']}次, "
                f"空读: {stats_copy['empty_reads']}次, "
                f"重复: {stats_copy['repeat_frames']}次, "
                f"缓冲区: {buffer_level}/{self.buffer_size}"
            )
            self.last_stats_log_time = current_time

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        """
        读取一帧（主线程调用）

        特性：
        - 按目标帧率节拍输出
        - 缓冲区有帧时，取最新帧
        - 缓冲区空时，根据 empty_behavior 处理

        Returns:
            Tuple[bool, frame, timestamp]: (是否成功, 帧数据, 时间戳)
        """
        # 节拍控制：确保输出频率不超过 target_fps
        current_time = time.time()
        elapsed = current_time - self.last_output_time

        if elapsed < self.target_interval:
            sleep_time = self.target_interval - elapsed
            time.sleep(sleep_time)

        self.last_output_time = time.time()

        with self.buffer_lock:
            if len(self.buffer) > 0:
                # 取最新帧（缓冲区尾部）
                frame, timestamp = self.buffer[-1]

                # 清空缓冲区中的旧帧（只保留刚取出的最新状态）
                self.buffer.clear()

                with self.stats_lock:
                    self.stats['frames_output'] += 1

                return True, frame.copy(), timestamp
            else:
                # 缓冲区空
                with self.stats_lock:
                    self.stats['empty_reads'] += 1

                if self.empty_behavior == "repeat" and self.last_frame is not None:
                    # 重复最后一帧
                    with self.stats_lock:
                        self.stats['repeat_frames'] += 1
                    return True, self.last_frame.copy(), self.last_timestamp
                else:
                    # 跳过
                    return False, None, None

    def read_nowait(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        """
        立即读取最新帧（不等待节拍）

        用于需要立即获取帧的场景（如背景学习阶段）

        Returns:
            Tuple[bool, frame, timestamp]: (是否成功, 帧数据, 时间戳)
        """
        with self.buffer_lock:
            if len(self.buffer) > 0:
                frame, timestamp = self.buffer[-1]
                return True, frame.copy(), timestamp
            elif self.last_frame is not None:
                return True, self.last_frame.copy(), self.last_timestamp
            else:
                return False, None, None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            stats_copy = self.stats.copy()
        with self.buffer_lock:
            stats_copy['buffer_level'] = len(self.buffer)
            stats_copy['buffer_size'] = self.buffer_size
        stats_copy['target_fps'] = self.target_fps
        return stats_copy

    def get_buffer_level(self) -> int:
        """获取当前缓冲区水位"""
        with self.buffer_lock:
            return len(self.buffer)

    def is_opened(self) -> bool:
        """检查是否正常运行"""
        return self.cap is not None and self.cap.isOpened() and self.running

    def release(self):
        """释放资源"""
        logger.info(f"[{self.camera_name}] InputShaper: 正在停止...")

        self.running = False

        # 等待拉流线程结束
        if self.pull_thread.is_alive():
            self.pull_thread.join(timeout=5.0)
            if self.pull_thread.is_alive():
                logger.warning(f"[{self.camera_name}] InputShaper: 拉流线程未能在5秒内停止")

        # 释放 VideoCapture（如果是自己创建的）
        if self._owns_cap and self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        # 清空缓冲区
        with self.buffer_lock:
            self.buffer.clear()
            self.last_frame = None
            self.last_timestamp = None

        logger.info(f"[{self.camera_name}] InputShaper: 已释放")

    def __del__(self):
        """析构函数"""
        try:
            self.release()
        except:
            pass
