"""
YOLO检测器实例池
支持多个摄像头共享有限数量的YOLO实例，提高资源利用率
"""

import threading
import queue
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class YoloDetectorPool:
    """
    YOLO检测器实例池

    功能：
    - 管理多个YOLO检测器实例
    - 提供线程安全的实例借用/归还机制
    - 支持自适应检测器(AdaptivePersonDetector)和传统检测器(PersonDetector)
    - 统计使用情况（等待时间、使用次数等）

    使用示例：
        pool = YoloDetectorPool(pool_size=2, detector_type='adaptive')
        detections = pool.detect(frame)  # 自动获取空闲实例并归还
    """

    def __init__(
        self,
        pool_size: int = 2,
        detector_type: str = 'adaptive',  # 'adaptive' or 'traditional'
        detector_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化YOLO实例池

        Args:
            pool_size: 池中YOLO实例的数量
            detector_type: 检测器类型 ('adaptive' 使用AdaptivePersonDetector, 'traditional' 使用PersonDetector)
            detector_config: 检测器配置参数字典
        """
        self.pool_size = pool_size
        self.detector_type = detector_type
        self.detector_config = detector_config or {}

        # 使用Queue实现线程安全的实例池
        self.pool = queue.Queue(maxsize=pool_size)

        # 统计信息
        self.stats = {
            'total_detections': 0,           # 总检测次数
            'total_wait_time': 0.0,          # 总等待时间(秒)
            'max_wait_time': 0.0,            # 最大等待时间(秒)
            'instance_usage': [0] * pool_size,  # 每个实例的使用次数
            'created_at': datetime.now()
        }
        self.stats_lock = threading.Lock()

        # 初始化检测器实例
        logger.info(f"正在初始化YOLO实例池，大小: {pool_size}, 类型: {detector_type}")
        for i in range(pool_size):
            try:
                detector = self._create_detector(instance_id=i)
                self.pool.put((i, detector))  # (实例ID, 检测器对象)
                logger.info(f"YOLO实例 {i} 创建成功")
            except Exception as e:
                logger.error(f"创建YOLO实例 {i} 失败: {e}")
                raise

        logger.info(f"YOLO实例池初始化完成，可用实例: {self.pool.qsize()}/{pool_size}")

    def _create_detector(self, instance_id: int):
        """
        创建单个YOLO检测器实例

        Args:
            instance_id: 实例ID（用于日志标识）

        Returns:
            检测器对象
        """
        if self.detector_type == 'adaptive':
            # 使用自适应检测器（YOLOv11 + CPU优化）
            from core.person_detector_adaptive import AdaptivePersonDetector

            detector = AdaptivePersonDetector(
                model_dir=self.detector_config.get('model_dir', 'models'),
                conf_threshold=self.detector_config.get('conf_threshold', 0.5),
                iou_threshold=self.detector_config.get('iou_threshold', 0.4),
                force_engine=self.detector_config.get('force_engine'),
                num_threads=self.detector_config.get('num_threads')
            )
            logger.debug(f"实例 {instance_id} 使用 AdaptivePersonDetector, 引擎: {detector.engine.value}")

        elif self.detector_type == 'traditional':
            # 使用传统检测器（YOLOv8）
            from core.detector import PersonDetector

            detector = PersonDetector(
                model_path=self.detector_config.get('model_path'),
                device=self.detector_config.get('device', 'auto'),
                conf_threshold=self.detector_config.get('conf_threshold', 0.5),
                iou_threshold=self.detector_config.get('iou_threshold', 0.4)
            )
            logger.debug(f"实例 {instance_id} 使用 PersonDetector")

        else:
            raise ValueError(f"不支持的检测器类型: {self.detector_type}, 仅支持 'adaptive' 或 'traditional'")

        return detector

    def detect(self, frame, timeout: float = 30.0) -> List[Dict]:
        """
        使用池中的YOLO实例进行检测（自动管理实例的借用和归还）

        Args:
            frame: 输入图像帧 (numpy.ndarray)
            timeout: 等待空闲实例的超时时间(秒)，默认30秒

        Returns:
            检测结果列表: [{'bbox': [x1, y1, x2, y2], 'conf': 0.95, 'class_id': 0}, ...]

        Raises:
            queue.Empty: 如果在timeout时间内无法获取到空闲实例
        """
        # 记录等待开始时间
        wait_start = time.time()

        # 从池中获取空闲实例（阻塞等待）
        try:
            instance_id, detector = self.pool.get(timeout=timeout)
        except queue.Empty:
            logger.error(f"等待YOLO实例超时 ({timeout}秒)，池大小: {self.pool_size}")
            raise

        # 记录等待时间
        wait_time = time.time() - wait_start

        try:
            # 执行检测
            detections = detector.detect_image(frame)

            # 更新统计信息
            with self.stats_lock:
                self.stats['total_detections'] += 1
                self.stats['total_wait_time'] += wait_time
                self.stats['max_wait_time'] = max(self.stats['max_wait_time'], wait_time)
                self.stats['instance_usage'][instance_id] += 1

            # 记录日志（仅在等待时间超过1秒时）
            if wait_time > 1.0:
                logger.warning(f"YOLO实例 {instance_id} 等待时间: {wait_time:.2f}秒")

            return detections

        finally:
            # 归还实例到池中（无论是否发生异常）
            self.pool.put((instance_id, detector))

    def acquire(self, timeout: float = 30.0):
        """
        借用一个YOLO实例（需要手动调用release归还）

        适用于需要多次检测或者需要控制实例生命周期的场景

        Args:
            timeout: 等待空闲实例的超时时间(秒)

        Returns:
            (instance_id, detector): 实例ID和检测器对象

        Example:
            instance_id, detector = pool.acquire()
            try:
                result1 = detector.detect_image(frame1)
                result2 = detector.detect_image(frame2)
            finally:
                pool.release(instance_id, detector)
        """
        wait_start = time.time()
        instance_id, detector = self.pool.get(timeout=timeout)

        wait_time = time.time() - wait_start
        if wait_time > 1.0:
            logger.warning(f"借用YOLO实例 {instance_id} 等待时间: {wait_time:.2f}秒")

        return instance_id, detector

    def release(self, instance_id: int, detector):
        """
        归还YOLO实例到池中

        Args:
            instance_id: 实例ID
            detector: 检测器对象
        """
        self.pool.put((instance_id, detector))

    def get_stats(self) -> Dict[str, Any]:
        """
        获取池的统计信息

        Returns:
            统计信息字典
        """
        with self.stats_lock:
            avg_wait_time = (
                self.stats['total_wait_time'] / self.stats['total_detections']
                if self.stats['total_detections'] > 0
                else 0.0
            )

            return {
                'pool_size': self.pool_size,
                'detector_type': self.detector_type,
                'available_instances': self.pool.qsize(),
                'total_detections': self.stats['total_detections'],
                'avg_wait_time': round(avg_wait_time, 3),
                'max_wait_time': round(self.stats['max_wait_time'], 3),
                'instance_usage': self.stats['instance_usage'].copy(),
                'uptime_seconds': (datetime.now() - self.stats['created_at']).total_seconds()
            }

    def shutdown(self):
        """
        关闭实例池，清理所有YOLO实例
        """
        logger.info("正在关闭YOLO实例池...")

        # 清空队列中的所有实例
        while not self.pool.empty():
            try:
                instance_id, detector = self.pool.get_nowait()
                # 如果检测器有cleanup方法，调用它
                if hasattr(detector, 'cleanup'):
                    detector.cleanup()
                logger.debug(f"YOLO实例 {instance_id} 已清理")
            except queue.Empty:
                break

        logger.info("YOLO实例池已关闭")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器，自动清理资源"""
        self.shutdown()

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"YoloDetectorPool(size={stats['pool_size']}, "
            f"type={stats['detector_type']}, "
            f"available={stats['available_instances']}, "
            f"detections={stats['total_detections']})"
        )
