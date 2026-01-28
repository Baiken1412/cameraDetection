"""
YOLO 池测试用例

测试目标：
1. 池化资源共享正确性
2. 并发安全性
3. 超时处理
4. 一路慢不拖全局
"""
import pytest
import time
import threading
import queue
import numpy as np
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed


class MockDetector:
    """模拟 YOLO 检测器"""

    def __init__(self, delay_ms=50, instance_id=0):
        """
        Args:
            delay_ms: 模拟检测耗时（毫秒）
            instance_id: 实例 ID
        """
        self.delay_ms = delay_ms
        self.instance_id = instance_id
        self.detect_count = 0

    def detect_image(self, frame):
        """模拟检测"""
        time.sleep(self.delay_ms / 1000.0)
        self.detect_count += 1
        # 返回模拟检测结果
        return [{"bbox": [100, 100, 200, 200], "conf": 0.9, "class_id": 0}]


class TestYoloPoolBasic:
    """基础功能测试"""

    def test_pool_initialization(self):
        """测试：池初始化正确"""
        from core.yolo_pool import YoloDetectorPool

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.return_value = MockDetector()

            pool = YoloDetectorPool(pool_size=2, detector_type='adaptive')

            # 断言：创建了正确数量的实例
            assert mock_create.call_count == 2
            assert pool.pool.qsize() == 2

            pool.shutdown()

    def test_detect_returns_result(self):
        """测试：检测返回正确结果"""
        from core.yolo_pool import YoloDetectorPool

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.return_value = MockDetector(delay_ms=10)

            pool = YoloDetectorPool(pool_size=1)

            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pool.detect(frame)

            # 断言：返回检测结果
            assert result is not None
            assert len(result) > 0
            assert "bbox" in result[0]

            pool.shutdown()

    def test_instance_reuse(self):
        """测试：实例被正确复用"""
        from core.yolo_pool import YoloDetectorPool

        detector = MockDetector(delay_ms=10)

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.return_value = detector

            pool = YoloDetectorPool(pool_size=1)

            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # 多次检测
            for _ in range(5):
                pool.detect(frame)

            # 断言：同一个实例被复用
            assert detector.detect_count == 5

            pool.shutdown()


class TestYoloPoolConcurrency:
    """并发安全测试"""

    def test_concurrent_detect_no_deadlock(self):
        """测试：并发检测不死锁"""
        from core.yolo_pool import YoloDetectorPool

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.return_value = MockDetector(delay_ms=50)

            pool = YoloDetectorPool(pool_size=2)

            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = []
            errors = []

            def detect_task(task_id):
                try:
                    result = pool.detect(frame, timeout=5.0)
                    results.append((task_id, result))
                except Exception as e:
                    errors.append((task_id, str(e)))

            # 启动 10 个并发任务（超过池大小）
            threads = []
            for i in range(10):
                t = threading.Thread(target=detect_task, args=(i,))
                threads.append(t)
                t.start()

            # 等待所有任务完成（设置超时）
            for t in threads:
                t.join(timeout=10)

            pool.shutdown()

            # 断言：所有任务完成，无死锁
            assert len(results) == 10, f"只有 {len(results)}/10 任务完成，可能死锁"
            assert len(errors) == 0, f"有 {len(errors)} 个任务失败: {errors}"

    def test_slow_task_does_not_block_others(self):
        """测试：慢任务不阻塞其他任务（核心测试）"""
        from core.yolo_pool import YoloDetectorPool

        # 创建两种检测器：快和慢
        fast_detector = MockDetector(delay_ms=50, instance_id=0)
        slow_detector = MockDetector(delay_ms=500, instance_id=1)

        detector_queue = [fast_detector, slow_detector]

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.side_effect = lambda instance_id: detector_queue[instance_id]

            pool = YoloDetectorPool(pool_size=2)

            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # 并发提交任务
            results = {}

            def task(name, delay_before=0):
                if delay_before > 0:
                    time.sleep(delay_before)
                start = time.time()
                pool.detect(frame, timeout=5.0)
                return name, time.time() - start

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(task, "fast1"),
                    executor.submit(task, "fast2", delay_before=0.1),
                    executor.submit(task, "fast3", delay_before=0.2),
                ]

                for f in as_completed(futures):
                    name, elapsed = f.result()
                    results[name] = elapsed

            pool.shutdown()

            # 断言：快任务应该快速完成，不被慢检测器完全阻塞
            # 至少有一个任务应该在 200ms 内完成
            fast_tasks = [v for v in results.values() if v < 0.3]
            assert len(fast_tasks) > 0, \
                f"所有任务都被拖慢: {results}"


class TestYoloPoolTimeout:
    """超时处理测试"""

    def test_timeout_raises_exception(self):
        """测试：超时应抛出异常"""
        from core.yolo_pool import YoloDetectorPool

        # 创建一个非常慢的检测器
        slow_detector = MockDetector(delay_ms=2000)

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.return_value = slow_detector

            pool = YoloDetectorPool(pool_size=1)

            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # 先占用唯一的实例
            def occupy():
                pool.detect(frame, timeout=10.0)

            t = threading.Thread(target=occupy)
            t.start()
            time.sleep(0.1)  # 确保实例被占用

            # 尝试获取，应该超时
            with pytest.raises(queue.Empty):
                pool.detect(frame, timeout=0.5)

            t.join()
            pool.shutdown()


class TestYoloPoolStats:
    """统计信息测试"""

    def test_stats_accuracy(self):
        """测试：统计信息准确"""
        from core.yolo_pool import YoloDetectorPool

        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_create.return_value = MockDetector(delay_ms=10)

            pool = YoloDetectorPool(pool_size=2)

            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # 执行 10 次检测
            for _ in range(10):
                pool.detect(frame)

            stats = pool.get_stats()

            # 断言：统计正确
            assert stats['total_detections'] == 10
            assert stats['pool_size'] == 2
            assert stats['available_instances'] == 2  # 都应归还

            pool.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
