"""
慢流问题专项测试

这是当前系统遇到的核心问题：
- 市局本地 8 路摄像头稳定
- 加入顺义 1 路后全局变慢
- 去掉顺义后恢复

此测试用于验证问题是否解决。
"""
import pytest
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestSlowStreamProblem:
    """
    慢流问题验证测试

    测试场景还原：
    - 8 路快流（本地，延迟 ~10ms）
    - 1 路慢流（跨域，延迟 ~300ms）

    验收标准：
    - 快流的 read() 响应时间不应被慢流影响
    - 快流平均响应 < 600ms（目标帧率 2fps = 500ms 间隔）
    - 快流最大响应 < 1000ms
    """

    def test_8_fast_1_slow_isolation(self):
        """
        核心测试：8 路快流 + 1 路慢流，验证隔离效果

        这是 Geoffrey Huntley 方法的核心：
        - 此测试是"裁判"
        - 通过此测试 = 问题解决
        - 失败此测试 = 问题未解决
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # ========== 创建测试环境 ==========
        # 8 路快流（模拟市局本地）
        fast_shapers = []
        for i in range(8):
            mock_cap = MockVideoCapture(fps=25, delay_ms=10)  # 本地，延迟低
            shaper = InputShaper(
                mock_cap,
                camera_name=f"市局_{i}",
                target_fps=2.0,
                buffer_size=30
            )
            fast_shapers.append(shaper)

        # 1 路慢流（模拟顺义跨域）
        slow_cap = MockVideoCapture(fps=5, delay_ms=300)  # 跨域，延迟高
        slow_shaper = InputShaper(
            slow_cap,
            camera_name="顺义",
            target_fps=2.0,
            buffer_size=30
        )

        # ========== 并发读取 ==========
        fast_read_times = {i: [] for i in range(8)}
        slow_read_times = []
        errors = []

        def read_fast(idx, shaper):
            for _ in range(10):
                try:
                    start = time.time()
                    ret, frame, ts = shaper.read()
                    elapsed = time.time() - start
                    fast_read_times[idx].append(elapsed)
                except Exception as e:
                    errors.append(f"fast_{idx}: {e}")

        def read_slow():
            for _ in range(10):
                try:
                    start = time.time()
                    ret, frame, ts = slow_shaper.read()
                    elapsed = time.time() - start
                    slow_read_times.append(elapsed)
                except Exception as e:
                    errors.append(f"slow: {e}")

        # 启动所有线程
        threads = []
        for i, shaper in enumerate(fast_shapers):
            t = threading.Thread(target=read_fast, args=(i, shaper))
            threads.append(t)
        threads.append(threading.Thread(target=read_slow))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # ========== 清理 ==========
        for shaper in fast_shapers:
            shaper.release()
        slow_shaper.release()

        # ========== 验证结果 ==========

        # 1. 无错误
        assert len(errors) == 0, f"运行出现错误: {errors}"

        # 2. 计算快流统计
        all_fast_times = []
        for times in fast_read_times.values():
            all_fast_times.extend(times)

        avg_fast = statistics.mean(all_fast_times)
        max_fast = max(all_fast_times)
        p95_fast = sorted(all_fast_times)[int(len(all_fast_times) * 0.95)]

        print(f"\n快流响应时间统计:")
        print(f"  平均: {avg_fast * 1000:.0f}ms")
        print(f"  最大: {max_fast * 1000:.0f}ms")
        print(f"  P95:  {p95_fast * 1000:.0f}ms")

        # 3. 验收标准
        assert avg_fast < 0.6, \
            f"快流平均响应 {avg_fast * 1000:.0f}ms > 600ms，被慢流拖慢"

        assert max_fast < 1.0, \
            f"快流最大响应 {max_fast * 1000:.0f}ms > 1000ms，被慢流阻塞"

        assert p95_fast < 0.7, \
            f"快流 P95 响应 {p95_fast * 1000:.0f}ms > 700ms，存在抖动"

    def test_without_slow_stream_baseline(self):
        """
        基线测试：仅 8 路快流（无慢流）

        用于对比，确认快流本身性能正常
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        fast_shapers = []
        for i in range(8):
            mock_cap = MockVideoCapture(fps=25, delay_ms=10)
            shaper = InputShaper(mock_cap, camera_name=f"cam_{i}", target_fps=2.0)
            fast_shapers.append(shaper)

        all_times = []

        def read_stream(shaper):
            for _ in range(10):
                start = time.time()
                shaper.read()
                all_times.append(time.time() - start)

        threads = [threading.Thread(target=read_stream, args=(s,)) for s in fast_shapers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for shaper in fast_shapers:
            shaper.release()

        avg_time = statistics.mean(all_times)
        max_time = max(all_times)

        print(f"\n8 路快流基线:")
        print(f"  平均: {avg_time * 1000:.0f}ms")
        print(f"  最大: {max_time * 1000:.0f}ms")

        # 基线应该很好
        assert avg_time < 0.55, f"基线平均响应过高: {avg_time * 1000:.0f}ms"

    def test_slow_stream_independent_yolo(self):
        """
        测试：慢流使用独立 YOLO 实例的隔离效果

        模拟 yolo_pool_id = 0（独立实例）
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture
        from core.yolo_pool import YoloDetectorPool
        from unittest.mock import patch, MagicMock
        import numpy as np

        # 创建快流共享池
        with patch.object(YoloDetectorPool, '_create_detector') as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect_image.return_value = [{"bbox": [0, 0, 100, 100]}]
            mock_create.return_value = mock_detector

            shared_pool = YoloDetectorPool(pool_size=2)

            # 慢流独立检测器
            slow_detector = MagicMock()
            slow_detector.detect_image.side_effect = lambda f: (time.sleep(0.3), [{"bbox": [0, 0, 100, 100]}])[1]

            # 模拟检测场景
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            fast_detect_times = []
            slow_detect_times = []

            def fast_detect():
                for _ in range(5):
                    start = time.time()
                    shared_pool.detect(frame, timeout=5.0)
                    fast_detect_times.append(time.time() - start)

            def slow_detect():
                for _ in range(3):
                    start = time.time()
                    slow_detector.detect_image(frame)
                    slow_detect_times.append(time.time() - start)

            t1 = threading.Thread(target=fast_detect)
            t2 = threading.Thread(target=slow_detect)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            shared_pool.shutdown()

            avg_fast = statistics.mean(fast_detect_times)
            print(f"\n独立 YOLO 测试:")
            print(f"  快流检测平均: {avg_fast * 1000:.0f}ms")

            # 快流不应被慢流检测拖慢
            assert avg_fast < 0.2, f"快流检测被拖慢: {avg_fast * 1000:.0f}ms"


class TestRegressionPrevention:
    """
    回归预防测试

    确保修复后不会再次出现问题
    """

    def test_cap_read_not_blocking_output(self):
        """
        测试：cap.read() 的延迟不应传递到 shaper.read()

        这是问题的根本原因验证
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # 创建一个延迟 500ms 的流
        mock_cap = MockVideoCapture(fps=5, delay_ms=500)
        shaper = InputShaper(mock_cap, camera_name="delayed", target_fps=2.0)

        # 等待缓冲区填充
        time.sleep(2)

        # 测量 read() 响应时间
        read_times = []
        for _ in range(5):
            start = time.time()
            shaper.read()
            read_times.append(time.time() - start)

        shaper.release()

        # read() 应该只受 target_fps 控制，不受源延迟影响
        # target_fps=2.0 → 间隔 500ms
        # 源延迟不应叠加
        avg_read = statistics.mean(read_times)
        assert avg_read < 0.7, \
            f"read() 平均 {avg_read * 1000:.0f}ms，源延迟泄漏到输出"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
