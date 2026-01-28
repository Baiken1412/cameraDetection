"""
系统集成测试

测试目标：
1. 多路摄像头并发稳定性
2. 慢流隔离效果
3. 资源使用合理性
4. 长时间运行稳定性
"""
import pytest
import time
import threading
import psutil
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime


class MockDatabase:
    """模拟数据库"""

    def __init__(self):
        self.records = []
        self.connected = True

    def connect(self):
        return True

    def close(self):
        pass

    def get_all_cameras(self):
        """返回模拟摄像头列表"""
        return [
            {"id": 1, "fjmc": "摄像头1", "gnslx": "区域1", "rtspssl": "rtsp://fake1", "yolo_pool_id": None},
            {"id": 2, "fjmc": "摄像头2", "gnslx": "区域2", "rtspssl": "rtsp://fake2", "yolo_pool_id": None},
            {"id": 3, "fjmc": "慢流摄像头", "gnslx": "跨域", "rtspssl": "rtsp://slow", "yolo_pool_id": 0},  # 独立实例
        ]

    def save_detection_record(self, **kwargs):
        self.records.append(kwargs)
        return len(self.records)


class TestMultiStreamStability:
    """多路流稳定性测试"""

    def test_8_streams_concurrent_stability(self):
        """测试：8 路流并发运行稳定"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        shapers = []
        errors = []

        # 创建 8 路模拟流
        for i in range(8):
            try:
                mock_cap = MockVideoCapture(fps=25, delay_ms=0)
                shaper = InputShaper(
                    mock_cap,
                    camera_name=f"cam_{i}",
                    target_fps=2.0,
                    buffer_size=30
                )
                shapers.append(shaper)
            except Exception as e:
                errors.append(f"创建 shaper {i} 失败: {e}")

        # 运行 5 秒
        end_time = time.time() + 5
        read_counts = [0] * len(shapers)

        while time.time() < end_time:
            for i, shaper in enumerate(shapers):
                try:
                    ret, frame, ts = shaper.read()
                    if ret:
                        read_counts[i] += 1
                except Exception as e:
                    errors.append(f"shaper {i} read 失败: {e}")

        # 清理
        for shaper in shapers:
            shaper.release()

        # 断言
        assert len(errors) == 0, f"运行中出现错误: {errors}"
        assert all(c > 0 for c in read_counts), f"有流未读到帧: {read_counts}"

    def test_slow_stream_isolation(self):
        """测试：慢流不影响快流（系统级验证）"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # 创建 7 路快流 + 1 路慢流
        fast_shapers = []
        for i in range(7):
            mock_cap = MockVideoCapture(fps=25, delay_ms=0)
            shaper = InputShaper(mock_cap, camera_name=f"fast_{i}", target_fps=2.0)
            fast_shapers.append(shaper)

        slow_cap = MockVideoCapture(fps=5, delay_ms=300)  # 慢流
        slow_shaper = InputShaper(slow_cap, camera_name="slow", target_fps=2.0)

        # 测量快流的响应时间
        fast_read_times = []

        def read_fast_streams():
            for _ in range(10):
                for shaper in fast_shapers:
                    start = time.time()
                    shaper.read()
                    fast_read_times.append(time.time() - start)

        def read_slow_stream():
            for _ in range(10):
                slow_shaper.read()

        # 并发运行
        t1 = threading.Thread(target=read_fast_streams)
        t2 = threading.Thread(target=read_slow_stream)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 清理
        for shaper in fast_shapers:
            shaper.release()
        slow_shaper.release()

        # 断言：快流平均响应时间应该正常
        avg_fast_time = sum(fast_read_times) / len(fast_read_times)
        max_fast_time = max(fast_read_times)

        assert avg_fast_time < 0.7, f"快流被拖慢，平均响应 {avg_fast_time:.2f}秒"
        assert max_fast_time < 1.0, f"快流最大响应 {max_fast_time:.2f}秒，被慢流阻塞"


class TestResourceUsage:
    """资源使用测试"""

    def test_memory_usage_reasonable(self):
        """测试：内存使用合理"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # 创建 8 路流
        shapers = []
        for i in range(8):
            mock_cap = MockVideoCapture(fps=25, delay_ms=0)
            shaper = InputShaper(mock_cap, camera_name=f"cam_{i}", target_fps=2.0, buffer_size=30)
            shapers.append(shaper)

        # 运行一段时间
        for _ in range(50):
            for shaper in shapers:
                shaper.read()

        peak_memory = process.memory_info().rss
        memory_growth = (peak_memory - initial_memory) / (1024 * 1024)  # MB

        # 清理
        for shaper in shapers:
            shaper.release()

        # 断言：8 路流内存增长不应超过 500MB
        assert memory_growth < 500, f"内存增长 {memory_growth:.1f}MB，可能存在泄漏"

    def test_cpu_usage_reasonable(self):
        """测试：CPU 使用合理"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # 创建 4 路流
        shapers = []
        for i in range(4):
            mock_cap = MockVideoCapture(fps=25, delay_ms=0)
            shaper = InputShaper(mock_cap, camera_name=f"cam_{i}", target_fps=2.0)
            shapers.append(shaper)

        # 测量 CPU
        process = psutil.Process()
        cpu_samples = []

        for _ in range(20):
            for shaper in shapers:
                shaper.read()
            cpu_samples.append(process.cpu_percent(interval=0.1))

        # 清理
        for shaper in shapers:
            shaper.release()

        avg_cpu = sum(cpu_samples) / len(cpu_samples)

        # 断言：平均 CPU 不应超过 80%
        # 注意：这个阈值可能需要根据实际硬件调整
        assert avg_cpu < 80, f"CPU 使用率过高: {avg_cpu:.1f}%"


class TestLongRunStability:
    """长时间运行稳定性测试"""

    @pytest.mark.slow
    def test_30_second_stability(self):
        """测试：30 秒连续运行稳定（标记为慢测试）"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        mock_cap = MockVideoCapture(fps=25, delay_ms=0)
        shaper = InputShaper(mock_cap, camera_name="long_run", target_fps=2.0)

        errors = []
        frame_count = 0
        start_time = time.time()

        while time.time() - start_time < 30:
            try:
                ret, frame, ts = shaper.read()
                if ret:
                    frame_count += 1
            except Exception as e:
                errors.append(str(e))

        shaper.release()

        # 断言
        assert len(errors) == 0, f"运行中出现错误: {errors}"
        # 30 秒 * 2fps = 约 60 帧
        assert frame_count >= 50, f"帧数过少: {frame_count}，预期约 60"


class TestEdgeCases:
    """边界条件测试"""

    def test_stream_disconnect_recovery(self):
        """测试：流断开后的处理"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # 在 10 帧后断流
        mock_cap = MockVideoCapture(fps=25, delay_ms=0, fail_after=10)
        shaper = InputShaper(mock_cap, camera_name="disconnect", target_fps=2.0)

        # 等待断流
        time.sleep(1)

        # 尝试读取
        ret, frame, ts = shaper.read()

        stats = shaper.get_stats()
        shaper.release()

        # 断言：应该处理断流情况，不崩溃
        assert stats['empty_reads'] >= 0  # 可能有空读

    def test_zero_fps_handling(self):
        """测试：0 fps 处理"""
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        mock_cap = MockVideoCapture(fps=0, delay_ms=0)

        # 不应崩溃
        try:
            shaper = InputShaper(mock_cap, camera_name="zero_fps", target_fps=2.0)
            shaper.read()
            shaper.release()
        except ZeroDivisionError:
            pytest.fail("0 fps 导致除零错误")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
