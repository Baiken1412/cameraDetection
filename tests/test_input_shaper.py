"""
InputShaper 测试用例

测试目标：
1. 帧率整形功能正确性
2. 缓冲区溢出处理
3. 慢流不阻塞输出
4. 资源正确释放
"""
import pytest
import time
import threading
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from datetime import datetime


class MockVideoCapture:
    """模拟 VideoCapture，可控制帧率和延迟"""

    def __init__(self, fps=25, delay_ms=0, fail_after=None):
        """
        Args:
            fps: 模拟的源帧率
            delay_ms: 每帧读取的额外延迟（毫秒），模拟网络慢流
            fail_after: 在读取多少帧后开始失败（模拟断流）
        """
        self.fps = fps
        self.delay_ms = delay_ms
        self.fail_after = fail_after
        self.frame_count = 0
        self.opened = True
        self.frame_interval = 1.0 / fps if fps > 0 else 0.04
        self.last_frame_time = time.time()

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened:
            return False, None

        # 模拟帧率间隔
        elapsed = time.time() - self.last_frame_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)

        # 模拟网络延迟
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        self.last_frame_time = time.time()
        self.frame_count += 1

        # 模拟断流
        if self.fail_after and self.frame_count > self.fail_after:
            return False, None

        # 生成测试帧
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self.opened = False

    def set(self, prop, value):
        pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0


class TestInputShaperBasic:
    """基础功能测试"""

    def test_output_fps_matches_target(self):
        """测试：输出帧率应接近目标帧率"""
        from input_shaper import InputShaper

        mock_cap = MockVideoCapture(fps=25, delay_ms=0)
        shaper = InputShaper(
            mock_cap,
            camera_name="test",
            target_fps=2.0,
            buffer_size=30
        )

        # 读取 10 帧，测量实际输出帧率
        start_time = time.time()
        frames_read = 0

        for _ in range(10):
            ret, frame, ts = shaper.read()
            if ret:
                frames_read += 1

        elapsed = time.time() - start_time
        actual_fps = frames_read / elapsed if elapsed > 0 else 0

        shaper.release()

        # 断言：实际帧率应在目标帧率的 ±20% 范围内
        assert 1.6 <= actual_fps <= 2.4, f"实际帧率 {actual_fps:.2f} 超出目标范围 [1.6, 2.4]"

    def test_buffer_overflow_handling(self):
        """测试：缓冲区满时应丢弃旧帧，不阻塞"""
        from input_shaper import InputShaper

        # 源帧率高，输出帧率低，必然导致缓冲区溢出
        mock_cap = MockVideoCapture(fps=30, delay_ms=0)
        shaper = InputShaper(
            mock_cap,
            camera_name="test",
            target_fps=1.0,  # 每秒只输出 1 帧
            buffer_size=10   # 小缓冲区
        )

        # 等待一段时间让缓冲区填满
        time.sleep(2)

        # 检查统计：应有溢出
        stats = shaper.get_stats()

        shaper.release()

        # 断言：应该有溢出记录
        assert stats['buffer_overflows'] > 0 or stats['frames_dropped'] > 0, \
            "高帧率源 + 低输出帧率应导致缓冲区溢出"

    def test_empty_buffer_returns_none(self):
        """测试：缓冲区空时返回 None（skip 模式）"""
        from input_shaper import InputShaper

        # 创建一个会快速失败的 mock
        mock_cap = MockVideoCapture(fps=25, delay_ms=0, fail_after=5)
        shaper = InputShaper(
            mock_cap,
            camera_name="test",
            target_fps=2.0,
            buffer_size=30,
            empty_behavior="skip"
        )

        # 等待源停止
        time.sleep(1)

        # 消耗缓冲区
        for _ in range(50):
            shaper.read()

        # 再读应该返回 None
        ret, frame, ts = shaper.read()

        shaper.release()

        # skip 模式下，空缓冲应返回 False
        # 注意：如果 last_frame 存在且是 repeat 模式，行为不同
        assert ret == False or frame is None, "空缓冲区 skip 模式应返回 None"


class TestInputShaperSlowStream:
    """慢流隔离测试 - 这是当前系统的核心问题"""

    def test_slow_stream_does_not_block_output(self):
        """测试：慢流不应阻塞输出（核心测试）"""
        from input_shaper import InputShaper

        # 模拟跨域慢流：每帧延迟 500ms
        mock_cap = MockVideoCapture(fps=5, delay_ms=500)
        shaper = InputShaper(
            mock_cap,
            camera_name="slow_stream",
            target_fps=2.0,
            buffer_size=30
        )

        # 测量 read() 的响应时间
        read_times = []
        for _ in range(5):
            start = time.time()
            ret, frame, ts = shaper.read()
            read_time = time.time() - start
            read_times.append(read_time)

        shaper.release()

        avg_read_time = sum(read_times) / len(read_times)
        max_read_time = max(read_times)

        # 断言：read() 不应被源的延迟阻塞
        # 目标帧率 2fps = 0.5秒间隔，read() 应该在 0.5-0.6 秒内返回
        # 而不是被源的 500ms 延迟累加
        assert max_read_time < 1.0, \
            f"read() 最大耗时 {max_read_time:.2f}秒，慢流延迟泄漏到输出层"

    def test_multiple_streams_isolation(self):
        """测试：多路流之间相互隔离"""
        from input_shaper import InputShaper

        # 创建一路快流和一路慢流
        fast_cap = MockVideoCapture(fps=25, delay_ms=0)
        slow_cap = MockVideoCapture(fps=5, delay_ms=300)

        fast_shaper = InputShaper(fast_cap, camera_name="fast", target_fps=2.0)
        slow_shaper = InputShaper(slow_cap, camera_name="slow", target_fps=2.0)

        # 并发读取
        fast_times = []
        slow_times = []

        def read_fast():
            for _ in range(5):
                start = time.time()
                fast_shaper.read()
                fast_times.append(time.time() - start)

        def read_slow():
            for _ in range(5):
                start = time.time()
                slow_shaper.read()
                slow_times.append(time.time() - start)

        t1 = threading.Thread(target=read_fast)
        t2 = threading.Thread(target=read_slow)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        fast_shaper.release()
        slow_shaper.release()

        # 断言：快流不应被慢流拖慢
        fast_avg = sum(fast_times) / len(fast_times)
        assert fast_avg < 0.7, f"快流被慢流拖慢: 平均 read 时间 {fast_avg:.2f}秒"


class TestInputShaperResource:
    """资源管理测试"""

    def test_release_stops_pull_thread(self):
        """测试：release() 应停止拉流线程"""
        from input_shaper import InputShaper

        mock_cap = MockVideoCapture(fps=25)
        shaper = InputShaper(mock_cap, camera_name="test", target_fps=2.0)

        # 确认线程在运行
        assert shaper.pull_thread.is_alive(), "拉流线程应该在运行"

        # 释放资源
        shaper.release()

        # 等待线程结束
        time.sleep(0.5)

        # 断言：线程应该停止
        assert not shaper.pull_thread.is_alive(), "release() 后拉流线程应该停止"

    def test_no_memory_leak_on_long_run(self):
        """测试：长时间运行不应内存泄漏"""
        import tracemalloc
        from input_shaper import InputShaper

        tracemalloc.start()

        mock_cap = MockVideoCapture(fps=25)
        shaper = InputShaper(mock_cap, camera_name="test", target_fps=2.0, buffer_size=30)

        # 记录初始内存
        snapshot1 = tracemalloc.take_snapshot()

        # 运行一段时间
        for _ in range(100):
            shaper.read()

        # 记录结束内存
        snapshot2 = tracemalloc.take_snapshot()

        shaper.release()
        tracemalloc.stop()

        # 比较内存增长
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats)

        # 断言：内存增长不应超过 10MB
        assert total_diff < 10 * 1024 * 1024, \
            f"内存增长 {total_diff / 1024 / 1024:.2f}MB，可能存在泄漏"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
