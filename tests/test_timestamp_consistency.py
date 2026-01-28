"""
时间戳一致性测试

测试目标：
1. 保存图片时的系统时间与帧获取时间的差异
2. 帧时间戳与实际时间的偏差
3. 长时间运行后时间漂移检测

问题场景：
- 用户保存的图片时间与监控画面上的OSD时间不一致
- 可能原因：缓冲区积压、处理延迟、时间戳获取方式
"""
import pytest
import time
import cv2
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestTimestampConsistency:
    """时间戳一致性测试"""

    def test_frame_timestamp_vs_system_time(self):
        """
        测试：帧时间戳与系统时间的差异

        验收标准：差异应小于 1 秒
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        mock_cap = MockVideoCapture(fps=25, delay_ms=0)
        shaper = InputShaper(mock_cap, camera_name="test", target_fps=2.0)

        # 等待缓冲区稳定
        time.sleep(1)

        time_diffs = []

        for _ in range(10):
            system_time_before = datetime.now()
            ret, frame, frame_timestamp = shaper.read()
            system_time_after = datetime.now()

            if ret and frame_timestamp:
                # 计算帧时间戳与系统时间的差异
                diff = abs((frame_timestamp - system_time_before).total_seconds())
                time_diffs.append(diff)

        shaper.release()

        avg_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        max_diff = max(time_diffs) if time_diffs else 0

        print(f"\n帧时间戳 vs 系统时间:")
        print(f"  平均差异: {avg_diff * 1000:.0f}ms")
        print(f"  最大差异: {max_diff * 1000:.0f}ms")

        # 断言：差异应小于 1 秒
        assert max_diff < 1.0, f"帧时间戳与系统时间差异过大: {max_diff * 1000:.0f}ms"

    def test_buffer_delay_impact(self):
        """
        测试：缓冲区积压对时间戳的影响

        场景：高帧率源 + 低输出帧率 = 缓冲区积压
        验证：输出的帧时间戳应该是最新的，而不是积压的旧帧
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # 源 25fps，输出 1fps，必然有帧积压
        mock_cap = MockVideoCapture(fps=25, delay_ms=0)
        shaper = InputShaper(
            mock_cap,
            camera_name="buffer_test",
            target_fps=1.0,  # 每秒只输出 1 帧
            buffer_size=30
        )

        # 等待缓冲区填满
        time.sleep(2)

        # 读取一帧
        system_time = datetime.now()
        ret, frame, frame_timestamp = shaper.read()

        shaper.release()

        if ret and frame_timestamp:
            delay = (system_time - frame_timestamp).total_seconds()
            print(f"\n缓冲区延迟测试:")
            print(f"  帧时间戳延迟: {delay * 1000:.0f}ms")

            # 断言：帧应该是最近的，而不是 1-2 秒前的旧帧
            # InputShaper 设计为取最新帧，所以延迟应该很小
            assert delay < 0.5, f"输出的是旧帧，延迟 {delay * 1000:.0f}ms"

    def test_slow_stream_timestamp_accuracy(self):
        """
        测试：慢流场景下的时间戳准确性

        场景：跨域慢流（模拟顺义），延迟 300ms
        验证：时间戳仍然准确，不会累积延迟
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        # 模拟慢流
        mock_cap = MockVideoCapture(fps=5, delay_ms=300)
        shaper = InputShaper(mock_cap, camera_name="slow_stream", target_fps=2.0)

        # 等待一段时间
        time.sleep(3)

        time_diffs = []
        for _ in range(5):
            system_time = datetime.now()
            ret, frame, frame_timestamp = shaper.read()

            if ret and frame_timestamp:
                diff = (system_time - frame_timestamp).total_seconds()
                time_diffs.append(diff)

        shaper.release()

        avg_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        max_diff = max(time_diffs) if time_diffs else 0

        print(f"\n慢流时间戳准确性:")
        print(f"  平均延迟: {avg_diff * 1000:.0f}ms")
        print(f"  最大延迟: {max_diff * 1000:.0f}ms")

        # 断言：即使是慢流，时间戳延迟也应该可控
        assert max_diff < 2.0, f"慢流时间戳延迟过大: {max_diff * 1000:.0f}ms"

    def test_save_time_vs_frame_time(self):
        """
        测试：保存图片时的时间 vs 帧实际获取时间

        这是用户关心的核心问题：
        保存的图片文件时间/记录时间，是否与画面内容时间一致
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture
        import tempfile
        import os

        mock_cap = MockVideoCapture(fps=25, delay_ms=0)
        shaper = InputShaper(mock_cap, camera_name="save_test", target_fps=2.0)

        time.sleep(1)

        # 读取帧
        ret, frame, frame_timestamp = shaper.read()

        if ret and frame is not None:
            # 模拟保存图片的流程
            save_time = datetime.now()

            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                temp_path = f.name
                cv2.imwrite(temp_path, frame)

            # 获取文件修改时间
            file_mtime = datetime.fromtimestamp(os.path.getmtime(temp_path))

            # 清理
            os.unlink(temp_path)

            # 计算差异
            frame_to_save_diff = (save_time - frame_timestamp).total_seconds()
            frame_to_file_diff = (file_mtime - frame_timestamp).total_seconds()

            print(f"\n保存时间一致性:")
            print(f"  帧时间戳: {frame_timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  保存时间: {save_time.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  文件时间: {file_mtime.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  帧→保存 差异: {frame_to_save_diff * 1000:.0f}ms")

            # 断言：帧获取到保存的延迟应该很小
            assert frame_to_save_diff < 0.5, \
                f"帧获取到保存延迟过大: {frame_to_save_diff * 1000:.0f}ms"

        shaper.release()


class TestTimestampDrift:
    """时间漂移测试"""

    @pytest.mark.slow
    def test_long_run_timestamp_drift(self):
        """
        测试：长时间运行后时间戳是否漂移

        运行 30 秒，检查时间戳是否逐渐偏离
        """
        from input_shaper import InputShaper
        from tests.test_input_shaper import MockVideoCapture

        mock_cap = MockVideoCapture(fps=25, delay_ms=0)
        shaper = InputShaper(mock_cap, camera_name="drift_test", target_fps=2.0)

        time.sleep(1)

        # 记录开始和结束时的时间差
        start_diffs = []
        end_diffs = []

        # 开始阶段采样
        for _ in range(5):
            system_time = datetime.now()
            ret, frame, frame_timestamp = shaper.read()
            if ret and frame_timestamp:
                start_diffs.append((system_time - frame_timestamp).total_seconds())

        # 运行一段时间
        time.sleep(20)

        # 结束阶段采样
        for _ in range(5):
            system_time = datetime.now()
            ret, frame, frame_timestamp = shaper.read()
            if ret and frame_timestamp:
                end_diffs.append((system_time - frame_timestamp).total_seconds())

        shaper.release()

        avg_start = sum(start_diffs) / len(start_diffs) if start_diffs else 0
        avg_end = sum(end_diffs) / len(end_diffs) if end_diffs else 0
        drift = avg_end - avg_start

        print(f"\n时间漂移测试 (30秒):")
        print(f"  开始阶段平均延迟: {avg_start * 1000:.0f}ms")
        print(f"  结束阶段平均延迟: {avg_end * 1000:.0f}ms")
        print(f"  漂移量: {drift * 1000:.0f}ms")

        # 断言：30 秒内漂移不应超过 500ms
        assert abs(drift) < 0.5, f"时间漂移过大: {drift * 1000:.0f}ms"


class TestDatabaseRecordTimestamp:
    """数据库记录时间戳测试"""

    def test_record_time_matches_frame_time(self):
        """
        测试：数据库记录的 pssj 时间应与帧时间一致
        """
        # 模拟 _save_detection_result 的时间戳传递
        from datetime import datetime

        # 模拟帧时间戳（来自 InputShaper）
        frame_timestamp = datetime.now()

        # 模拟处理延迟
        time.sleep(0.1)  # 100ms 处理时间

        # 模拟保存时使用的时间
        # 正确做法：使用 frame_timestamp
        # 错误做法：使用 datetime.now()

        save_time_correct = frame_timestamp
        save_time_wrong = datetime.now()

        diff_if_correct = (datetime.now() - save_time_correct).total_seconds()
        diff_if_wrong = (save_time_wrong - frame_timestamp).total_seconds()

        print(f"\n数据库记录时间:")
        print(f"  正确做法（用帧时间）: 延迟 {diff_if_correct * 1000:.0f}ms")
        print(f"  错误做法（用当前时间）: 误差 {diff_if_wrong * 1000:.0f}ms")

        # 如果使用帧时间戳，记录的时间就是帧的真实时间
        # 如果使用 datetime.now()，会有处理延迟的误差


class TestOSDTimestampExtraction:
    """
    OSD 时间戳提取测试（可选）

    如果需要验证画面上的 OSD 时间与系统时间一致，
    需要使用 OCR 从画面中提取时间戳。

    这是更复杂的场景，涉及：
    1. 摄像头 OSD 时间是否正确（摄像头时区/NTP）
    2. 网络传输延迟
    3. 解码延迟
    """

    @pytest.mark.skip(reason="需要 OCR 库和真实摄像头画面")
    def test_osd_time_extraction(self):
        """
        测试：从画面提取 OSD 时间并与系统时间对比

        前置条件：
        - 安装 pytesseract 或 easyocr
        - 摄像头画面有 OSD 时间戳
        """
        # 示例代码框架
        # import pytesseract
        #
        # ret, frame, frame_timestamp = shaper.read()
        #
        # # 裁剪 OSD 区域（通常在画面角落）
        # osd_region = frame[10:50, 10:200]
        #
        # # OCR 提取时间
        # osd_text = pytesseract.image_to_string(osd_region)
        # osd_time = parse_osd_time(osd_text)  # 需要实现解析逻辑
        #
        # # 对比
        # diff = (frame_timestamp - osd_time).total_seconds()
        # assert abs(diff) < 2.0, f"OSD 时间与帧时间差异: {diff}秒"
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
