"""
背景差分变化检测单元测试
覆盖 specs/背景差分变化检测.md 的核心需求
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBackgroundSubtractorInit:
    """背景建模器初始化测试"""

    def test_mog2_initialization(self):
        """MOG2 背景建模器应正确初始化"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 1,
                'min_change_area': 500,
                'consecutive_frames_threshold': 2,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': True
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 创建测试帧并初始化背景建模器
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector._calculate_background_subtraction(test_frame)

            assert detector.bg_initialized is True
            assert detector.bg_subtractor is not None

    def test_knn_initialization(self):
        """KNN 背景建模器应正确初始化"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 1,
                'min_change_area': 500,
                'consecutive_frames_threshold': 2,
                'bg_subtractor_type': 'KNN',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 400,
                'bg_detect_shadows': True
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector._calculate_background_subtraction(test_frame)

            assert detector.bg_initialized is True
            assert detector.bg_subtractor is not None

    def test_default_config_values(self):
        """默认配置值应符合 specs"""
        import config_loader as config
        cfg = config.RTSP_MONITOR_CONFIG

        assert cfg.get('bg_subtractor_type', 'MOG2') == 'MOG2'
        assert cfg.get('bg_learning_rate', 0.001) == 0.001
        assert cfg.get('bg_history', 500) == 500
        assert cfg.get('bg_detect_shadows', True) is True


class TestChangeThreshold:
    """变化阈值检测测试（change_threshold）"""

    def test_threshold_config(self):
        """change_threshold 配置应为 0.02 (2%)"""
        import config_loader as config
        threshold = config.RTSP_MONITOR_CONFIG.get('change_threshold', 0.02)
        assert threshold == 0.02

    def test_below_threshold_no_change(self):
        """低于阈值的变化不应触发检测"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,  # 2%
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 模拟低变化比例
            with patch.object(detector, '_calculate_background_subtraction', return_value=0.01):
                result = detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))

            assert result is False

    def test_above_threshold_triggers_change(self):
        """高于阈值的变化应触发检测"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,  # 设为1，立即触发
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 模拟高变化比例
            with patch.object(detector, '_calculate_background_subtraction', return_value=0.05):
                result = detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))

            assert result is True


class TestConsecutiveFrames:
    """连续帧确认测试（consecutive_frames_threshold）"""

    def test_consecutive_frames_config(self):
        """consecutive_frames_threshold 配置应为 2"""
        import config_loader as config
        threshold = config.RTSP_MONITOR_CONFIG.get('consecutive_frames_threshold', 2)
        assert threshold == 2

    def test_single_frame_change_not_enough(self):
        """单帧变化不足以触发（需要连续帧）"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 2,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 第一帧有变化
            with patch.object(detector, '_calculate_background_subtraction', return_value=0.05):
                result1 = detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))

            # 第一帧不应触发（因为需要连续2帧）
            assert result1 is False
            assert detector.consecutive_change_count == 1

    def test_consecutive_frames_triggers_change(self):
        """连续帧变化应触发检测"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 2,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            with patch.object(detector, '_calculate_background_subtraction', return_value=0.05):
                # 第一帧
                result1 = detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))
                assert result1 is False

                # 第二帧（连续）
                result2 = detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))
                assert result2 is True

    def test_reset_counter_on_no_change(self):
        """无变化帧应重置计数器"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 2,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 有变化帧
            with patch.object(detector, '_calculate_background_subtraction', return_value=0.05):
                detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))
                assert detector.consecutive_change_count == 1

            # 无变化帧
            with patch.object(detector, '_calculate_background_subtraction', return_value=0.01):
                detector.detect_change(np.zeros((480, 640, 3), dtype=np.uint8))
                assert detector.consecutive_change_count == 0


class TestMinChangeArea:
    """最小变化面积过滤测试（min_change_area）"""

    def test_min_change_area_config(self):
        """min_change_area 配置应为 500 像素"""
        import config_loader as config
        min_area = config.RTSP_MONITOR_CONFIG.get('min_change_area', 500)
        assert min_area == 500

    def test_filter_small_areas(self):
        """小于阈值的变化区域应被过滤"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 500,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 创建一个有小区域变化的掩码
            mask = np.zeros((480, 640), dtype=np.uint8)
            # 添加一个 10x10 的小区域（100 像素 < 500）
            mask[100:110, 100:110] = 255

            filtered = detector._filter_small_areas(mask)

            # 小区域应被过滤
            assert np.count_nonzero(filtered) == 0

    def test_keep_large_areas(self):
        """大于阈值的变化区域应被保留"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 500,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 创建一个有大区域变化的掩码
            mask = np.zeros((480, 640), dtype=np.uint8)
            # 添加一个 30x30 的大区域（900 像素 > 500）
            mask[100:130, 100:130] = 255

            filtered = detector._filter_small_areas(mask)

            # 大区域应被保留
            assert np.count_nonzero(filtered) > 0


class TestMorphologyOperations:
    """形态学操作测试"""

    def test_morphology_open_operation(self):
        """开运算应去除小噪点"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 1,  # 开运算
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 创建带有小噪点的掩码
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[50, 50] = 255  # 单像素噪点
            mask[30:60, 30:60] = 255  # 大区域

            result = detector._apply_morphology(mask)

            # 单像素噪点应被去除，大区域应保留
            assert np.count_nonzero(result) < np.count_nonzero(mask)
            assert np.count_nonzero(result) > 0

    def test_morphology_close_operation(self):
        """闭运算应填充小空洞"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 2,  # 闭运算
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 创建带有小空洞的掩码
            mask = np.ones((100, 100), dtype=np.uint8) * 255
            mask[50, 50] = 0  # 单像素空洞

            result = detector._apply_morphology(mask)

            # 空洞应被填充
            assert np.count_nonzero(result) >= np.count_nonzero(mask)

    def test_no_morphology_operation(self):
        """禁用形态学操作时应原样返回"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,  # 禁用
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
            original_count = np.count_nonzero(mask)

            result = detector._apply_morphology(mask)

            # 应原样返回
            assert np.count_nonzero(result) == original_count


class TestDetectorReset:
    """检测器重置测试"""

    def test_reset_clears_state(self):
        """重置应清除所有状态"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 2,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 初始化背景建模器
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detector._calculate_background_subtraction(test_frame)
            detector.consecutive_change_count = 5
            detector.last_change_ratio = 0.1

            # 重置
            detector.reset()

            assert detector.bg_subtractor is None
            assert detector.bg_initialized is False
            assert detector.consecutive_change_count == 0
            assert detector.last_change_ratio is None


class TestChangeRatioCalculation:
    """变化比例计算测试"""

    def test_calculate_ratio_all_zeros(self):
        """全零掩码应返回 0"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            mask = np.zeros((100, 100), dtype=np.uint8)
            ratio = detector._calculate_ratio(mask)

            assert ratio == 0.0

    def test_calculate_ratio_all_ones(self):
        """全白掩码应返回 1"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            mask = np.ones((100, 100), dtype=np.uint8) * 255
            ratio = detector._calculate_ratio(mask)

            assert ratio == 1.0

    def test_calculate_ratio_partial(self):
        """部分变化应返回正确比例"""
        with patch('image_detection.config') as mock_config:
            mock_config.RTSP_MONITOR_CONFIG = {
                'change_threshold': 0.02,
                'morphology_operation': 0,
                'min_change_area': 0,
                'consecutive_frames_threshold': 1,
                'bg_subtractor_type': 'MOG2',
                'bg_learning_rate': 0.001,
                'bg_history': 500,
                'bg_var_threshold': 10,
                'bg_detect_shadows': False
            }

            from image_detection import ImageChangeDetection
            detector = ImageChangeDetection()

            # 50% 变化
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[:50, :] = 255

            ratio = detector._calculate_ratio(mask)

            assert 0.49 < ratio < 0.51  # 约 50%


class TestLearningRateBehavior:
    """学习率行为测试"""

    def test_learning_rate_config(self):
        """bg_learning_rate 配置应为 0.001"""
        import config_loader as config
        rate = config.RTSP_MONITOR_CONFIG.get('bg_learning_rate', 0.001)
        assert rate == 0.001

    def test_history_config(self):
        """bg_history 配置应为 500"""
        import config_loader as config
        history = config.RTSP_MONITOR_CONFIG.get('bg_history', 500)
        assert history == 500

    def test_var_threshold_config(self):
        """bg_var_threshold 配置应为 10"""
        import config_loader as config
        threshold = config.RTSP_MONITOR_CONFIG.get('bg_var_threshold', 10)
        assert threshold == 10
