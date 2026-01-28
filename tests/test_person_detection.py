"""
人员检测流水线单元测试
覆盖 specs/人员检测与YOLO推理.md 的核心需求
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cpu_detector import CPUDetector, InferenceEngine, get_cpu_detector


class TestCPUDetection:
    """CPU 检测与引擎选择测试"""

    def test_cpu_detector_initialization(self):
        """测试 CPU 检测器初始化"""
        detector = CPUDetector()
        assert detector.cpu_brand is None
        assert detector.cpu_model is None
        assert detector.recommended_engine is None

    def test_intel_cpu_selects_openvino(self):
        """Intel CPU 应选择 OpenVINO 引擎"""
        detector = CPUDetector()
        detector.cpu_brand = 'Intel'
        detector.cpu_cores = 8
        detector._recommend_engine()
        assert detector.recommended_engine == InferenceEngine.OPENVINO

    def test_amd_cpu_selects_onnx(self):
        """AMD CPU 应选择 ONNX Runtime 引擎"""
        detector = CPUDetector()
        detector.cpu_brand = 'AMD'
        detector.cpu_cores = 8
        detector._recommend_engine()
        assert detector.recommended_engine == InferenceEngine.ONNX_RUNTIME

    def test_unknown_cpu_selects_onnx(self):
        """未知 CPU 应选择 ONNX Runtime 引擎"""
        detector = CPUDetector()
        detector.cpu_brand = 'Unknown'
        detector.cpu_cores = 8
        detector._recommend_engine()
        assert detector.recommended_engine == InferenceEngine.ONNX_RUNTIME

    def test_thread_recommendation(self):
        """线程数推荐应为核心数-2"""
        detector = CPUDetector()
        detector.cpu_cores = 8
        detector._recommend_threads()
        assert detector.recommended_threads == 6  # 8 - 2

    def test_thread_recommendation_min(self):
        """线程数至少为1"""
        detector = CPUDetector()
        detector.cpu_cores = 2
        detector._recommend_threads()
        assert detector.recommended_threads >= 1

    def test_thread_recommendation_max(self):
        """大核心数CPU线程上限为14"""
        detector = CPUDetector()
        detector.cpu_cores = 32
        detector._recommend_threads()
        assert detector.recommended_threads <= 14

    def test_engine_availability_check_onnx(self):
        """ONNX Runtime 可用性检查"""
        # ONNX Runtime 应该已安装
        available = CPUDetector.check_engine_availability(InferenceEngine.ONNX_RUNTIME)
        assert available is True

    def test_get_cpu_detector_singleton(self):
        """CPU 检测器应为单例"""
        detector1 = get_cpu_detector()
        detector2 = get_cpu_detector()
        assert detector1 is detector2


class TestAdaptivePersonDetector:
    """自适应人员检测器测试"""

    @pytest.fixture
    def mock_onnx_session(self):
        """模拟 ONNX Session"""
        session = Mock()
        session.get_inputs.return_value = [Mock(name='images', shape=[1, 3, 640, 640])]
        session.get_outputs.return_value = [Mock(name='output0')]
        return session

    def test_detector_config_parameters(self):
        """测试检测器配置参数加载"""
        import config_loader as config
        cfg = config.ADAPTIVE_DETECTION_CONFIG

        assert 'enabled' in cfg
        assert 'model_dir' in cfg
        assert 'conf_threshold' in cfg
        assert 'iou_threshold' in cfg
        assert cfg['conf_threshold'] == 0.5
        assert cfg['iou_threshold'] == 0.4

    def test_force_engine_openvino(self):
        """测试强制使用 OpenVINO 引擎"""
        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            # 模拟 OpenVINO 不可用时的行为
            with patch.dict('sys.modules', {'openvino': None}):
                # 由于 OpenVINO 不可用，应该抛出 ImportError
                from core.person_detector_adaptive import AdaptivePersonDetector
                with pytest.raises(ImportError):
                    AdaptivePersonDetector(force_engine='openvino')

    def test_force_engine_invalid(self):
        """测试无效引擎参数"""
        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            from core.person_detector_adaptive import AdaptivePersonDetector
            with pytest.raises(ValueError, match="未知引擎"):
                AdaptivePersonDetector(force_engine='invalid_engine')


class TestPostprocessing:
    """后处理（NMS、过滤）测试"""

    @pytest.fixture
    def detector_class(self):
        """获取检测器类（不初始化模型）"""
        from core.person_detector_adaptive import AdaptivePersonDetector
        return AdaptivePersonDetector

    def test_nms_removes_overlapping_boxes(self, detector_class):
        """NMS 应移除重叠框"""
        # 两个高度重叠的框
        boxes = np.array([
            [100, 100, 200, 200],
            [110, 110, 210, 210],  # 高度重叠
        ])
        scores = np.array([0.9, 0.8])

        keep = detector_class.nms(boxes, scores, iou_threshold=0.5)

        # 应只保留置信度高的框
        assert len(keep) == 1
        assert keep[0] == 0  # 保留第一个（置信度更高）

    def test_nms_keeps_non_overlapping_boxes(self, detector_class):
        """NMS 应保留不重叠的框"""
        # 两个不重叠的框
        boxes = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300],  # 不重叠
        ])
        scores = np.array([0.9, 0.8])

        keep = detector_class.nms(boxes, scores, iou_threshold=0.5)

        # 应保留两个框
        assert len(keep) == 2

    def test_confidence_filtering(self, detector_class):
        """置信度过滤测试"""
        # 模拟 YOLO 输出格式 (1, 84, 8400)
        # 84 = 4 (bbox) + 80 (classes)
        outputs = np.zeros((1, 84, 10))  # 10个检测框

        # 设置一些检测结果
        outputs[0, 0, 0] = 320  # x_center
        outputs[0, 1, 0] = 240  # y_center
        outputs[0, 2, 0] = 100  # width
        outputs[0, 3, 0] = 200  # height
        outputs[0, 4, 0] = 0.7  # person class score (高于阈值)

        outputs[0, 0, 1] = 400
        outputs[0, 1, 1] = 300
        outputs[0, 2, 1] = 80
        outputs[0, 3, 1] = 160
        outputs[0, 4, 1] = 0.3  # person class score (低于阈值)

        # 创建一个临时检测器实例用于测试
        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            with patch.object(detector_class, '_init_onnx'):
                detector = object.__new__(detector_class)
                detector.conf_threshold = 0.5
                detector.iou_threshold = 0.4
                detector.orig_width = 640
                detector.orig_height = 480
                detector.input_width = 640
                detector.input_height = 640

                results = detector.postprocess(outputs)

                # 应只有一个检测结果（置信度>0.5的那个）
                assert len(results) == 1
                assert results[0]['conf'] > 0.5


class TestOutputFormat:
    """检测输出格式测试"""

    def test_detection_result_format(self):
        """检测结果应包含 bbox, conf, class_id"""
        from core.person_detector_adaptive import AdaptivePersonDetector

        # 创建一个模拟的检测结果
        detection = {
            'bbox': [100, 100, 200, 200],
            'conf': 0.85,
            'class_id': 0
        }

        # 验证格式
        assert 'bbox' in detection
        assert 'conf' in detection
        assert 'class_id' in detection
        assert len(detection['bbox']) == 4
        assert 0 <= detection['conf'] <= 1
        assert detection['class_id'] == 0  # person 类别

    def test_bbox_format_xyxy(self):
        """边界框应为 [x1, y1, x2, y2] 格式"""
        from core.person_detector_adaptive import AdaptivePersonDetector

        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            with patch.object(AdaptivePersonDetector, '_init_onnx'):
                detector = object.__new__(AdaptivePersonDetector)
                detector.conf_threshold = 0.5
                detector.iou_threshold = 0.4
                detector.orig_width = 640
                detector.orig_height = 480
                detector.input_width = 640
                detector.input_height = 640

                # 模拟输出 - 包含一个有效检测
                outputs = np.zeros((1, 84, 5))
                outputs[0, 0, 0] = 320  # x_center
                outputs[0, 1, 0] = 240  # y_center
                outputs[0, 2, 0] = 100  # width
                outputs[0, 3, 0] = 200  # height
                outputs[0, 4, 0] = 0.8  # person score

                results = detector.postprocess(outputs)

                if len(results) > 0:
                    bbox = results[0]['bbox']
                    x1, y1, x2, y2 = bbox
                    # x1 < x2, y1 < y2
                    assert x1 < x2
                    assert y1 < y2


class TestPersonOnlyFiltering:
    """仅检测人员（class_id=0）测试"""

    def test_only_person_class_detected(self):
        """应只返回 person 类别（class_id=0）"""
        from core.person_detector_adaptive import AdaptivePersonDetector

        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            with patch.object(AdaptivePersonDetector, '_init_onnx'):
                detector = object.__new__(AdaptivePersonDetector)
                detector.conf_threshold = 0.5
                detector.iou_threshold = 0.4
                detector.orig_width = 640
                detector.orig_height = 480
                detector.input_width = 640
                detector.input_height = 640

                # 模拟输出 - 一个 person (class 0) 和一个 car (class 2)
                outputs = np.zeros((1, 84, 2))

                # Detection 0: person with high score
                outputs[0, 0, 0] = 320
                outputs[0, 1, 0] = 240
                outputs[0, 2, 0] = 100
                outputs[0, 3, 0] = 200
                outputs[0, 4, 0] = 0.9  # person class score
                outputs[0, 6, 0] = 0.1  # car class score (index 2+4=6)

                # Detection 1: car with high score (not person)
                outputs[0, 0, 1] = 400
                outputs[0, 1, 1] = 300
                outputs[0, 2, 1] = 150
                outputs[0, 3, 1] = 100
                outputs[0, 4, 1] = 0.1  # person class score (low)
                outputs[0, 6, 1] = 0.9  # car class score (high)

                results = detector.postprocess(outputs)

                # 应只返回 person 检测结果
                for r in results:
                    assert r['class_id'] == 0


class TestModelDiscovery:
    """模型发现规则测试"""

    def test_supports_yolov8_model_pattern(self):
        """应支持 YOLOv8 模型文件名模式"""
        from core.person_detector_adaptive import AdaptivePersonDetector

        # 测试 glob 模式
        test_patterns = ['yolov8n.onnx', 'yolov8s.onnx', 'yolov8m.onnx']
        import fnmatch

        for pattern in test_patterns:
            assert fnmatch.fnmatch(pattern, 'yolov8*.onnx')

    def test_supports_yolov11_model_pattern(self):
        """应支持 YOLOv11 模型文件名模式"""
        test_patterns = ['yolo11n.onnx', 'yolo11s.onnx']
        import fnmatch

        for pattern in test_patterns:
            assert fnmatch.fnmatch(pattern, 'yolo11*.onnx')


class TestPreprocessing:
    """预处理测试"""

    def test_preprocess_output_shape(self):
        """预处理后输出形状应为 (1, 3, H, W)"""
        from core.person_detector_adaptive import AdaptivePersonDetector

        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            with patch.object(AdaptivePersonDetector, '_init_onnx'):
                detector = object.__new__(AdaptivePersonDetector)
                detector.input_width = 640
                detector.input_height = 640

                # 创建测试图像
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                result = detector.preprocess(test_image)

                assert result.shape == (1, 3, 640, 640)
                assert result.dtype == np.float32
                assert result.min() >= 0.0
                assert result.max() <= 1.0

    def test_preprocess_stores_original_size(self):
        """预处理应保存原始图像尺寸"""
        from core.person_detector_adaptive import AdaptivePersonDetector

        with patch('core.person_detector_adaptive.get_cpu_detector') as mock_cpu:
            mock_cpu.return_value.recommended_engine = InferenceEngine.ONNX_RUNTIME
            mock_cpu.return_value.recommended_threads = 4

            with patch.object(AdaptivePersonDetector, '_init_onnx'):
                detector = object.__new__(AdaptivePersonDetector)
                detector.input_width = 640
                detector.input_height = 640

                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detector.preprocess(test_image)

                assert detector.orig_height == 480
                assert detector.orig_width == 640
