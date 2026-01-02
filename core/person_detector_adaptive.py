"""
自适应人员检测器 - YOLOv11
自动选择最优推理引擎（OpenVINO / ONNX Runtime）
专注于人员检测和计数，弃用 ReID
"""
import logging
from typing import List, Dict, Union
from pathlib import Path
import cv2
import numpy as np
from utils.cpu_detector import get_cpu_detector, InferenceEngine
from utils.image_processor import load_image

logger = logging.getLogger(__name__)


class AdaptivePersonDetector:
    """
    自适应人员检测器
    - 自动检测 CPU 类型
    - Intel CPU → OpenVINO
    - AMD/其他 CPU → ONNX Runtime
    - 只检测人员（class_id=0）
    """

    def __init__(
        self,
        model_dir: str = "models",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        force_engine: str = None,  # 强制使用指定引擎: 'openvino' 或 'onnx'
        num_threads: int = None
    ):
        """
        初始化自适应检测器

        Args:
            model_dir: 模型目录
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU 阈值
            force_engine: 强制使用指定引擎
            num_threads: 线程数（None=自动）
        """
        self.model_dir = Path(model_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # CPU 自动检测
        cpu_detector = get_cpu_detector()

        # 确定使用的引擎
        if force_engine:
            if force_engine == 'openvino':
                self.engine = InferenceEngine.OPENVINO
            elif force_engine == 'onnx':
                self.engine = InferenceEngine.ONNX_RUNTIME
            else:
                raise ValueError(f"未知引擎: {force_engine}")
            logger.info(f"强制使用推理引擎: {self.engine.value}")
        else:
            self.engine = cpu_detector.recommended_engine
            logger.info(f"自动选择推理引擎: {self.engine.value}")

        # 确定线程数
        if num_threads is None:
            num_threads = cpu_detector.recommended_threads

        self.num_threads = num_threads
        logger.info(f"使用线程数: {num_threads}")

        # 加载对应的检测器
        if self.engine == InferenceEngine.OPENVINO:
            self._init_openvino()
        else:
            self._init_onnx()

        logger.info("人员检测器初始化完成")

    def _init_openvino(self):
        """初始化 OpenVINO 推理引擎"""
        try:
            from openvino import Core  # 新版 OpenVINO 导入方式
        except ImportError:
            raise ImportError(
                "OpenVINO 未安装，请运行: pip install openvino\n"
                "或者在 config.py 中设置 force_engine='onnx' 使用 ONNX Runtime"
            )

        # 查找 OpenVINO 模型文件
        # OpenVINO IR 格式: .xml + .bin
        # 支持两种路径：直接在 models/ 或在 models/*_openvino_model/ 子目录中
        xml_files = list(self.model_dir.glob("yolo11*.xml"))
        if not xml_files:
            # 尝试在 openvino_model 子目录中查找
            xml_files = list(self.model_dir.glob("*_openvino_model/*.xml"))

        if not xml_files:
            raise FileNotFoundError(
                f"未找到 OpenVINO 模型文件（.xml）在 {self.model_dir}\n"
                f"请运行模型转换脚本生成 OpenVINO IR 格式模型"
            )

        model_path = xml_files[0]
        logger.info(f"加载 OpenVINO 模型: {model_path}")

        # 创建 OpenVINO Core
        self.core = Core()

        # 读取模型
        self.model = self.core.read_model(model=str(model_path))

        # 编译模型
        # OpenVINO 2025.x 使用新的配置方式
        compile_config = {}
        if self.num_threads:
            # 新版 OpenVINO 使用 INFERENCE_NUM_THREADS
            compile_config["INFERENCE_NUM_THREADS"] = str(self.num_threads)

        self.compiled_model = self.core.compile_model(
            model=self.model,
            device_name="CPU",
            config=compile_config
        )

        # 获取输入输出层
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 获取输入尺寸
        self.input_shape = self.input_layer.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        logger.info(f"OpenVINO 模型加载完成，输入尺寸: {self.input_width}x{self.input_height}")

    def _init_onnx(self):
        """初始化 ONNX Runtime 推理引擎"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime 未安装，请运行: pip install onnxruntime")

        # 查找 ONNX 模型文件
        onnx_files = list(self.model_dir.glob("yolo11*.onnx"))

        if not onnx_files:
            raise FileNotFoundError(
                f"未找到 ONNX 模型文件（.onnx）在 {self.model_dir}\n"
                f"请运行模型转换脚本生成 ONNX 格式模型"
            )

        model_path = onnx_files[0]
        logger.info(f"加载 ONNX 模型: {model_path}")

        # 会话选项
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 创建会话
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 获取输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        logger.info(f"ONNX Runtime 模型加载完成，输入尺寸: {self.input_width}x{self.input_height}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 保存原始尺寸
        self.orig_height, self.orig_width = image.shape[:2]

        # Resize
        img = cv2.resize(image, (self.input_width, self.input_height))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, outputs: np.ndarray) -> List[Dict]:
        """
        后处理 YOLO 输出

        Args:
            outputs: YOLO 模型输出

        Returns:
            检测结果列表 [{'bbox': [x1,y1,x2,y2], 'conf': 0.95, 'class_id': 0}, ...]
        """
        # YOLOv11 输出格式同 YOLOv8: (1, 84, 8400)
        predictions = outputs.squeeze(0).T  # (8400, 84)

        # 提取 bbox 和 scores
        boxes = predictions[:, :4]  # (8400, 4) - [x_center, y_center, w, h]
        scores = predictions[:, 4:]  # (8400, 80) - class scores

        # 只保留 person 类别 (class_id=0)
        person_scores = scores[:, 0]

        # 过滤低置信度
        mask = person_scores > self.conf_threshold
        boxes = boxes[mask]
        person_scores = person_scores[mask]

        if len(boxes) == 0:
            return []

        # 转换 bbox 格式: [x_center, y_center, w, h] -> [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # 缩放到原始图像尺寸
        boxes_xyxy[:, [0, 2]] *= self.orig_width / self.input_width
        boxes_xyxy[:, [1, 3]] *= self.orig_height / self.input_height

        # NMS
        indices = self.nms(boxes_xyxy, person_scores, self.iou_threshold)

        # 构建检测结果
        detections = []
        for idx in indices:
            detection = {
                'bbox': boxes_xyxy[idx].tolist(),
                'conf': float(person_scores[idx]),
                'class_id': 0
            }
            detections.append(detection)

        return detections

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image_path: Union[str, Path]) -> List[Dict]:
        """
        检测单张图片中的人员

        Args:
            image_path: 图片路径

        Returns:
            检测结果列表
        """
        # 读取图片（支持中文路径）
        try:
            image = load_image(str(image_path))
        except Exception as e:
            logger.error(f"无法读取图片: {image_path}, 错误: {e}")
            return []

        return self.detect_image(image)

    def detect_image(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人员

        Args:
            image: BGR 图像（numpy array）

        Returns:
            检测结果列表
        """
        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        if self.engine == InferenceEngine.OPENVINO:
            outputs = self.compiled_model([input_tensor])[self.output_layer]
        else:  # ONNX Runtime
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})[0]

        # 后处理
        detections = self.postprocess(outputs)

        return detections

    def get_person_count(self, image: np.ndarray) -> int:
        """
        获取图像中的人员数量

        Args:
            image: BGR 图像

        Returns:
            人员数量
        """
        detections = self.detect_image(image)
        return len(detections)

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            detections: 检测结果

        Returns:
            绘制了检测框的图像
        """
        vis_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            conf = detection['conf']

            x1, y1, x2, y2 = map(int, bbox)

            # 绘制矩形框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"Person {conf:.2f}"
            cv2.putText(
                vis_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 绘制人数统计
        person_count = len(detections)
        cv2.putText(
            vis_image, f"Count: {person_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )

        return vis_image


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    detector = AdaptivePersonDetector(
        model_dir="models",
        conf_threshold=0.5
    )

    print(f"\n使用推理引擎: {detector.engine.value}")
    print(f"输入尺寸: {detector.input_width}x{detector.input_height}")
