"""
ONNX 人员检测器 - 使用 ONNX Runtime 加速 YOLO 推理
CPU 上比 PyTorch 快 5-10 倍
"""
import logging
from typing import List, Dict, Union
from pathlib import Path
import cv2
import numpy as np
from utils.image_processor import load_image

logger = logging.getLogger(__name__)


class PersonDetectorONNX:
    """
    ONNX 人员检测器 - 基于 YOLOv8 ONNX 模型
    只检测person类别(class_id=0)
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        num_threads: int = None
    ):
        """
        初始化 ONNX 人员检测器

        Args:
            model_path: ONNX 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU 阈值
            num_threads: CPU 线程数（None=自动）
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 检查模型文件
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX 模型文件不存在: {model_path}")

        # 设置线程数
        if num_threads is None:
            import os
            num_threads = max(1, os.cpu_count() - 2)

        # ONNX Runtime 会话选项
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 创建推理会话
        logger.info(f"加载 ONNX 模型: {model_path}")
        logger.info(f"使用线程数: {num_threads}")

        self.session = ort.InferenceSession(
            model_path,
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

        logger.info(f"ONNX 模型加载完成")
        logger.info(f"输入尺寸: {self.input_width}x{self.input_height}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像

        Args:
            image: BGR 图像

        Returns:
            预处理后的图像张量
        """
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
            检测结果列表
        """
        # YOLOv8 输出格式: (1, 84, 8400) - [batch, 84_channels, 8400_boxes]
        # 84 = 4(bbox) + 80(classes)
        predictions = outputs[0]  # (1, 84, 8400)

        # 转置: (8400, 84)
        predictions = predictions.squeeze(0).T  # (8400, 84)

        # 提取 bbox 和 class scores
        boxes = predictions[:, :4]  # (8400, 4) - [x_center, y_center, w, h]
        scores = predictions[:, 4:]  # (8400, 80) - class scores

        # 只保留 person 类别 (class_id=0)
        person_scores = scores[:, 0]  # (8400,)

        # 过滤低置信度
        mask = person_scores > self.conf_threshold
        boxes = boxes[mask]
        person_scores = person_scores[mask]

        if len(boxes) == 0:
            return []

        # 转换 bbox 格式: [x_center, y_center, w, h] -> [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

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
                'class_id': 0  # person
            }
            detections.append(detection)

        return detections

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression

        Args:
            boxes: (N, 4) - [x1, y1, x2, y2]
            scores: (N,)
            iou_threshold: IOU 阈值

        Returns:
            保留的索引列表
        """
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

        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # 后处理
        detections = self.postprocess(outputs[0])

        logger.debug(f"检测到 {len(detections)} 个人员 in {image_path}")
        return detections

    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 4
    ) -> Dict[str, List[Dict]]:
        """
        批量检测（注意：ONNX 批处理需要固定尺寸，这里逐张处理）

        Args:
            image_paths: 图片路径列表
            batch_size: 批量大小（暂不支持真正的批处理）

        Returns:
            检测结果字典
        """
        results = {}
        for image_path in image_paths:
            results[str(image_path)] = self.detect(image_path)
        return results
