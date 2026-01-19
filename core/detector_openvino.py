"""
OpenVINO 人员检测器 - 使用 INT8 量化加速
比 ONNX Runtime 快 1.5-2 倍
"""
import logging
import os
import tempfile
import shutil
from typing import List, Dict, Union
from pathlib import Path
import cv2
import numpy as np
from utils.image_processor import load_image

logger = logging.getLogger(__name__)


class PersonDetectorOpenVINO:
    """
    OpenVINO 人员检测器 - 基于 YOLOv11 INT8 量化模型
    只检测 person 类别 (class_id=0)
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        num_threads: int = None
    ):
        """
        初始化 OpenVINO 人员检测器

        Args:
            model_path: OpenVINO IR 模型路径 (.xml 文件)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU 阈值
            num_threads: CPU 线程数（None=自动）
        """
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("请安装 openvino: pip install openvino")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 检查模型文件
        if not Path(model_path).exists():
            raise FileNotFoundError(f"OpenVINO 模型文件不存在: {model_path}")

        # 设置线程数
        if num_threads is None:
            num_threads = max(1, os.cpu_count() - 2)

        logger.info(f"加载 OpenVINO 模型: {model_path}")
        logger.info(f"使用线程数: {num_threads}")

        # 创建 OpenVINO Core
        self.core = ov.Core()

        # 设置线程数
        self.core.set_property("CPU", {"INFERENCE_NUM_THREADS": num_threads})

        # 使用临时目录加载模型（避免中文路径问题）
        self._tmpdir = tempfile.mkdtemp()
        model_name = Path(model_path).stem

        for ext in ['.xml', '.bin']:
            src = Path(model_path).with_suffix(ext)
            if src.exists():
                shutil.copy(str(src), os.path.join(self._tmpdir, model_name + ext))

        tmp_model = os.path.join(self._tmpdir, model_name + ".xml")

        # 编译模型
        self.compiled_model = self.core.compile_model(tmp_model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()

        # 获取输入输出信息
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 获取输入尺寸
        input_shape = self.input_layer.shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        logger.info(f"OpenVINO 模型加载完成")
        logger.info(f"输入尺寸: {self.input_width}x{self.input_height}")

    def __del__(self):
        """清理临时文件"""
        if hasattr(self, '_tmpdir') and os.path.exists(self._tmpdir):
            try:
                del self.infer_request
                del self.compiled_model
                import gc
                gc.collect()
                import time
                time.sleep(0.1)
                shutil.rmtree(self._tmpdir, ignore_errors=True)
            except Exception:
                pass

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
        """后处理 YOLO 输出"""
        # YOLOv8/11 输出格式: (1, 84, 8400)
        predictions = outputs[0]

        # 转置: (8400, 84)
        predictions = predictions.squeeze(0).T

        # 提取 bbox 和 class scores
        boxes = predictions[:, :4]  # [x_center, y_center, w, h]
        scores = predictions[:, 4:]  # class scores

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
        """检测单张图片中的人员"""
        try:
            image = load_image(str(image_path))
        except Exception as e:
            logger.error(f"无法读取图片: {image_path}, 错误: {e}")
            return []

        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        self.infer_request.infer({0: input_tensor})
        outputs = self.infer_request.get_output_tensor(0).data

        # 后处理
        detections = self.postprocess(outputs)

        logger.debug(f"检测到 {len(detections)} 个人员 in {image_path}")
        return detections

    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 4
    ) -> Dict[str, List[Dict]]:
        """批量检测"""
        results = {}
        for image_path in image_paths:
            results[str(image_path)] = self.detect(image_path)
        return results
