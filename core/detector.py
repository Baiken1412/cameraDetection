"""
人员检测模块 - 使用YOLOv8检测图片中的所有人员
"""
import logging
from typing import List, Dict, Union
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from reid_config import Config
from utils.image_processor import load_image

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    人员检测器 - 基于YOLOv8
    只检测person类别(class_id=0)
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = 'auto',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4
    ):
        """
        初始化人员检测器

        Args:
            model_path: YOLOv8模型路径，默认使用yolov8n.pt
            device: 'cuda', 'cpu', 'auto'
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
        """
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 确保模型目录存在
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查模型文件是否存在
        model_file = Path(self.model_path)
        if not model_file.exists() or (model_file.exists() and model_file.stat().st_size == 0):
            logger.warning(f"模型文件不存在或为空: {self.model_path}")
            logger.error(f"YOLO模型文件未找到，请将 {Config.YOLO_MODEL_NAME} 文件放置在以下位置之一：")
            logger.error(f"  1. {self.model_path}")
            logger.error(f"  2. exe所在目录的 models/ 文件夹中")
            logger.error(f"  3. 或者修改config.py中的模型路径配置")
            # 在离线环境下，尝试使用模型名称可能会导致下载失败
            # 但如果用户有缓存的模型，仍然可以尝试加载
            logger.info(f"尝试从缓存加载模型: {Config.YOLO_MODEL_NAME}")
            model_to_load = Config.YOLO_MODEL_NAME
        else:
            model_to_load = str(self.model_path)
            logger.info(f"找到模型文件: {self.model_path}")

        # 加载YOLO模型
        logger.info(f"加载YOLO模型: {model_to_load}, 设备: {self.device}")
        self.model = YOLO(model_to_load)
        
        # 如果使用自动下载，尝试将模型复制到指定位置
        if not model_file.exists() and model_to_load == Config.YOLO_MODEL_NAME:
            try:
                # YOLO下载的模型通常在缓存目录，尝试找到并复制
                import shutil
                cache_paths = [
                    Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics' / Config.YOLO_MODEL_NAME,
                    Path.home() / '.ultralytics' / Config.YOLO_MODEL_NAME,
                ]
                for cache_path in cache_paths:
                    if cache_path.exists():
                        shutil.copy2(cache_path, model_file)
                        logger.info(f"模型已复制到: {self.model_path}")
                        break
            except Exception as e:
                logger.warning(f"无法复制模型文件: {e}，将使用缓存中的模型")
        
        logger.info("YOLO模型加载完成")

    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == 'auto':
            return Config.DEVICE
        return device

    def detect(self, image_path: Union[str, Path]) -> List[Dict]:
        """
        检测单张图片中的人员

        Args:
            image_path: 图片路径

        Returns:
            List[Dict]: 人员检测结果列表
                [{'bbox': [x1, y1, x2, y2], 'conf': 0.95, 'class_id': 0}, ...]
        """
        image_path = str(image_path)

        # 读取图片（支持中文路径）
        try:
            img = load_image(image_path)
        except Exception as e:
            logger.error(f"无法读取图片: {image_path}, 错误: {e}")
            return []

        # YOLO推理
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # 只检测person类别
            device=self.device,
            verbose=False
        )

        # 解析结果
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                # 获取边界框坐标 (xyxy格式)
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                detection = {
                    'bbox': [float(x) for x in xyxy],  # [x1, y1, x2, y2]
                    'conf': conf,
                    'class_id': class_id
                }
                detections.append(detection)

        logger.debug(f"检测到 {len(detections)} 个人员 in {image_path}")
        return detections

    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> Dict[str, List[Dict]]:
        """
        批量检测多张图片

        Args:
            image_paths: 图片路径列表
            batch_size: 批量大小

        Returns:
            Dict[str, List[Dict]]: 以图片路径为key的检测结果字典
                {
                    'image1.jpg': [{'bbox': [x1,y1,x2,y2], 'conf': 0.95}, ...],
                    'image2.jpg': [...]
                }
        """
        logger.info(f"开始批量检测 {len(image_paths)} 张图片")

        results_dict = {}

        # 分批处理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # 批量预测
            batch_results = self.model.predict(
                [str(p) for p in batch_paths],
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],  # 只检测person类别
                device=self.device,
                verbose=False
            )

            # 解析每张图片的结果
            for image_path, result in zip(batch_paths, batch_results):
                image_path = str(image_path)
                detections = []

                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])

                    detection = {
                        'bbox': [float(x) for x in xyxy],
                        'conf': conf,
                        'class_id': class_id
                    }
                    detections.append(detection)

                results_dict[image_path] = detections

            logger.debug(f"已处理 {i + len(batch_paths)}/{len(image_paths)} 张图片")

        total_detections = sum(len(dets) for dets in results_dict.values())
        logger.info(f"批量检测完成，共检测到 {total_detections} 个人员")

        return results_dict

    def detect_and_crop(
        self,
        image_path: Union[str, Path],
        save_dir: Union[str, Path] = None,
        min_bbox_size: tuple = (64, 128)
    ) -> List[Dict]:
        """
        检测并裁剪人员区域

        Args:
            image_path: 图片路径
            save_dir: 保存裁剪图片的目录
            min_bbox_size: 最小边界框尺寸 (width, height)

        Returns:
            List[Dict]: 包含裁剪图片的检测结果
                [{'bbox': [x1,y1,x2,y2], 'conf': 0.95, 'crop': numpy_array, 'crop_path': str}, ...]
        """
        image_path = str(image_path)

        # 检测人员
        detections = self.detect(image_path)

        # 读取原始图片（支持中文路径）
        try:
            img = load_image(image_path)
        except Exception as e:
            logger.error(f"无法读取图片: {image_path}, 错误: {e}")
            return []

        # 裁剪每个检测到的人员
        results = []
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 检查边界框大小
            width = x2 - x1
            height = y2 - y1
            if width < min_bbox_size[0] or height < min_bbox_size[1]:
                logger.warning(f"跳过过小的检测框: {width}x{height} < {min_bbox_size}")
                continue

            # 裁剪人员区域
            crop = img[y1:y2, x1:x2]

            detection['crop'] = crop

            # 可选：保存裁剪图片
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                image_name = Path(image_path).stem
                crop_filename = f"{image_name}_person{idx}.jpg"
                crop_path = save_dir / crop_filename

                cv2.imwrite(str(crop_path), crop)
                detection['crop_path'] = str(crop_path)

            results.append(detection)

        logger.debug(f"裁剪了 {len(results)} 个人员区域")
        return results

    def visualize_detections(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path] = None
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image_path: 图片路径
            output_path: 输出路径（可选）

        Returns:
            np.ndarray: 绘制了检测框的图片
        """
        image_path = str(image_path)
        detections = self.detect(image_path)

        # 读取图片（支持中文路径）
        try:
            img = load_image(image_path)
        except Exception as e:
            logger.error(f"无法读取图片: {image_path}, 错误: {e}")
            return None

        # 绘制检测框
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['conf']

            x1, y1, x2, y2 = map(int, bbox)

            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制置信度标签
            label = f"Person {conf:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 保存结果
        if output_path:
            cv2.imwrite(str(output_path), img)
            logger.info(f"可视化结果已保存: {output_path}")

        return img
