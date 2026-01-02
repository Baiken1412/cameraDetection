"""
ONNX ReID 特征提取器 - 使用 ONNX Runtime 加速推理
CPU 上比 PyTorch 快 3-5 倍
"""
import logging
from typing import List, Union
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ReIDFeatureExtractorONNX:
    """
    ONNX ReID 特征提取器 - 基于 OSNet ONNX 模型
    提取512维特征向量
    """

    def __init__(
        self,
        model_path: str = None,
        num_threads: int = None
    ):
        """
        初始化特征提取器

        Args:
            model_path: ONNX 模型路径
            num_threads: CPU 线程数（None=自动）
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        self.model_path = model_path

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
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]  # 256
        self.input_width = input_shape[3]   # 128

        logger.info(f"ONNX 模型加载完成")
        logger.info(f"输入尺寸: {self.input_width}x{self.input_height}")

        # ImageNet 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        预处理图像

        Args:
            image: numpy 数组(BGR) 或 PIL Image

        Returns:
            预处理后的图像张量
        """
        # 转换为 PIL Image
        if isinstance(image, np.ndarray):
            # OpenCV BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Resize
        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)

        # 转换为 numpy array
        img = np.array(image, dtype=np.float32)

        # 归一化到 [0, 1]
        img = img / 255.0

        # ImageNet 归一化
        img = (img - self.mean) / self.std

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)

        return img

    def extract_feature(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        提取单张图片的特征向量

        Args:
            image: numpy数组、PIL Image 或图片路径

        Returns:
            L2归一化的特征向量 (512,)
        """
        # 加载图片
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        features = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # L2 归一化
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

        # 转换为 1D 数组
        features = features.flatten()

        return features

    def extract_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        batch_size: int = 4
    ) -> np.ndarray:
        """
        批量提取特征向量（CPU 推荐小批量）

        Args:
            images: 图片列表
            batch_size: 批量大小（CPU 推荐 2-8）

        Returns:
            特征矩阵 (N, 512)
        """
        logger.info(f"开始批量提取特征，共 {len(images)} 张图片")

        if not images:
            logger.warning("图片列表为空")
            return np.empty((0, 512), dtype=np.float32)

        all_features = []

        # 分批处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # 预处理批量图片
            batch_tensors = []
            for img in batch_images:
                # 加载图片
                if isinstance(img, (str, Path)):
                    img = Image.open(img)

                # 预处理
                img_tensor = self.preprocess(img)
                batch_tensors.append(img_tensor)

            # 合并为 batch
            batch_tensor = np.concatenate(batch_tensors, axis=0)

            # 批量推理
            features = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]

            # L2 归一化
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

            all_features.append(features)

            logger.debug(f"已处理 {i + len(batch_images)}/{len(images)} 张图片")

        # 合并所有特征
        if not all_features:
            return np.empty((0, 512), dtype=np.float32)

        all_features = np.vstack(all_features)

        logger.info(f"特征提取完成，形状: {all_features.shape}")

        return all_features

    def save_features(self, features: np.ndarray, save_path: Union[str, Path]):
        """
        保存特征到文件

        Args:
            features: 特征矩阵
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features)
        logger.info(f"特征已保存: {save_path}, 形状: {features.shape}")

    @staticmethod
    def load_features(load_path: Union[str, Path]) -> np.ndarray:
        """
        从文件加载特征

        Args:
            load_path: 文件路径

        Returns:
            特征矩阵
        """
        features = np.load(load_path)
        logger.info(f"特征已加载: {load_path}, 形状: {features.shape}")
        return features
