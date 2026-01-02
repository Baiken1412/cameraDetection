"""
ReID特征提取模块 - 使用OSNet提取人员外观特征
"""
import logging
from typing import List, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchreid

from reid_config import Config

logger = logging.getLogger(__name__)


class ReIDFeatureExtractor:
    """
    ReID特征提取器 - 基于OSNet
    提取512维特征向量
    """

    def __init__(
        self,
        model_name: str = 'osnet_x1_0',
        device: str = 'auto',
        pretrained: bool = True
    ):
        """
        初始化特征提取器

        Args:
            model_name: 模型名称 ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5')
            device: 'cuda', 'cpu', 'auto'
            pretrained: 是否使用预训练权重
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.pretrained = pretrained

        # 加载模型
        logger.info(f"加载ReID模型: {model_name}, 设备: {self.device}")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,  # 预训练类别数
            loss='softmax',
            pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("ReID模型加载完成")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),  # (256, 128)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.IMAGENET_MEAN,
                std=Config.IMAGENET_STD
            )
        ])

    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == 'auto':
            device_str = Config.DEVICE
        else:
            device_str = device
        return torch.device(device_str)

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: numpy数组(BGR)或PIL Image

        Returns:
            torch.Tensor: 预处理后的图像张量 (1, 3, H, W)
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            # OpenCV BGR -> RGB
            image = Image.fromarray(image[:, :, ::-1])

        # 应用变换
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度

        return img_tensor

    def extract_feature(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        提取单张图片的特征向量

        Args:
            image: numpy数组、PIL Image或图片路径

        Returns:
            np.ndarray: L2归一化的特征向量 (512,)
        """
        # 加载图片
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # 预处理
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)

        # 提取特征
        with torch.no_grad():
            features = self.model(img_tensor)

        # L2归一化
        features = F.normalize(features, p=2, dim=1)

        # 转换为numpy
        features = features.cpu().numpy().flatten()

        return features

    def extract_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        批量提取特征向量

        Args:
            images: 图片列表（numpy数组、PIL Image或路径）
            batch_size: 批量大小

        Returns:
            np.ndarray: 特征矩阵 (N, 512)
        """
        logger.info(f"开始批量提取特征，共 {len(images)} 张图片")

        # 检查是否为空列表
        if not images:
            logger.warning("图片列表为空，返回空特征数组")
            # 返回一个空的numpy数组，形状为 (0, feature_dim)
            # 这里假设特征维度为512（根据文档），实际维度应该从模型获取
            # 为了更准确，可以返回模型输出维度
            try:
                # 尝试获取模型的输出维度（如果模型有定义）
                if hasattr(self.model, 'fc') and hasattr(self.model.fc, 'out_features'):
                    feature_dim = self.model.fc.out_features
                elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features'):
                    feature_dim = self.model.classifier.out_features
                else:
                    # 默认使用512（ReID模型常见维度）
                    feature_dim = 512
                return np.empty((0, feature_dim), dtype=np.float32)
            except Exception:
                # 如果无法获取维度，返回默认的512维空数组
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
                img_tensor = self.preprocess_image(img)
                batch_tensors.append(img_tensor)

            # 合并为batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_tensor = batch_tensor.to(self.device)

            # 批量提取特征
            with torch.no_grad():
                features = self.model(batch_tensor)

            # L2归一化
            features = F.normalize(features, p=2, dim=1)

            # 转换为numpy
            features = features.cpu().numpy()
            all_features.append(features)

            logger.debug(f"已处理 {i + len(batch_images)}/{len(images)} 张图片")

        # 合并所有特征
        if not all_features:
            logger.warning("没有成功提取任何特征，返回空特征数组")
            # 返回空数组（这种情况下理论上不应该发生，因为images不为空）
            return np.empty((0, 512), dtype=np.float32)
        
        all_features = np.vstack(all_features)

        logger.info(f"特征提取完成，形状: {all_features.shape}")

        return all_features

    def extract_from_crops(
        self,
        crop_paths: List[Union[str, Path]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        从裁剪的人像图片中提取特征

        Args:
            crop_paths: 裁剪图片路径列表
            batch_size: 批量大小

        Returns:
            np.ndarray: 特征矩阵 (N, 512)
        """
        return self.extract_batch(crop_paths, batch_size)

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
            np.ndarray: 特征矩阵
        """
        features = np.load(load_path)
        logger.info(f"特征已加载: {load_path}, 形状: {features.shape}")
        return features

    def get_feature_dimension(self) -> int:
        """
        获取特征维度

        Returns:
            int: 特征维度
        """
        return Config.FEATURE_DIM
