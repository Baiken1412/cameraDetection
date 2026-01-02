"""
集成特征提取器 - 使用多个模型进行投票
"""
import logging
from typing import List, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchreid

from reid_config import Config

logger = logging.getLogger(__name__)


class EnsembleFeatureExtractor:
    """
    集成特征提取器
    使用多个ReID模型提取特征，然后融合
    """

    def __init__(
        self,
        model_names: List[str] = None,
        fusion_method: str = 'concat',
        device: str = 'auto'
    ):
        """
        初始化集成特征提取器

        Args:
            model_names: 模型名称列表，默认使用3个模型
            fusion_method: 特征融合方法
                - 'concat': 拼接（推荐）
                - 'average': 平均
                - 'weighted': 加权平均
            device: 计算设备
        """
        if model_names is None:
            # 默认使用3个互补的模型
            model_names = ['osnet_x1_0', 'resnet50', 'densenet121']

        self.model_names = model_names
        self.fusion_method = fusion_method
        self.device = self._get_device(device)

        logger.info(f"初始化集成特征提取器...")
        logger.info(f"  模型数量: {len(model_names)}")
        logger.info(f"  模型列表: {', '.join(model_names)}")
        logger.info(f"  融合方法: {fusion_method}")
        logger.info(f"  设备: {self.device}")

        # 加载所有模型
        self.models = []
        self.feature_dims = []

        for model_name in model_names:
            logger.info(f"  加载模型: {model_name}...")
            try:
                model = torchreid.models.build_model(
                    name=model_name,
                    num_classes=1000,
                    loss='softmax',
                    pretrained=True
                )
                model = model.to(self.device)
                model.eval()
                self.models.append(model)

                # 测试特征维度
                test_input = torch.randn(1, 3, 256, 128).to(self.device)
                with torch.no_grad():
                    test_output = model(test_input)
                feature_dim = test_output.shape[1]
                self.feature_dims.append(feature_dim)

                logger.info(f"    ✓ 加载成功，特征维度: {feature_dim}")

            except Exception as e:
                logger.error(f"    ✗ 加载失败: {e}")
                raise

        # 计算融合后的特征维度
        if fusion_method == 'concat':
            self.output_dim = sum(self.feature_dims)
        else:
            self.output_dim = self.feature_dims[0]

        logger.info(f"\n✓ 集成特征提取器初始化完成")
        logger.info(f"  输出特征维度: {self.output_dim}")

        # 图像预处理
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(Config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.IMAGENET_MEAN,
                std=Config.IMAGENET_STD
            )
        ])

        # 模型权重（用于加权融合）
        if fusion_method == 'weighted':
            # 可以根据模型在验证集上的表现设置权重
            # 这里使用默认权重
            self.weights = np.array([1.0, 1.2, 0.8])  # resnet50权重更高
            self.weights = self.weights / self.weights.sum()
            logger.info(f"  加权系数: {self.weights}")

    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == 'auto':
            device_str = Config.DEVICE
        else:
            device_str = device
        return torch.device(device_str)

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """预处理图像"""
        if isinstance(image, np.ndarray):
            # BGR -> RGB
            image = Image.fromarray(image[:, :, ::-1])

        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def extract_feature(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        提取单张图片的集成特征

        Args:
            image: 图像

        Returns:
            np.ndarray: 融合后的特征向量
        """
        # 加载图片
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # 预处理
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)

        # 从每个模型提取特征
        features_list = []

        with torch.no_grad():
            for model in self.models:
                features = model(img_tensor)
                # L2归一化
                features = F.normalize(features, p=2, dim=1)
                features_list.append(features)

        # 融合特征
        fused_features = self._fuse_features(features_list)

        # 转换为numpy
        fused_features = fused_features.cpu().numpy().flatten()

        return fused_features

    def extract_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        批量提取集成特征

        Args:
            images: 图片列表
            batch_size: 批量大小

        Returns:
            np.ndarray: 特征矩阵 (N, output_dim)
        """
        logger.info(f"开始批量提取集成特征，共 {len(images)} 张图片")
        logger.info(f"  使用 {len(self.models)} 个模型")

        all_features = []

        # 分批处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # 预处理批量图片
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                img_tensor = self.preprocess_image(img)
                batch_tensors.append(img_tensor)

            # 合并为batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_tensor = batch_tensor.to(self.device)

            # 从每个模型提取特征
            batch_features_list = []

            with torch.no_grad():
                for model_idx, model in enumerate(self.models):
                    features = model(batch_tensor)
                    # L2归一化
                    features = F.normalize(features, p=2, dim=1)
                    batch_features_list.append(features)

            # 融合特征
            fused_features = self._fuse_features(batch_features_list)

            # 转换为numpy
            fused_features = fused_features.cpu().numpy()
            all_features.append(fused_features)

            logger.debug(f"  已处理 {i + len(batch_images)}/{len(images)} 张图片")

        # 合并所有特征
        all_features = np.vstack(all_features)

        logger.info(f"集成特征提取完成，形状: {all_features.shape}")

        return all_features

    def _fuse_features(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多个模型的特征

        Args:
            features_list: 特征列表 [features1, features2, features3]

        Returns:
            torch.Tensor: 融合后的特征
        """
        if self.fusion_method == 'concat':
            # 拼接所有特征
            fused = torch.cat(features_list, dim=1)
            # 再次L2归一化
            fused = F.normalize(fused, p=2, dim=1)

        elif self.fusion_method == 'average':
            # 平均所有特征
            fused = torch.stack(features_list).mean(dim=0)
            # L2归一化
            fused = F.normalize(fused, p=2, dim=1)

        elif self.fusion_method == 'weighted':
            # 加权平均
            weighted_features = []
            for i, features in enumerate(features_list):
                weighted_features.append(features * self.weights[i])
            fused = torch.stack(weighted_features).sum(dim=0)
            # L2归一化
            fused = F.normalize(fused, p=2, dim=1)

        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")

        return fused

    def get_feature_dimension(self) -> int:
        """获取输出特征维度"""
        return self.output_dim

    def save_features(self, features: np.ndarray, save_path: Union[str, Path]):
        """保存特征"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features)
        logger.info(f"集成特征已保存: {save_path}, 形状: {features.shape}")

    @staticmethod
    def load_features(load_path: Union[str, Path]) -> np.ndarray:
        """加载特征"""
        features = np.load(load_path)
        logger.info(f"集成特征已加载: {load_path}, 形状: {features.shape}")
        return features
