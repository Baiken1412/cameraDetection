"""
相似度计算模块 - 计算特征向量之间的相似度
"""
import logging
from typing import List, Dict
import numpy as np
from scipy.spatial.distance import cdist

from reid_config import Config

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    相似度计算器
    支持余弦相似度和欧氏距离
    """

    def __init__(self, metric: str = 'cosine'):
        """
        初始化相似度计算器

        Args:
            metric: 相似度度量方式 ('cosine', 'euclidean')
        """
        self.metric = metric
        logger.info(f"初始化相似度计算器，度量方式: {metric}")

    def compute_similarity_matrix(
        self,
        features: np.ndarray,
        metric: str = None
    ) -> np.ndarray:
        """
        计算特征矩阵的相似度矩阵

        Args:
            features: 特征矩阵 (N, D)
            metric: 相似度度量方式（可选，覆盖初始化的metric）

        Returns:
            np.ndarray: 相似度矩阵 (N, N)
                余弦相似度：值域[0, 1]，越大越相似
                欧氏距离：值域[0, +∞)，越小越相似
        """
        metric = metric or self.metric

        N = len(features)
        logger.info(f"计算相似度矩阵，特征数量: {N}, 度量方式: {metric}")

        if metric == 'cosine':
            # 余弦相似度 = 1 - 余弦距离
            # 由于特征已L2归一化，可直接用点积计算
            similarity_matrix = np.dot(features, features.T)

            # 确保对角线为1（数值精度问题）
            np.fill_diagonal(similarity_matrix, 1.0)

            # 裁剪到[0, 1]范围
            similarity_matrix = np.clip(similarity_matrix, 0, 1)

        elif metric == 'euclidean':
            # 欧氏距离（越小越相似）
            distance_matrix = cdist(features, features, metric='euclidean')

            # 转换为相似度：sim = 1 / (1 + dist)
            similarity_matrix = 1 / (1 + distance_matrix)

        else:
            raise ValueError(f"不支持的度量方式: {metric}")

        logger.info(f"相似度矩阵计算完成，形状: {similarity_matrix.shape}")

        return similarity_matrix

    def apply_same_image_constraint(
        self,
        similarity_matrix: np.ndarray,
        metadata: List[Dict]
    ) -> np.ndarray:
        """
        应用同图约束 - 同一张图片的人员禁止合并

        Args:
            similarity_matrix: 相似度矩阵 (N, N)
            metadata: 元数据列表
                [{'person_id': 0, 'image_id': 'image001', ...}, ...]

        Returns:
            np.ndarray: 应用约束后的相似度矩阵
        """
        logger.info("应用同图约束...")

        N = len(metadata)
        constraint_count = 0

        # 遍历所有pair
        for i in range(N):
            for j in range(N):
                if i != j:  # 不处理对角线
                    # 检查是否来自同一张图片
                    if metadata[i]['image_id'] == metadata[j]['image_id']:
                        similarity_matrix[i][j] = -1.0  # 设为-1表示禁止合并
                        constraint_count += 1

        logger.info(f"同图约束已应用，共 {constraint_count // 2} 对被标记为禁止合并")

        return similarity_matrix

    def get_top_k_similar(
        self,
        similarity_matrix: np.ndarray,
        person_idx: int,
        k: int = 5,
        exclude_negatives: bool = True
    ) -> List[Dict]:
        """
        获取与指定人员最相似的Top-K人员

        Args:
            similarity_matrix: 相似度矩阵 (N, N)
            person_idx: 人员索引
            k: 返回的数量
            exclude_negatives: 是否排除负相似度（同图约束标记）

        Returns:
            List[Dict]: Top-K相似人员列表
                [{'person_idx': 5, 'similarity': 0.85}, ...]
        """
        similarities = similarity_matrix[person_idx].copy()

        # 排除自己
        similarities[person_idx] = -np.inf

        # 排除负相似度
        if exclude_negatives:
            similarities[similarities < 0] = -np.inf

        # 获取Top-K索引
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # 构建结果
        results = []
        for idx in top_k_indices:
            if similarities[idx] > -np.inf:
                results.append({
                    'person_idx': int(idx),
                    'similarity': float(similarities[idx])
                })

        return results

    def get_similarity_stats(self, similarity_matrix: np.ndarray) -> Dict:
        """
        获取相似度矩阵的统计信息

        Args:
            similarity_matrix: 相似度矩阵 (N, N)

        Returns:
            Dict: 统计信息
                {
                    'mean': 平均相似度,
                    'std': 标准差,
                    'min': 最小相似度（排除对角线和负值）,
                    'max': 最大相似度（排除对角线）,
                    'median': 中位数
                }
        """
        # 复制矩阵并去除对角线
        sim_copy = similarity_matrix.copy()
        np.fill_diagonal(sim_copy, np.nan)

        # 获取上三角部分（避免重复计算）
        upper_triangle = sim_copy[np.triu_indices_from(sim_copy, k=1)]

        # 排除负值（同图约束标记）
        valid_similarities = upper_triangle[upper_triangle >= 0]

        stats = {
            'mean': float(np.mean(valid_similarities)) if len(valid_similarities) > 0 else 0.0,
            'std': float(np.std(valid_similarities)) if len(valid_similarities) > 0 else 0.0,
            'min': float(np.min(valid_similarities)) if len(valid_similarities) > 0 else 0.0,
            'max': float(np.max(valid_similarities)) if len(valid_similarities) > 0 else 0.0,
            'median': float(np.median(valid_similarities)) if len(valid_similarities) > 0 else 0.0,
            'total_pairs': len(valid_similarities),
            'constrained_pairs': int(np.sum(upper_triangle < 0))
        }

        logger.info(f"相似度统计: 平均={stats['mean']:.3f}, "
                   f"中位数={stats['median']:.3f}, "
                   f"范围=[{stats['min']:.3f}, {stats['max']:.3f}]")

        return stats

    def filter_by_threshold(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        根据阈值过滤相似度矩阵

        Args:
            similarity_matrix: 相似度矩阵 (N, N)
            threshold: 阈值

        Returns:
            np.ndarray: 过滤后的矩阵（低于阈值的设为0）
        """
        filtered_matrix = similarity_matrix.copy()

        # 低于阈值的设为0
        filtered_matrix[filtered_matrix < threshold] = 0

        # 保留对角线为1
        np.fill_diagonal(filtered_matrix, 1.0)

        # 保留同图约束标记(-1)
        filtered_matrix[similarity_matrix < 0] = -1.0

        logger.info(f"相似度矩阵已过滤，阈值: {threshold}")

        return filtered_matrix

    @staticmethod
    def save_similarity_matrix(
        similarity_matrix: np.ndarray,
        save_path: str
    ):
        """保存相似度矩阵"""
        np.save(save_path, similarity_matrix)
        logger.info(f"相似度矩阵已保存: {save_path}")

    @staticmethod
    def load_similarity_matrix(load_path: str) -> np.ndarray:
        """加载相似度矩阵"""
        similarity_matrix = np.load(load_path)
        logger.info(f"相似度矩阵已加载: {load_path}, 形状: {similarity_matrix.shape}")
        return similarity_matrix
