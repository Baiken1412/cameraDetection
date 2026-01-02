"""
人员锚点管理模块
支持三阶段工作流程：初始化锚点 -> 运行期补充 -> 稳定期复核
"""
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import json

logger = logging.getLogger(__name__)


class AnchorManager:
    """
    人员锚点管理器

    核心概念：
    - Anchor（锚点）：为每个固定人员建立的稳定特征表示
    - 每个锚点包含多个样本的embedding，取均值作为anchor_embedding
    - 锚点用于快速匹配新的人像
    """

    def __init__(self, anchor_file: Path = None):
        """
        初始化锚点管理器

        Args:
            anchor_file: 锚点数据保存文件路径
        """
        self.anchor_file = Path(anchor_file) if anchor_file else None

        # 锚点数据结构
        # {merged_person_id: {
        #     'anchor_embedding': np.array,  # 锚点特征向量
        #     'person_ids': [0, 1, 5],       # 包含的原始person_id
        #     'sample_count': 3,              # 样本数量
        #     'is_confirmed': True,           # 是否已确认
        #     'created_at': '2025-12-13',
        #     'updated_at': '2025-12-13',
        #     'confidence': 0.95              # 锚点可信度
        # }}
        self.anchors = {}

        # 加载已有的锚点
        if self.anchor_file and self.anchor_file.exists():
            self.load_anchors()

        logger.info("锚点管理器初始化完成")

    def create_anchor(
        self,
        merged_person_id: int,
        person_ids: List[int],
        embeddings: np.ndarray,
        is_confirmed: bool = True
    ) -> Dict:
        """
        创建人员锚点

        Args:
            merged_person_id: 合并人员ID
            person_ids: 包含的原始person_id列表
            embeddings: 特征向量矩阵 (N, feature_dim)
            is_confirmed: 是否已人工确认

        Returns:
            Dict: 锚点信息
        """
        if len(embeddings) == 0:
            raise ValueError("需要至少一个embedding来创建锚点")

        # 计算锚点embedding（使用均值）
        anchor_embedding = np.mean(embeddings, axis=0)

        # 归一化
        norm = np.linalg.norm(anchor_embedding)
        if norm > 0:
            anchor_embedding = anchor_embedding / norm

        # 计算锚点可信度（基于内部一致性）
        confidence = self._calculate_anchor_confidence(embeddings)

        # 创建锚点记录
        anchor = {
            'merged_person_id': merged_person_id,
            'anchor_embedding': anchor_embedding,
            'person_ids': person_ids,
            'sample_count': len(person_ids),
            'is_confirmed': is_confirmed,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': float(confidence)
        }

        self.anchors[merged_person_id] = anchor

        logger.info(
            f"创建锚点: merged_person_id={merged_person_id}, "
            f"样本数={len(person_ids)}, 可信度={confidence:.3f}"
        )

        return anchor

    def update_anchor(
        self,
        merged_person_id: int,
        new_person_ids: List[int],
        new_embeddings: np.ndarray
    ):
        """
        更新已有锚点（增加新样本）

        Args:
            merged_person_id: 锚点ID
            new_person_ids: 新增的person_id
            new_embeddings: 新增的特征向量
        """
        if merged_person_id not in self.anchors:
            raise ValueError(f"锚点 {merged_person_id} 不存在")

        anchor = self.anchors[merged_person_id]

        # 获取现有embedding（通过反推）
        old_embedding = anchor['anchor_embedding']
        old_count = anchor['sample_count']

        # 计算新的加权平均
        new_count = len(new_person_ids)
        total_count = old_count + new_count

        # 加权平均
        updated_embedding = (
            old_embedding * old_count +
            np.mean(new_embeddings, axis=0) * new_count
        ) / total_count

        # 归一化
        norm = np.linalg.norm(updated_embedding)
        if norm > 0:
            updated_embedding = updated_embedding / norm

        # 更新锚点
        anchor['anchor_embedding'] = updated_embedding
        anchor['person_ids'].extend(new_person_ids)
        anchor['sample_count'] = total_count
        anchor['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 重新计算可信度
        # 这里简化处理，实际可以收集所有embeddings重新计算
        anchor['confidence'] = min(0.99, anchor['confidence'] + 0.01)

        logger.info(
            f"更新锚点: merged_person_id={merged_person_id}, "
            f"新增样本={new_count}, 总样本={total_count}"
        )

    def match_to_anchors(
        self,
        embedding: np.ndarray,
        threshold_high: float = 0.7,
        threshold_low: float = 0.6
    ) -> Tuple[Optional[int], float, str]:
        """
        将新embedding与所有锚点匹配

        Args:
            embedding: 待匹配的特征向量
            threshold_high: 自动归入阈值
            threshold_low: 待确认阈值

        Returns:
            Tuple[merged_person_id, similarity, status]:
                - merged_person_id: 匹配到的锚点ID（None表示未匹配）
                - similarity: 相似度分数
                - status: 'auto_matched' | 'pending_confirm' | 'unknown'
        """
        if not self.anchors:
            return None, 0.0, 'unknown'

        # 计算与所有锚点的相似度
        best_match_id = None
        best_similarity = -1.0

        for merged_id, anchor in self.anchors.items():
            if not anchor['is_confirmed']:
                continue  # 只匹配已确认的锚点

            anchor_emb = anchor['anchor_embedding']

            # 余弦相似度
            similarity = float(np.dot(embedding, anchor_emb))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = merged_id

        # 判断匹配状态
        if best_similarity >= threshold_high:
            status = 'auto_matched'
        elif best_similarity >= threshold_low:
            status = 'pending_confirm'
        else:
            best_match_id = None
            status = 'unknown'

        logger.debug(
            f"锚点匹配: best_id={best_match_id}, "
            f"similarity={best_similarity:.3f}, status={status}"
        )

        return best_match_id, best_similarity, status

    def batch_match_to_anchors(
        self,
        embeddings: np.ndarray,
        threshold_high: float = 0.7,
        threshold_low: float = 0.6
    ) -> List[Tuple[Optional[int], float, str]]:
        """
        批量匹配到锚点

        Args:
            embeddings: 特征向量矩阵 (N, feature_dim)
            threshold_high: 自动归入阈值
            threshold_low: 待确认阈值

        Returns:
            List[Tuple]: 匹配结果列表
        """
        results = []
        for embedding in embeddings:
            result = self.match_to_anchors(
                embedding,
                threshold_high,
                threshold_low
            )
            results.append(result)

        return results

    def get_anchor_stats(self) -> Dict:
        """
        获取锚点统计信息

        Returns:
            Dict: 统计信息
        """
        confirmed_anchors = [
            a for a in self.anchors.values() if a['is_confirmed']
        ]

        stats = {
            'total_anchors': len(self.anchors),
            'confirmed_anchors': len(confirmed_anchors),
            'total_samples': sum(a['sample_count'] for a in self.anchors.values()),
            'avg_samples_per_anchor': (
                np.mean([a['sample_count'] for a in self.anchors.values()])
                if self.anchors else 0
            ),
            'avg_confidence': (
                np.mean([a['confidence'] for a in confirmed_anchors])
                if confirmed_anchors else 0
            )
        }

        return stats

    def list_anchors(self, confirmed_only: bool = False) -> List[Dict]:
        """
        列出所有锚点

        Args:
            confirmed_only: 是否只列出已确认的锚点

        Returns:
            List[Dict]: 锚点信息列表
        """
        anchors = []
        for merged_id, anchor in self.anchors.items():
            if confirmed_only and not anchor['is_confirmed']:
                continue

            info = {
                'merged_person_id': merged_id,
                'sample_count': anchor['sample_count'],
                'is_confirmed': anchor['is_confirmed'],
                'confidence': anchor['confidence'],
                'created_at': anchor['created_at'],
                'updated_at': anchor['updated_at']
            }
            anchors.append(info)

        return anchors

    def remove_anchor(self, merged_person_id: int):
        """
        删除锚点（谨慎操作）

        Args:
            merged_person_id: 要删除的锚点ID
        """
        if merged_person_id in self.anchors:
            del self.anchors[merged_person_id]
            logger.warning(f"已删除锚点: merged_person_id={merged_person_id}")
        else:
            raise ValueError(f"锚点 {merged_person_id} 不存在")

    def save_anchors(self, filepath: Path = None):
        """
        保存锚点数据

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath) if filepath else self.anchor_file
        if filepath is None:
            raise ValueError("未指定保存路径")

        # 准备保存数据
        data = {
            'version': '1.0',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'anchors': {}
        }

        # 转换numpy数组为列表
        for merged_id, anchor in self.anchors.items():
            anchor_copy = anchor.copy()
            anchor_copy['anchor_embedding'] = anchor['anchor_embedding'].tolist()
            data['anchors'][str(merged_id)] = anchor_copy

        # 保存到文件
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"锚点数据已保存: {filepath}")

    def load_anchors(self, filepath: Path = None):
        """
        加载锚点数据

        Args:
            filepath: 加载路径
        """
        filepath = Path(filepath) if filepath else self.anchor_file
        if filepath is None or not filepath.exists():
            logger.warning("锚点文件不存在")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 恢复锚点数据
        self.anchors = {}
        for merged_id_str, anchor_data in data.get('anchors', {}).items():
            merged_id = int(merged_id_str)

            # 转换embedding回numpy数组
            anchor_data['anchor_embedding'] = np.array(
                anchor_data['anchor_embedding'],
                dtype=np.float32
            )

            self.anchors[merged_id] = anchor_data

        logger.info(
            f"锚点数据已加载: {len(self.anchors)} 个锚点, "
            f"来自 {filepath}"
        )

    def _calculate_anchor_confidence(self, embeddings: np.ndarray) -> float:
        """
        计算锚点可信度（基于内部一致性）

        Args:
            embeddings: 特征向量矩阵 (N, feature_dim)

        Returns:
            float: 可信度分数 [0, 1]
        """
        if len(embeddings) == 1:
            return 0.8  # 单样本默认0.8

        # 计算所有样本间的平均相似度
        n = len(embeddings)
        similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # 转换为可信度
        # 相似度越高，可信度越高
        confidence = min(0.99, max(0.5, avg_similarity))

        return confidence

    def get_anchor_embedding(self, merged_person_id: int) -> Optional[np.ndarray]:
        """
        获取锚点的embedding

        Args:
            merged_person_id: 锚点ID

        Returns:
            np.ndarray: 锚点embedding，不存在返回None
        """
        if merged_person_id in self.anchors:
            return self.anchors[merged_person_id]['anchor_embedding']
        return None

    def is_anchor_confirmed(self, merged_person_id: int) -> bool:
        """
        检查锚点是否已确认

        Args:
            merged_person_id: 锚点ID

        Returns:
            bool: 是否已确认
        """
        if merged_person_id in self.anchors:
            return self.anchors[merged_person_id]['is_confirmed']
        return False

    def confirm_anchor(self, merged_person_id: int):
        """
        确认锚点

        Args:
            merged_person_id: 锚点ID
        """
        if merged_person_id in self.anchors:
            self.anchors[merged_person_id]['is_confirmed'] = True
            self.anchors[merged_person_id]['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"锚点已确认: merged_person_id={merged_person_id}")
        else:
            raise ValueError(f"锚点 {merged_person_id} 不存在")
