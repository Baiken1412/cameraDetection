"""
人员聚类模块 - 基于图的连通分量算法
"""
import logging
from typing import List, Dict
from collections import defaultdict
import numpy as np

from reid_config import Config

logger = logging.getLogger(__name__)


class GraphClustering:
    """
    基于图的聚类算法
    使用连通分量(Connected Components)方法
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        初始化聚类器

        Args:
            similarity_threshold: 相似度阈值
        """
        self.threshold = similarity_threshold
        logger.info(f"初始化图聚类器，阈值: {similarity_threshold}")

    def cluster(
        self,
        similarity_matrix: np.ndarray,
        metadata: List[Dict] = None
    ) -> List[List[int]]:
        """
        对相似度矩阵进行聚类

        Args:
            similarity_matrix: 相似度矩阵 (N, N)
                - 已应用同图约束（负值表示禁止合并）
            metadata: 元数据列表（可选，用于验证）

        Returns:
            List[List[int]]: 聚类结果
                [[0, 5, 12], [1, 3], [2], [4, 6, 7], ...]
                每个子列表是一个聚类组，包含person_id列表
        """
        N = len(similarity_matrix)
        logger.info(f"开始聚类，共 {N} 个人员，阈值: {self.threshold}")

        # 构建邻接表
        adj = self._build_adjacency_list(similarity_matrix, metadata)

        # DFS查找连通分量
        clusters = self._find_connected_components(adj, N)

        logger.info(f"聚类完成，共 {len(clusters)} 个分组")

        # 输出聚类统计
        cluster_sizes = [len(c) for c in clusters]
        logger.info(f"聚类大小统计: 最小={min(cluster_sizes)}, "
                   f"最大={max(cluster_sizes)}, "
                   f"平均={np.mean(cluster_sizes):.2f}")

        return clusters

    def _build_adjacency_list(
        self,
        similarity_matrix: np.ndarray,
        metadata: List[Dict] = None
    ) -> Dict[int, List[int]]:
        """
        构建邻接表

        Args:
            similarity_matrix: 相似度矩阵
            metadata: 元数据列表（用于双重验证）

        Returns:
            Dict[int, List[int]]: 邻接表
                {0: [5, 12], 5: [0, 12], ...}
        """
        N = len(similarity_matrix)
        adj = defaultdict(list)

        edge_count = 0

        # 遍历上三角矩阵（避免重复）
        for i in range(N):
            for j in range(i + 1, N):
                sim = similarity_matrix[i][j]

                # 检查相似度是否大于阈值且不是同图约束（负值）
                if sim >= self.threshold:
                    # 双重验证：如果提供了metadata，再次检查同图约束
                    if metadata is not None:
                        if metadata[i]['image_id'] == metadata[j]['image_id']:
                            logger.warning(
                                f"检测到同图约束违反: person_{i} 和 person_{j} "
                                f"来自同一图片但相似度≥阈值"
                            )
                            continue

                    # 添加边（无向图）
                    adj[i].append(j)
                    adj[j].append(i)
                    edge_count += 1

        logger.info(f"邻接表构建完成，共 {edge_count} 条边")

        return adj

    def _find_connected_components(
        self,
        adj: Dict[int, List[int]],
        n_nodes: int
    ) -> List[List[int]]:
        """
        使用DFS查找连通分量

        Args:
            adj: 邻接表
            n_nodes: 节点总数

        Returns:
            List[List[int]]: 连通分量列表
        """
        visited = [False] * n_nodes
        clusters = []

        for node in range(n_nodes):
            if not visited[node]:
                cluster = []
                self._dfs(node, adj, visited, cluster)
                clusters.append(cluster)

        return clusters

    def _dfs(
        self,
        node: int,
        adj: Dict[int, List[int]],
        visited: List[bool],
        cluster: List[int]
    ):
        """
        深度优先搜索

        Args:
            node: 当前节点
            adj: 邻接表
            visited: 访问标记数组
            cluster: 当前聚类组（递归填充）
        """
        visited[node] = True
        cluster.append(node)

        # 遍历所有邻居节点
        for neighbor in adj[node]:
            if not visited[neighbor]:
                self._dfs(neighbor, adj, visited, cluster)

    def refine_clusters(
        self,
        clusters: List[List[int]],
        metadata: List[Dict]
    ) -> List[List[int]]:
        """
        优化聚类结果 - 确保同图约束

        Args:
            clusters: 聚类结果
            metadata: 元数据列表

        Returns:
            List[List[int]]: 优化后的聚类结果
        """
        logger.info("优化聚类结果...")

        refined_clusters = []
        violations = 0

        for cluster in clusters:
            if len(cluster) == 1:
                # 单人聚类无需优化
                refined_clusters.append(cluster)
                continue

            # 检查聚类中是否有同图人员
            image_ids = [metadata[person_id]['image_id'] for person_id in cluster]

            if len(image_ids) == len(set(image_ids)):
                # 所有人员来自不同图片，合法
                refined_clusters.append(cluster)
            else:
                # 存在同图人员，需要拆分
                logger.warning(f"检测到同图人员在同一聚类中，拆分...")
                violations += 1

                # 按图片分组
                image_groups = defaultdict(list)
                for person_id in cluster:
                    image_id = metadata[person_id]['image_id']
                    image_groups[image_id].append(person_id)

                # 每个图片的人员分开
                for image_id, person_ids in image_groups.items():
                    for person_id in person_ids:
                        refined_clusters.append([person_id])

        if violations > 0:
            logger.warning(f"发现 {violations} 个违反同图约束的聚类，已拆分")
        else:
            logger.info("所有聚类均满足同图约束")

        logger.info(f"优化完成，最终 {len(refined_clusters)} 个分组")

        return refined_clusters

    def get_cluster_stats(self, clusters: List[List[int]]) -> Dict:
        """
        获取聚类统计信息

        Args:
            clusters: 聚类结果

        Returns:
            Dict: 统计信息
        """
        cluster_sizes = [len(c) for c in clusters]

        stats = {
            'total_clusters': len(clusters),
            'total_persons': sum(cluster_sizes),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0,
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1)
        }

        logger.info(
            f"聚类统计: 总组数={stats['total_clusters']}, "
            f"单人组数={stats['singleton_clusters']}, "
            f"平均组大小={stats['avg_cluster_size']:.2f}, "
            f"最大组大小={stats['max_cluster_size']}"
        )

        return stats

    def update_threshold(self, new_threshold: float):
        """
        更新相似度阈值

        Args:
            new_threshold: 新的阈值
        """
        logger.info(f"更新相似度阈值: {self.threshold} -> {new_threshold}")
        self.threshold = new_threshold

    def recluster(
        self,
        similarity_matrix: np.ndarray,
        new_threshold: float,
        metadata: List[Dict] = None
    ) -> List[List[int]]:
        """
        使用新阈值重新聚类

        Args:
            similarity_matrix: 相似度矩阵
            new_threshold: 新阈值
            metadata: 元数据列表

        Returns:
            List[List[int]]: 新的聚类结果
        """
        self.update_threshold(new_threshold)
        return self.cluster(similarity_matrix, metadata)
