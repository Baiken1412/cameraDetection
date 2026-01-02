"""
人工确认与合并模块
支持在自动ReID分析基础上进行人工确认和合并
整合锚点机制，支持三阶段工作流程
"""
import logging
from typing import List, Dict, Set, Union, Optional
from pathlib import Path
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)


class ManualConfirmationManager:
    """
    人工确认管理器
    负责管理人员分组的人工确认和合并操作
    """

    def __init__(self, confirmation_file: Union[str, Path] = None):
        """
        初始化人工确认管理器

        Args:
            confirmation_file: 人工确认数据的保存文件路径
        """
        self.confirmation_file = Path(confirmation_file) if confirmation_file else None

        # 存储人工确认的合并关系
        # 格式: {merged_person_id: [person_id1, person_id2, ...]}
        self.merge_mapping = {}

        # 存储合并历史记录
        self.merge_history = []

        # 存储人员/分组名称
        # 格式: {merged_person_id (str): "张三"}
        self.names = {}

        # 加载已有的确认数据
        if self.confirmation_file and self.confirmation_file.exists():
            self.load_confirmations()

        logger.info("人工确认管理器初始化完成")

    def merge_groups(
        self,
        person_ids: List[int],
        merged_person_id: int = None,
        operator: str = "system",
        note: str = "",
        create_anchor: bool = True
    ) -> int:
        """
        合并多个人员组
        支持从已有合并组中重新组合（拆分/重组）

        Args:
            person_ids: 要合并的person_id列表
            merged_person_id: 合并后的ID（如果为None则自动生成）
            operator: 操作者
            note: 备注
            create_anchor: 是否同时创建锚点（默认True）

        Returns:
            int: 合并后的person_id
        """
        if len(person_ids) < 2:
            raise ValueError("至少需要2个person_id才能合并")

        # 新逻辑：允许重新组合
        # 如果 person_id 已在其他合并组中，先从原组中移除
        removed_from_groups = []
        for pid in person_ids:
            for merged_id, original_ids in list(self.merge_mapping.items()):
                if pid in original_ids:
                    # 从原合并组中移除
                    original_ids.remove(pid)
                    removed_from_groups.append((merged_id, pid))
                    logger.info(f"从合并组 {merged_id} 中移除 person_id={pid}")

                    # 如果原合并组只剩1个或0个，删除该合并组
                    if len(original_ids) <= 1:
                        logger.info(f"合并组 {merged_id} 人数不足，删除该组")
                        del self.merge_mapping[merged_id]
                    else:
                        # 更新原合并组
                        self.merge_mapping[merged_id] = original_ids

        # 生成合并ID
        if merged_person_id is None:
            merged_person_id = self._generate_merged_id()

        # 记录合并关系
        self.merge_mapping[merged_person_id] = sorted(person_ids)

        # 记录操作历史
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': 'merge',
            'merged_person_id': merged_person_id,
            'original_person_ids': person_ids,
            'operator': operator,
            'note': note,
            'create_anchor': create_anchor
        }
        self.merge_history.append(history_entry)

        logger.info(
            f"合并完成: person_ids={person_ids} -> merged_person_id={merged_person_id}"
        )

        return merged_person_id

    def unmerge_group(
        self,
        merged_person_id: int,
        operator: str = "system",
        note: str = ""
    ) -> List[int]:
        """
        取消合并

        Args:
            merged_person_id: 要取消的合并ID
            operator: 操作者
            note: 备注

        Returns:
            List[int]: 恢复的原始person_id列表
        """
        if merged_person_id not in self.merge_mapping:
            raise ValueError(f"merged_person_id {merged_person_id} 不存在")

        # 获取原始IDs
        original_ids = self.merge_mapping.pop(merged_person_id)

        # 记录操作历史
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': 'unmerge',
            'merged_person_id': merged_person_id,
            'original_person_ids': original_ids,
            'operator': operator,
            'note': note
        }
        self.merge_history.append(history_entry)

        logger.info(
            f"取消合并: merged_person_id={merged_person_id} -> person_ids={original_ids}"
        )

        return original_ids

    def get_merged_id(self, person_id: int) -> int:
        """
        获取person_id对应的merged_person_id

        Args:
            person_id: 原始person_id

        Returns:
            int: merged_person_id（如果未合并则返回原始ID）
        """
        for merged_id, original_ids in self.merge_mapping.items():
            if person_id in original_ids:
                return merged_id
        return person_id

    def get_original_ids(self, merged_person_id: int) -> List[int]:
        """
        获取merged_person_id对应的原始person_id列表

        Args:
            merged_person_id: 合并后的ID

        Returns:
            List[int]: 原始person_id列表
        """
        if merged_person_id in self.merge_mapping:
            return self.merge_mapping[merged_person_id]
        return [merged_person_id]

    def set_name(self, person_id: int, name: str) -> int:
        """
        为person_id设置名称

        Args:
            person_id: 人员ID (可以是原始ID或合并后的ID)
            name: 人员姓名

        Returns:
            int: 该person_id所属的merged_id
        """
        # 获取该person_id所属的merged_id
        merged_id = self.get_merged_id(person_id)
        self.names[str(merged_id)] = name
        logger.info(f"设置名称: merged_id={merged_id}, name={name}")
        return merged_id

    def get_name(self, person_id: int) -> str:
        """
        获取person_id的名称

        Args:
            person_id: 人员ID

        Returns:
            str: 人员姓名（如果未设置则返回空字符串）
        """
        merged_id = self.get_merged_id(person_id)
        return self.names.get(str(merged_id), "")

    def apply_confirmations_to_results(self, results: Dict) -> Dict:
        """
        将人工确认应用到分析结果中

        Args:
            results: ReID分析结果

        Returns:
            Dict: 应用人工确认后的结果
        """
        logger.info("应用人工确认到结果...")

        # 深拷贝结果以避免修改原数据
        import copy
        confirmed_results = copy.deepcopy(results)

        # 创建person_id到group的映射
        person_to_group = {}
        for group in results['groups']:
            for person in group['persons']:
                person_to_group[person['person_id']] = group

        # 应用合并关系
        merged_groups = {}
        used_person_ids = set()

        for merged_id, original_ids in self.merge_mapping.items():
            # 收集所有被合并的人员信息
            merged_persons = []
            for pid in original_ids:
                if pid in person_to_group:
                    # 找到对应的人员信息
                    for person in person_to_group[pid]['persons']:
                        if person['person_id'] == pid:
                            # 添加merged_person_id字段
                            person_copy = person.copy()
                            person_copy['merged_person_id'] = merged_id
                            merged_persons.append(person_copy)
                            used_person_ids.add(pid)
                            break

            if merged_persons:
                merged_groups[merged_id] = merged_persons

        # 重建groups列表
        new_groups = []

        # 添加合并后的组
        for merged_id, persons in merged_groups.items():
            group_info = {
                'group_id': merged_id,
                'merged_group': True,
                'person_count': len(persons),
                'avg_similarity': self._calculate_avg_similarity(persons, results),
                'persons': persons
            }
            new_groups.append(group_info)

        # 添加未被合并的原始组
        for group in results['groups']:
            # 检查组中是否有未被合并的人员
            unmerged_persons = []
            for person in group['persons']:
                if person['person_id'] not in used_person_ids:
                    person_copy = person.copy()
                    person_copy['merged_person_id'] = person['person_id']
                    unmerged_persons.append(person_copy)

            if unmerged_persons:
                group_copy = group.copy()
                group_copy['merged_group'] = False
                group_copy['persons'] = unmerged_persons
                group_copy['person_count'] = len(unmerged_persons)
                new_groups.append(group_copy)

        # 按人数降序排序
        new_groups.sort(key=lambda x: x['person_count'], reverse=True)

        # 重新分配连续的group_id（从0开始）
        for idx, group in enumerate(new_groups):
            group['group_id'] = idx

        # 更新结果
        confirmed_results['groups'] = new_groups
        confirmed_results['total_groups'] = len(new_groups)
        confirmed_results['manual_confirmations_applied'] = True
        confirmed_results['merge_count'] = len(self.merge_mapping)

        logger.info(
            f"人工确认应用完成: 原始组数={len(results['groups'])}, "
            f"确认后组数={len(new_groups)}, "
            f"合并操作数={len(self.merge_mapping)}"
        )

        return confirmed_results

    def save_confirmations(self, filepath: Union[str, Path] = None):
        """
        保存人工确认数据

        Args:
            filepath: 保存路径（如果为None则使用初始化时的路径）
        """
        filepath = Path(filepath) if filepath else self.confirmation_file
        if filepath is None:
            raise ValueError("未指定保存路径")

        data = {
            'version': '1.0',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'merge_mapping': self.merge_mapping,
            'merge_history': self.merge_history,
            'names': self.names
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"人工确认数据已保存: {filepath}")

    def load_confirmations(self, filepath: Union[str, Path] = None):
        """
        加载人工确认数据

        Args:
            filepath: 加载路径（如果为None则使用初始化时的路径）
        """
        filepath = Path(filepath) if filepath else self.confirmation_file
        if filepath is None or not filepath.exists():
            logger.warning("人工确认文件不存在")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换键为整数（JSON会将整数键转为字符串）
        self.merge_mapping = {
            int(k): v for k, v in data.get('merge_mapping', {}).items()
        }
        self.merge_history = data.get('merge_history', [])
        self.names = data.get('names', {})

        logger.info(
            f"人工确认数据已加载: {len(self.merge_mapping)} 个合并关系, "
            f"{len(self.merge_history)} 条历史记录"
        )

    def get_merge_suggestions(
        self,
        results: Dict,
        similarity_threshold: float = 0.6,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        基于相似度提供合并建议

        Args:
            results: ReID分析结果
            similarity_threshold: 高相似度阈值
            min_similarity: 最低相似度阈值

        Returns:
            List[Dict]: 合并建议列表
        """
        suggestions = []

        # 遍历所有组对，寻找可能需要合并的组
        groups = results.get('groups', [])

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group_i = groups[i]
                group_j = groups[j]

                # 计算两组之间的平均相似度
                avg_sim = self._calculate_cross_group_similarity(
                    group_i, group_j, results
                )

                if avg_sim >= min_similarity:
                    suggestion = {
                        'group_id_1': group_i['group_id'],
                        'group_id_2': group_j['group_id'],
                        'avg_similarity': avg_sim,
                        'confidence': 'high' if avg_sim >= similarity_threshold else 'medium',
                        'person_count_1': group_i['person_count'],
                        'person_count_2': group_j['person_count']
                    }
                    suggestions.append(suggestion)

        # 按相似度降序排序
        suggestions.sort(key=lambda x: x['avg_similarity'], reverse=True)

        logger.info(f"生成了 {len(suggestions)} 个合并建议")

        return suggestions

    def _is_already_merged(self, person_id: int) -> bool:
        """检查person_id是否已在合并关系中"""
        for original_ids in self.merge_mapping.values():
            if person_id in original_ids:
                return True
        return False

    def _generate_merged_id(self) -> int:
        """生成新的merged_person_id"""
        if not self.merge_mapping:
            return 100000  # 从100000开始以区分原始ID
        return max(self.merge_mapping.keys()) + 1

    def _calculate_avg_similarity(self, persons: List[Dict], results: Dict) -> float:
        """计算组内平均相似度（从相似度矩阵计算）"""
        import numpy as np
        from pathlib import Path

        # 单人组相似度为1
        if len(persons) == 1:
            return 1.0

        # 尝试加载相似度矩阵
        try:
            similarity_path = Path(self.config.RESULTS_DIR) / "similarity_matrix.npy"
            if not similarity_path.exists():
                logger.warning("相似度矩阵文件不存在，使用默认值0.0")
                return 0.0

            similarity_matrix = np.load(similarity_path)

            # 计算组内所有人员对的相似度
            person_ids = [p['person_id'] for p in persons]
            similarities = []

            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    pid_i = person_ids[i]
                    pid_j = person_ids[j]
                    if pid_i < similarity_matrix.shape[0] and pid_j < similarity_matrix.shape[1]:
                        sim = similarity_matrix[pid_i][pid_j]
                        if sim >= 0:  # 排除同图约束标记（-1）
                            similarities.append(sim)

            if similarities:
                return float(np.mean(similarities))
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"计算相似度时出错: {e}")
            return 0.0

    def _calculate_cross_group_similarity(
        self,
        group1: Dict,
        group2: Dict,
        results: Dict
    ) -> float:
        """计算两组之间的平均相似度"""
        # 这里返回默认值，实际应用时可以从相似度矩阵计算
        return 0.0

    def export_merge_report(self, filepath: Union[str, Path]):
        """
        导出合并报告

        Args:
            filepath: 报告保存路径
        """
        report = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_merges': len(self.merge_mapping),
            'merge_details': []
        }

        for merged_id, original_ids in self.merge_mapping.items():
            detail = {
                'merged_person_id': merged_id,
                'original_person_ids': original_ids,
                'person_count': len(original_ids)
            }
            report['merge_details'].append(detail)

        report['merge_history'] = self.merge_history

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"合并报告已导出: {filepath}")

    def clear_confirmations(self):
        """清除所有人工确认数据"""
        self.merge_mapping.clear()
        self.merge_history.clear()
        logger.info("已清除所有人工确认数据")

    def get_statistics(self) -> Dict:
        """
        获取人工确认统计信息

        Returns:
            Dict: 统计信息
        """
        total_original_persons = sum(len(ids) for ids in self.merge_mapping.values())

        stats = {
            'total_merge_groups': len(self.merge_mapping),
            'total_original_persons': total_original_persons,
            'total_operations': len(self.merge_history),
            'avg_persons_per_merge': (
                total_original_persons / len(self.merge_mapping)
                if self.merge_mapping else 0
            )
        }

        return stats
