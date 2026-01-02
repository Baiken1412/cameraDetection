"""
数据库版本的人工确认管理器
使用MySQL数据库代替JSON文件
"""
import logging
from typing import List, Dict
# 从reid_database_modu导入DatabaseManager，避免与caseapp的database模块冲突
from reid_database_modu.database_manager import DatabaseManager
from reid_config import Config

logger = logging.getLogger(__name__)


class DatabaseConfirmationManager:
    """
    数据库版本的人工确认管理器
    所有数据存储在MySQL数据库中
    """

    def __init__(self):
        """初始化数据库管理器"""
        self.db = DatabaseManager()
        logger.info("数据库确认管理器初始化完成")

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
        """
        if len(person_ids) < 2:
            raise ValueError("至少需要2个person_id才能合并")

        # 获取现有的合并关系
        existing_mappings = self.db.get_all_merge_mappings()

        # 从已有的合并组中移除这些person_id
        for existing_merged_id, existing_ids in list(existing_mappings.items()):
            removed_ids = []
            for pid in person_ids:
                if pid in existing_ids:
                    existing_ids.remove(pid)
                    removed_ids.append(pid)
                    logger.info(f"从合并组 {existing_merged_id} 中移除 person_id={pid}")

            # 如果原合并组人数不足，删除该组
            if len(existing_ids) <= 1:
                logger.info(f"合并组 {existing_merged_id} 人数不足，删除该组")
                self.db.delete_merge_mapping(existing_merged_id)
            elif removed_ids:
                # 更新原合并组
                self.db.save_merge_mapping(existing_merged_id, existing_ids)

        # 生成新的合并ID
        if merged_person_id is None:
            # 获取最大ID
            all_ids = list(existing_mappings.keys())
            merged_person_id = max(all_ids) + 1 if all_ids else 100000

        # 保存新的合并关系
        self.db.save_merge_mapping(merged_person_id, sorted(person_ids))

        # 添加历史记录
        self.db.add_merge_history(
            operation='merge',
            merged_person_id=merged_person_id,
            person_ids=person_ids,
            operator=operator,
            note=note
        )

        # 同步更新数据库的 groups 和 group_members 表
        self.db.update_groups_after_merge(merged_person_id, person_ids)

        logger.info(f"合并完成: person_ids={person_ids} -> merged_person_id={merged_person_id}")

        return merged_person_id

    def unmerge_group(
        self,
        merged_person_id: int,
        operator: str = "system",
        note: str = ""
    ) -> List[int]:
        """取消合并"""
        mappings = self.db.get_all_merge_mappings()

        if merged_person_id not in mappings:
            raise ValueError(f"merged_person_id {merged_person_id} 不存在")

        original_ids = mappings[merged_person_id]

        # 删除合并关系
        self.db.delete_merge_mapping(merged_person_id)

        # 添加历史记录
        self.db.add_merge_history(
            operation='unmerge',
            merged_person_id=merged_person_id,
            person_ids=original_ids,
            operator=operator,
            note=note
        )

        # 同步更新数据库的 groups 和 group_members 表
        self.db.update_groups_after_unmerge(merged_person_id, original_ids)

        logger.info(f"取消合并: merged_person_id={merged_person_id} -> person_ids={original_ids}")

        return original_ids

    def get_merged_id(self, person_id: int) -> int:
        """获取person_id对应的merged_person_id"""
        mappings = self.db.get_all_merge_mappings()

        for merged_id, original_ids in mappings.items():
            if person_id in original_ids:
                return merged_id
        return person_id

    def get_original_ids(self, merged_person_id: int) -> List[int]:
        """获取merged_person_id对应的原始person_id列表"""
        mappings = self.db.get_all_merge_mappings()

        if merged_person_id in mappings:
            return mappings[merged_person_id]
        return [merged_person_id]

    def set_name(self, person_id: int, name: str) -> int:
        """设置人员名称"""
        merged_id = self.get_merged_id(person_id)
        self.db.set_person_name(merged_id, name)
        logger.info(f"设置名称: merged_id={merged_id}, name={name}")
        return merged_id

    def get_name(self, person_id: int) -> str:
        """获取人员名称"""
        merged_id = self.get_merged_id(person_id)
        return self.db.get_person_name(merged_id)

    def apply_confirmations_to_results(self, results: Dict) -> Dict:
        """将人工确认应用到分析结果中"""
        import copy
        import numpy as np
        from pathlib import Path

        confirmed_results = copy.deepcopy(results)

        # 获取所有合并关系
        merge_mapping = self.db.get_all_merge_mappings()

        if not merge_mapping:
            # 如果没有合并关系，直接返回原结果
            return confirmed_results

        logger.info("应用人工确认到结果...")

        # 尝试加载相似度矩阵以计算合并组的真实相似度
        similarity_matrix = None
        try:
            from reid_config import Config
            similarity_path = Path(Config.RESULTS_DIR) / "similarity_matrix.npy"
            if similarity_path.exists():
                similarity_matrix = np.load(similarity_path)
                logger.info("已加载相似度矩阵用于计算合并组相似度")
        except Exception as e:
            logger.warning(f"无法加载相似度矩阵: {e}")

        # 创建person_id到group的映射
        person_to_group = {}
        for group in results['groups']:
            for person in group['persons']:
                person_to_group[person['person_id']] = group

        # 应用合并关系
        merged_groups = {}
        used_person_ids = set()

        for merged_id, original_ids in merge_mapping.items():
            merged_persons = []
            for pid in original_ids:
                if pid in person_to_group:
                    for person in person_to_group[pid]['persons']:
                        if person['person_id'] == pid:
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
            # 计算合并组的平均相似度
            avg_similarity = 0.0
            if similarity_matrix is not None and len(persons) > 1:
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
                    avg_similarity = float(np.mean(similarities))
            elif len(persons) == 1:
                avg_similarity = 1.0  # 单人组相似度为1

            group_info = {
                'group_id': merged_id,
                'merged_group': True,
                'person_count': len(persons),
                'avg_similarity': avg_similarity,
                'persons': persons
            }
            new_groups.append(group_info)

        # 添加未被合并的原始组
        for group in results['groups']:
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
        confirmed_results['merge_count'] = len(merge_mapping)

        logger.info(
            f"人工确认应用完成: 原始组数={len(results['groups'])}, "
            f"确认后组数={len(new_groups)}, "
            f"合并操作数={len(merge_mapping)}"
        )

        return confirmed_results

    def save_confirmations(self):
        """数据库版本不需要显式保存（自动保存）"""
        pass

    def load_confirmations(self):
        """数据库版本不需要显式加载（自动从数据库读取）"""
        pass

    def clear_confirmations(self):
        """清除所有人工确认数据"""
        logger.warning("正在清空所有确认数据...")

        # 获取所有合并关系
        mappings = self.db.get_all_merge_mappings()

        # 逐个取消合并
        for merged_id in list(mappings.keys()):
            try:
                self.unmerge_group(
                    merged_person_id=merged_id,
                    operator='system',
                    note='清空所有确认数据'
                )
            except Exception as e:
                logger.error(f"取消合并 {merged_id} 失败: {e}")

        logger.info(f"已清空 {len(mappings)} 个合并关系")

    def get_statistics(self) -> Dict:
        """获取人工确认统计信息"""
        mappings = self.db.get_all_merge_mappings()
        total_original_persons = sum(len(ids) for ids in mappings.values())

        stats = {
            'total_merge_groups': len(mappings),
            'total_original_persons': total_original_persons,
            'avg_persons_per_merge': (
                total_original_persons / len(mappings)
                if mappings else 0
            )
        }

        return stats

    @property
    def merge_mapping(self):
        """兼容性属性：返回合并映射"""
        return self.db.get_all_merge_mappings()

    @property
    def names(self):
        """兼容性属性：返回所有名称"""
        return self.db.get_all_names()
