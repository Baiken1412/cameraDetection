"""
待确认队列管理模块
管理需要人工确认的人员匹配
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PendingQueue:
    """
    待确认队列管理器

    管理处于"待确认"状态的人员匹配：
    - 相似度在 threshold_low 和 threshold_high 之间
    - 需要人工确认是否归入某个锚点
    """

    def __init__(self, queue_file: Path = None):
        """
        初始化待确认队列

        Args:
            queue_file: 队列数据保存文件路径
        """
        self.queue_file = Path(queue_file) if queue_file else None

        # 待确认项目列表
        # [{
        #     'person_id': 10,
        #     'suggested_anchor_id': 100000,
        #     'similarity': 0.65,
        #     'image_id': 'image_005',
        #     'crop_path': '...',
        #     'created_at': '2025-12-13',
        #     'status': 'pending'  # 'pending' | 'confirmed' | 'rejected'
        # }]
        self.queue = []

        # 加载已有队列
        if self.queue_file and self.queue_file.exists():
            self.load_queue()

        logger.info("待确认队列初始化完成")

    def add_pending_item(
        self,
        person_id: int,
        suggested_anchor_id: int,
        similarity: float,
        image_id: str,
        crop_path: str,
        metadata: Dict = None
    ):
        """
        添加待确认项

        Args:
            person_id: 人员ID
            suggested_anchor_id: 建议的锚点ID
            similarity: 相似度分数
            image_id: 图片ID
            crop_path: 裁剪图片路径
            metadata: 其他元数据
        """
        item = {
            'person_id': person_id,
            'suggested_anchor_id': suggested_anchor_id,
            'similarity': float(similarity),
            'image_id': image_id,
            'crop_path': crop_path,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'pending',
            'metadata': metadata or {}
        }

        self.queue.append(item)

        logger.info(
            f"添加待确认项: person_id={person_id}, "
            f"suggested_anchor={suggested_anchor_id}, "
            f"similarity={similarity:.3f}"
        )

    def get_pending_items(
        self,
        status: str = 'pending',
        limit: int = None
    ) -> List[Dict]:
        """
        获取待确认项

        Args:
            status: 状态过滤 ('pending' | 'confirmed' | 'rejected' | 'all')
            limit: 返回数量限制

        Returns:
            List[Dict]: 待确认项列表
        """
        if status == 'all':
            items = self.queue
        else:
            items = [item for item in self.queue if item['status'] == status]

        if limit:
            items = items[:limit]

        return items

    def confirm_item(
        self,
        person_id: int,
        confirmed_anchor_id: int = None
    ) -> Dict:
        """
        确认待确认项

        Args:
            person_id: 待确认的person_id
            confirmed_anchor_id: 确认的锚点ID（None表示拒绝建议）

        Returns:
            Dict: 确认的项目信息
        """
        for item in self.queue:
            if item['person_id'] == person_id and item['status'] == 'pending':
                if confirmed_anchor_id is not None:
                    item['status'] = 'confirmed'
                    item['confirmed_anchor_id'] = confirmed_anchor_id
                    logger.info(
                        f"确认待确认项: person_id={person_id} -> "
                        f"anchor_id={confirmed_anchor_id}"
                    )
                else:
                    item['status'] = 'rejected'
                    logger.info(f"拒绝待确认项: person_id={person_id}")

                item['confirmed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                return item

        raise ValueError(f"未找到待确认的 person_id={person_id}")

    def batch_confirm(self, confirmations: List[Dict]) -> int:
        """
        批量确认

        Args:
            confirmations: 确认列表
                [{'person_id': 10, 'confirmed_anchor_id': 100000}, ...]

        Returns:
            int: 成功确认的数量
        """
        count = 0
        for conf in confirmations:
            try:
                self.confirm_item(
                    person_id=conf['person_id'],
                    confirmed_anchor_id=conf.get('confirmed_anchor_id')
                )
                count += 1
            except Exception as e:
                logger.warning(f"批量确认失败: {e}")

        logger.info(f"批量确认完成: {count}/{len(confirmations)}")
        return count

    def remove_item(self, person_id: int):
        """
        移除待确认项

        Args:
            person_id: 要移除的person_id
        """
        original_len = len(self.queue)
        self.queue = [item for item in self.queue if item['person_id'] != person_id]

        removed = original_len - len(self.queue)
        if removed > 0:
            logger.info(f"移除待确认项: person_id={person_id}, 数量={removed}")

    def get_queue_stats(self) -> Dict:
        """
        获取队列统计信息

        Returns:
            Dict: 统计信息
        """
        pending_items = [item for item in self.queue if item['status'] == 'pending']
        confirmed_items = [item for item in self.queue if item['status'] == 'confirmed']
        rejected_items = [item for item in self.queue if item['status'] == 'rejected']

        # 统计每个锚点的待确认数量
        anchor_distribution = {}
        for item in pending_items:
            anchor_id = item['suggested_anchor_id']
            anchor_distribution[anchor_id] = anchor_distribution.get(anchor_id, 0) + 1

        stats = {
            'total_items': len(self.queue),
            'pending_count': len(pending_items),
            'confirmed_count': len(confirmed_items),
            'rejected_count': len(rejected_items),
            'anchor_distribution': anchor_distribution,
            'oldest_pending': (
                min([item['created_at'] for item in pending_items])
                if pending_items else None
            )
        }

        return stats

    def clear_processed_items(self, keep_days: int = 7):
        """
        清理已处理的项目（保留最近N天）

        Args:
            keep_days: 保留天数
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        original_len = len(self.queue)

        # 保留待确认的和最近处理的
        self.queue = [
            item for item in self.queue
            if item['status'] == 'pending' or
            item.get('confirmed_at', '9999-99-99') >= cutoff_str
        ]

        removed = original_len - len(self.queue)
        if removed > 0:
            logger.info(f"清理已处理项目: 移除 {removed} 项")

    def get_items_by_anchor(
        self,
        anchor_id: int,
        status: str = 'pending'
    ) -> List[Dict]:
        """
        获取特定锚点的待确认项

        Args:
            anchor_id: 锚点ID
            status: 状态过滤

        Returns:
            List[Dict]: 待确认项列表
        """
        items = []
        for item in self.queue:
            if (item['suggested_anchor_id'] == anchor_id and
                (status == 'all' or item['status'] == status)):
                items.append(item)

        return items

    def save_queue(self, filepath: Path = None):
        """
        保存队列数据

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath) if filepath else self.queue_file
        if filepath is None:
            raise ValueError("未指定保存路径")

        data = {
            'version': '1.0',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'queue': self.queue
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"待确认队列已保存: {filepath}")

    def load_queue(self, filepath: Path = None):
        """
        加载队列数据

        Args:
            filepath: 加载路径
        """
        filepath = Path(filepath) if filepath else self.queue_file
        if filepath is None or not filepath.exists():
            logger.warning("待确认队列文件不存在")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.queue = data.get('queue', [])

        logger.info(f"待确认队列已加载: {len(self.queue)} 项")

    def export_pending_report(self, filepath: Path):
        """
        导出待确认报告

        Args:
            filepath: 报告保存路径
        """
        stats = self.get_queue_stats()

        report = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'pending_items': self.get_pending_items(status='pending')
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"待确认报告已导出: {filepath}")
