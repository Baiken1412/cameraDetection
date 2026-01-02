"""
人员管理模块
提供人员标注、命名、合并等功能
"""
from typing import List, Dict, Optional
from loguru import logger
from reid_database import ReIDDatabase


class PersonManager:
    """人员管理器"""
    
    def __init__(self):
        """初始化人员管理器"""
        self.reid_db = ReIDDatabase()
    
    def is_enabled(self) -> bool:
        """检查ReID数据库是否可用"""
        return self.reid_db.enabled
    
    def get_all_groups(self) -> List[Dict]:
        """
        获取所有人员分组
        
        Returns:
            分组列表，每个分组包含group_id, group_name, person_count, persons等信息
        """
        if not self.is_enabled():
            logger.warning("ReID数据库未启用，无法获取分组")
            return []
        
        try:
            groups = self.reid_db.get_all_groups()
            return groups
        except Exception as e:
            logger.error(f"获取分组失败: {e}")
            return []
    
    def set_person_name(self, person_id: int, name: str) -> bool:
        """
        为人员设置名称
        
        Args:
            person_id: 人员ID（可以是person_id或merged_person_id）
            name: 人员姓名
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.is_enabled():
            logger.warning("ReID数据库未启用，无法设置名称")
            return False
        
        if not name or not name.strip():
            logger.warning("人员姓名不能为空")
            return False
        
        return self.reid_db.set_person_name(person_id, name.strip())
    
    def merge_persons(self, person_ids: List[int], operator: str = "user") -> bool:
        """
        合并多个人员为同一人
        
        Args:
            person_ids: 要合并的人员ID列表（至少2个）
            operator: 操作者
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.is_enabled():
            logger.warning("ReID数据库未启用，无法合并人员")
            return False
        
        if len(person_ids) < 2:
            logger.warning("至少需要2个person_id才能合并")
            return False
        
        return self.reid_db.merge_persons(person_ids, operator)
    
    def unmerge_persons(self, merged_person_id: int, operator: str = "user") -> bool:
        """
        取消人员合并
        
        Args:
            merged_person_id: 合并后的人员ID
            operator: 操作者
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.is_enabled():
            logger.warning("ReID数据库未启用，无法取消合并")
            return False
        
        return self.reid_db.unmerge_persons(merged_person_id, operator)
    
    def get_person_name(self, person_id: int) -> str:
        """
        获取人员名称
        
        Args:
            person_id: 人员ID
            
        Returns:
            人员姓名，如果未设置返回空字符串
        """
        if not self.is_enabled():
            return ""
        
        return self.reid_db.get_person_name(person_id)
    
    def get_person_info(self, person_id: int) -> Optional[Dict]:
        """
        获取人员详细信息
        
        Args:
            person_id: 人员ID
            
        Returns:
            人员信息字典，包含person_id, name, image_path等
        """
        if not self.is_enabled():
            return None
        
        try:
            persons = self.reid_db.get_all_persons()
            for person in persons:
                if person.person_id == person_id:
                    name = self.get_person_name(person_id)
                    return {
                        'person_id': person.person_id,
                        'name': name,
                        'image_path': person.image_path,
                        'crop_path': person.crop_path,
                        'confidence': person.confidence
                    }
            return None
        except Exception as e:
            logger.error(f"获取人员信息失败: {e}")
            return None
    
    def search_persons_by_name(self, name: str) -> List[Dict]:
        """
        根据姓名搜索人员
        
        Args:
            name: 人员姓名（支持模糊搜索）
            
        Returns:
            匹配的人员列表
        """
        if not self.is_enabled():
            return []
        
        try:
            groups = self.get_all_groups()
            results = []
            name_lower = name.lower()
            
            for group in groups:
                group_name = group.get('group_name', '')
                if name_lower in group_name.lower():
                    # 找到匹配的分组，返回该分组的所有人员
                    for person in group.get('persons', []):
                        results.append({
                            'person_id': person['person_id'],
                            'name': group_name,
                            'image_path': person.get('image_path', ''),
                            'crop_path': person.get('crop_path', ''),
                            'group_id': group.get('group_id')
                        })
            
            return results
        except Exception as e:
            logger.error(f"搜索人员失败: {e}")
            return []

