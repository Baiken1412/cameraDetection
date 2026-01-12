"""
ReID人员识别集成模块
用于在检测到人员后自动识别人员姓名
"""
import sys
import os
import tempfile
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from loguru import logger
import config_loader as config
from utils.image_processor import load_image

# 直接使用本地模块（不再依赖外部路径）
# ReID系统核心模块已整合到caseapp项目中
REID_AVAILABLE = False
REID_IMPORT_ERROR = None
try:
    from core.detector import PersonDetector
    from core.feature_extractor import ReIDFeatureExtractor
    REID_AVAILABLE = True
except ImportError as e:
    REID_IMPORT_ERROR = str(e)
    logger.warning(f"ReID系统模块导入失败，人员识别功能将不可用: {e}")
    logger.debug(f"导入错误详情: {e}", exc_info=True)
    REID_AVAILABLE = False
except Exception as e:
    REID_IMPORT_ERROR = str(e)
    logger.error(f"ReID系统模块导入时发生未知错误: {e}", exc_info=True)
    REID_AVAILABLE = False

# 导入ReID数据库模块（多数据源支持）
try:
    from reid_database import ReIDDatabase
    REID_DB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ReID数据库模块导入失败: {e}")
    REID_DB_AVAILABLE = False


class ReIDIntegration:
    """ReID人员识别集成类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ReIDIntegration, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化ReID系统"""
        if self._initialized:
            return
        
        self._initialized = True
        self.enabled = False
        self.detector = None
        self.extractor = None
        self.reid_db = None  # 使用 caseapp 的 ReIDDatabase
        self.match_threshold = config.REID_CONFIG.get('match_threshold', 0.7)
        
        # 检查配置是否启用
        if not config.REID_CONFIG.get('enabled', True):
            logger.info("ReID功能已在配置中禁用")
            return
        
        if not REID_AVAILABLE:
            logger.warning("ReID系统不可用，人员识别功能已禁用")
            return
        
        try:
            # 初始化 ReID 数据库连接（使用 caseapp 的配置）
            if REID_DB_AVAILABLE:
                self.reid_db = ReIDDatabase()
                if not self.reid_db.enabled:
                    logger.warning("ReID数据库未启用，人员识别功能将不可用")
                    return
            
            # 初始化检测器和特征提取器
            logger.info("正在初始化ReID系统...")
            self.detector = PersonDetector()
            self.extractor = ReIDFeatureExtractor()
            self.match_threshold = config.REID_CONFIG.get('match_threshold', 0.7)
            self.enabled = True
            logger.info("✓ ReID系统初始化成功")
        except Exception as e:
            logger.error(f"ReID系统初始化失败: {e}", exc_info=True)
            self.enabled = False
    
    def identify_person_from_image_path(self, image_path: str) -> Optional[str]:
        """
        从图片路径识别人员姓名
        
        Args:
            image_path: 图片文件路径（本地文件系统路径）
            
        Returns:
            人员姓名（如果匹配成功），否则返回None
        """
        if not self.enabled:
            return None
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.warning(f"图片文件不存在: {image_path}")
                return None
            
            # 读取图片（支持中文路径）
            try:
                frame = load_image(image_path)
            except Exception as e:
                logger.warning(f"无法读取图片: {image_path}, 错误: {e}")
                return None
            
            # 使用现有的从frame识别的方法
            return self.identify_person_from_frame(frame)
        except Exception as e:
            logger.error(f"从图片路径识别人员失败 ({image_path}): {e}", exc_info=True)
            return None
    
    def identify_person_from_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        从视频帧识别人员姓名
        
        Args:
            frame: OpenCV视频帧（numpy数组，BGR格式）
            
        Returns:
            人员姓名（如果匹配成功），否则返回None
        """
        if not self.enabled:
            return None
        
        try:
            # 1. 检测人员
            # YOLOv8需要文件路径，先保存临时图片
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, frame)
            
            try:
                detections = self.detector.detect(tmp_path)
            finally:
                # 删除临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            if not detections:
                logger.debug("未检测到人员")
                return None
            
            # 2. 选择最大的人员区域（假设主要人员是最大的）
            largest_detection = max(detections, key=lambda d: 
                (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
            
            bbox = largest_detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 边界检查
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning("无效的裁剪区域")
                return None
            
            # 裁剪人员区域
            person_crop = frame[y1:y2, x1:x2]
            
            # 3. 提取特征
            feature = self.extractor.extract_feature(person_crop)
            
            # 4. 与数据库中的已命名人员匹配
            person_name = self._match_person(feature)
            
            return person_name
            
        except Exception as e:
            logger.error(f"人员识别失败: {e}", exc_info=True)
            return None
    
    def _match_person(self, feature: np.ndarray) -> Optional[str]:
        """
        与数据库中的已命名人员匹配
        
        Args:
            feature: 人员特征向量
            
        Returns:
            人员姓名（如果匹配成功），否则返回None
        """
        try:
            # 使用ReID数据库模块（多数据源支持）
            if not self.reid_db or not self.reid_db.enabled:
                logger.debug("ReID数据库未启用，无法匹配人员")
                return None
            
            # 从数据库获取所有已命名的人员
            # 说明：当前项目已将 ReID 数据库完整整合到 caseapp/reid_database_modu，
            # 不再需要兼容外部的 person-re-identification-system-main 的 database 包结构，
            # 因此这里直接从本地 reid_database_modu.models 导入模型类，避免一切与 database.* 的冲突。
            from reid_database_modu.models import PersonName, Person

            # 获取数据库会话
            session = self.reid_db.db_manager.get_session()

            try:
                # 查询所有已命名的人员
                named_persons = session.query(PersonName).all()

                # 调试：打印当前从 ReID 数据库读取到的全部命名记录
                try:
                    persons_debug = ", ".join(
                        f"{p.id}(merged_id={p.merged_person_id}, name={p.name})"
                        for p in named_persons
                    ) or "<无记录>"
                    logger.info(f"ReID数据库当前 person_names 记录: {persons_debug}")
                except Exception as debug_err:
                    logger.warning(f"打印 person_names 调试信息失败: {debug_err}")
                
                if not named_persons:
                    logger.debug("数据库中无已命名人员")
                    return None
                
                # 获取特征文件路径（使用配置适配器）
                from reid_config_adapter import Config as ReIDConfig
                features_path = Path(ReIDConfig.FEATURES_DIR) / "features.npy"
                metadata_path = Path(ReIDConfig.RESULTS_DIR) / "metadata.json"

                # 详细日志，帮助定位 'NoneType'.read 异常
                logger.debug(
                    f"准备加载特征和元数据文件: features_path={features_path} "
                    f"(exists={features_path.exists()}), "
                    f"metadata_path={metadata_path} (exists={metadata_path.exists()})"
                )
                
                # 加载特征和元数据
                if not features_path.exists() or not metadata_path.exists():
                    logger.warning(f"特征文件不存在: {features_path} 或 {metadata_path}")
                    return None
                
                # 加载特征矩阵
                try:
                    logger.debug(f"开始加载特征矩阵: {features_path}")
                    all_features = np.load(str(features_path))
                    # 检查加载结果是否为 None（虽然 np.load 通常不会返回 None，但为了安全）
                    if all_features is None:
                        logger.error(f"特征文件加载结果为空: {features_path}")
                        return None
                    logger.debug(f"特征矩阵加载完成，shape={getattr(all_features, 'shape', None)}")
                except FileNotFoundError as load_feat_err:
                    logger.error(f"特征文件不存在: {features_path}")
                    return None
                except Exception as load_feat_err:
                    logger.error(
                        f"加载特征文件失败 ({features_path}): {load_feat_err}",
                        exc_info=True,
                    )
                    # 显式抛出，以便在日志中准确看到调用栈位置
                    raise
                
                # 加载元数据
                import json
                try:
                    logger.debug(f"开始加载元数据: {metadata_path}")
                    # 使用 with open 确保文件正确关闭，并检查文件对象是否为 None
                    f = None
                    try:
                        f = open(metadata_path, 'r', encoding='utf-8')
                        if f is None:
                            logger.error(f"无法打开元数据文件: {metadata_path}")
                            return None
                        metadata = json.load(f)
                        # 检查元数据是否为 None
                        if metadata is None:
                            logger.error(f"元数据文件内容为空: {metadata_path}")
                            return None
                        logger.debug(f"元数据加载完成，条目数={len(metadata) if isinstance(metadata, (list, dict)) else 'unknown'}")
                    finally:
                        if f is not None:
                            f.close()
                except FileNotFoundError as load_meta_err:
                    logger.error(f"元数据文件不存在: {metadata_path}")
                    return None
                except json.JSONDecodeError as load_meta_err:
                    logger.error(f"元数据文件JSON格式错误: {metadata_path}, 错误: {load_meta_err}")
                    return None
                except Exception as load_meta_err:
                    logger.error(
                        f"加载元数据文件失败 ({metadata_path}): {load_meta_err}",
                        exc_info=True,
                    )
                    raise
                
                # 构建person_id到特征的映射
                person_id_to_feature = {}
                for i, meta in enumerate(metadata):
                    if i < len(all_features):
                        person_id_to_feature[meta['person_id']] = all_features[i]
                
                # 计算与所有已命名人员的相似度
                best_match = None
                best_similarity = 0.0
                match_threshold = self.match_threshold
                
                for named_person in named_persons:
                    merged_id = named_person.merged_person_id
                    person_name = named_person.name
                    
                    # 获取该merged_id对应的所有person_id
                    original_ids = self.reid_db.confirmation_manager.get_original_ids(merged_id)
                    
                    # 计算与这些person_id的平均相似度
                    similarities = []
                    for person_id in original_ids:
                        if person_id in person_id_to_feature:
                            person_feature = person_id_to_feature[person_id]
                            similarity = float(np.dot(feature, person_feature))
                            similarities.append(similarity)
                    
                    if similarities:
                        avg_similarity = float(np.mean(similarities))
                        if avg_similarity > best_similarity:
                            best_similarity = avg_similarity
                            best_match = person_name
                
                # 如果相似度超过阈值，返回人员姓名
                if best_match and best_similarity >= match_threshold:
                    logger.info(f"匹配到人员: {best_match}, 相似度: {best_similarity:.3f}")
                    return best_match
                else:
                    logger.debug(f"未找到匹配人员，最高相似度: {best_similarity:.3f}")
                    return None
                    
            finally:
                self.reid_db.db_manager.close_session()
                
        except Exception as e:
            logger.error(f"人员匹配失败: {e}", exc_info=True)
            return None

