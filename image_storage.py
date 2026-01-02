"""
图片存储模块
支持保存到文件系统或转换为Base64存储到数据库
"""
import cv2
import base64
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
import config
from utils.image_processor import save_image as cv2_save_image


class ImageStorage:
    """图片存储类"""
    
    def __init__(self):
        self.config = config.RTSP_MONITOR_CONFIG
        self.save_to_file = self.config['save_image_to_file']
        image_save_path = self.config['image_save_path']
        self.image_url_prefix = self.config.get('image_url_prefix', '')
        
        # 转换为绝对路径，确保路径正确
        if not os.path.isabs(image_save_path):
            # 相对路径：基于脚本所在目录
            script_dir = Path(__file__).parent.absolute()
            self.image_save_path = script_dir / image_save_path
        else:
            self.image_save_path = Path(image_save_path)
        
        # 确保保存目录存在
        self.image_save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"图片保存目录: {self.image_save_path.absolute()}")
        if self.image_url_prefix:
            logger.info(f"图片URL前缀: {self.image_url_prefix}")
    
    def save_image(self, frame, camera_id: int, camera_name: str, capture_time: datetime = None) -> str:
        """
        保存图片并返回存储路径或Base64编码

        Args:
            frame: 视频帧（numpy数组）
            camera_id: 摄像头ID
            camera_name: 摄像头名称
            capture_time: 帧的实际拍摄时间（用于生成准确的文件名），None则使用当前时间

        Returns:
            str: 文件路径或Base64编码字符串
        """
        try:
            if self.save_to_file:
                # 保存到文件系统
                return self._save_to_file(frame, camera_id, camera_name, capture_time)
            else:
                # 转换为Base64
                return self._convert_to_base64(frame)
        except Exception as e:
            logger.error(f"保存图片失败，摄像头ID: {camera_id}, 名称: {camera_name}: {e}")
            return None
    
    def _save_to_file(self, frame, camera_id: int, camera_name: str, capture_time: datetime = None) -> str:
        """
        保存图片到文件系统

        Args:
            frame: 视频帧
            camera_id: 摄像头ID
            camera_name: 摄像头名称
            capture_time: 帧的实际拍摄时间，None则使用当前时间

        Returns:
            str: 相对路径
        """
        try:
            # 检查帧是否有效
            if frame is None:
                logger.error("帧数据为None，无法保存")
                return None

            # 检查帧数据类型（兼容numpy数组和OpenCV的UMat）
            import numpy as np
            is_valid_frame = False
            try:
                # 检查是否为numpy数组
                if isinstance(frame, np.ndarray):
                    is_valid_frame = True
                # 检查是否为OpenCV的UMat
                elif hasattr(cv2, 'UMat') and isinstance(frame, cv2.UMat):
                    is_valid_frame = True
                # 检查是否有shape属性（兼容其他数组类型）
                elif hasattr(frame, 'shape') and hasattr(frame, 'dtype'):
                    is_valid_frame = True
            except Exception:
                pass

            if not is_valid_frame:
                logger.error(f"帧数据格式无效: {type(frame)}")
                return None

            # 检查帧尺寸
            if len(frame.shape) < 2:
                logger.error(f"帧尺寸无效: {frame.shape}")
                return None

            height, width = frame.shape[:2]
            if height == 0 or width == 0:
                logger.error(f"帧尺寸为0: {width}x{height}")
                return None

            # 检查帧尺寸是否合理（过小的可能是损坏的）
            if height < 100 or width < 100:
                logger.warning(f"帧尺寸过小，可能损坏: {width}x{height}")
                return None

            # 检查帧数据是否异常（检查是否有大量NaN或Inf）
            import numpy as np
            if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
                logger.error("帧包含NaN或Inf值，数据损坏")
                return None

            # 使用传入的拍摄时间，如果没有则使用当前时间
            actual_time = capture_time if capture_time is not None else datetime.now()

            # 创建日期目录：yyyyMMdd
            date_dir = actual_time.strftime('%Y%m%d')
            save_dir = Path(self.image_save_path) / date_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名：摄像头ID_摄像头名称_时间戳.jpg
            # ✅ 使用准确的拍摄时间，而不是当前系统时间
            timestamp = actual_time.strftime('%Y%m%d%H%M%S')
            # 清理文件名中的非法字符
            safe_name = self._sanitize_filename(camera_name)
            filename = f"{camera_id}_{safe_name}_{timestamp}.jpg"
            
            file_path = save_dir / filename
            success = False
            
            # 方法1: 尝试使用工具函数保存（支持中文路径）
            try:
                cv2_success = cv2_save_image(frame, file_path, quality=85)

                # 立即验证文件是否真的存在
                if cv2_success and file_path.exists():
                    try:
                        file_size = file_path.stat().st_size
                        # 检查文件大小，确保不是空文件
                        if file_size > 0:
                            success = True
                            logger.debug(f"使用cv2工具函数成功保存图片: {file_path} (大小: {file_size} 字节)")
                        else:
                            # 创建了空文件，删除它以便PIL重试
                            logger.warning(f"cv2工具函数创建了空文件，将删除并尝试PIL保存: {file_path}")
                            try:
                                file_path.unlink()
                            except:
                                pass
                    except Exception as size_error:
                        logger.warning(f"检查cv2保存的文件大小失败: {size_error}, 将尝试PIL保存")

                if not success:
                    logger.warning(
                        f"cv2工具函数保存失败或文件未生成 (返回值={cv2_success}, 文件存在={file_path.exists()}): {file_path}"
                    )
            except Exception as cv2_error:
                logger.warning(f"cv2工具函数保存异常: {cv2_error}, 路径: {file_path}")
            
            # 方法2: 如果cv2保存失败，尝试使用PIL/Pillow
            if not success:
                try:
                    from PIL import Image
                    import numpy as np
                    
                    logger.info(f"尝试使用PIL保存图片: {file_path}")
                    
                    # 转换BGR到RGB（OpenCV使用BGR，PIL使用RGB）
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    # 确保数据类型正确
                    if isinstance(frame_rgb, np.ndarray):
                        # 确保数据类型为uint8
                        if frame_rgb.dtype != np.uint8:
                            frame_rgb = frame_rgb.astype(np.uint8)
                        pil_image = Image.fromarray(frame_rgb)
                    else:
                        frame_array = np.array(frame_rgb)
                        if frame_array.dtype != np.uint8:
                            frame_array = frame_array.astype(np.uint8)
                        pil_image = Image.fromarray(frame_array)
                    
                    # 保存
                    pil_image.save(str(file_path), 'JPEG', quality=85, optimize=True)
                    
                    # 验证PIL保存是否成功
                    if file_path.exists() and file_path.stat().st_size > 0:
                        success = True
                        logger.info(f"使用PIL成功保存图片: {file_path}")
                    else:
                        logger.error(f"PIL保存后文件不存在或大小为0: {file_path}")
                        
                except Exception as pil_error:
                    logger.error(
                        f"PIL保存失败: {pil_error}, 类型: {type(pil_error).__name__}, "
                        f"路径: {file_path}", 
                        exc_info=True
                    )
            
            # 最终验证：如果仍然失败，检查可能的原因
            if not success or not file_path.exists():
                # 检查目录权限和磁盘空间
                try:
                    import shutil
                    disk_usage = shutil.disk_usage(save_dir)
                    free_space_mb = disk_usage.free / (1024 * 1024)
                    logger.error(
                        f"图片保存最终失败 - 路径: {file_path}, "
                        f"目录可写: {os.access(save_dir, os.W_OK)}, "
                        f"剩余空间: {free_space_mb:.2f} MB"
                    )
                except Exception as check_error:
                    logger.error(f"无法检查磁盘状态: {check_error}")
                
                return None
            
            # 最终检查文件大小（双重验证）
            try:
                file_size = file_path.stat().st_size
                if file_size == 0:
                    logger.error(f"保存的文件大小为0，删除空文件: {file_path}")
                    try:
                        file_path.unlink()  # 删除空文件
                    except:
                        pass
                    return None
                elif file_size < 100:  # 正常的JPEG图片应该至少几百字节
                    logger.warning(f"保存的文件大小异常小 ({file_size} 字节): {file_path}")
            except Exception as size_check_error:
                logger.error(f"检查文件大小失败: {size_check_error}, 路径: {file_path}")
                return None
            
            # 返回相对路径（日期目录/文件名，如：20251212/29_摄像头名称_20251212144441.jpg）
            relative_path = f"{date_dir}/{filename}"
            logger.info(f"图片已保存: {self.image_save_path / relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"保存图片到文件系统失败: {e}", exc_info=True)
            return None
    
    def _convert_to_base64(self, frame) -> str:
        """
        将图片转换为Base64编码
        
        Args:
            frame: 视频帧
        
        Returns:
            str: Base64编码字符串
        """
        try:
            # 编码为JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                logger.error("图片编码失败")
                return None
            
            # 转换为Base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
            
        except Exception as e:
            logger.error(f"图片Base64编码失败: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名中的非法字符
        
        Args:
            filename: 原始文件名
        
        Returns:
            str: 清理后的文件名
        """
        if not filename:
            return "unknown"
        
        # 保留中文字符、英文字母、数字、下划线和连字符
        # 特别注意：Windows文件名不能包含 / \ : * ? " < > |
        import re
        # 先替换Windows不允许的字符
        safe_name = filename.replace('/', '_').replace('\\', '_')
        safe_name = safe_name.replace(':', '_').replace('*', '_')
        safe_name = safe_name.replace('?', '_').replace('"', '_')
        safe_name = safe_name.replace('<', '_').replace('>', '_')
        safe_name = safe_name.replace('|', '_')
        
        # 再替换其他特殊字符
        safe_name = re.sub(r'[^\w\u4e00-\u9fa5_-]', '_', safe_name)
        return safe_name[:50]  # 限制长度
    
    def get_image_url(self, relative_path: str) -> str:
        """
        将相对路径转换为完整的URL路径
        
        Args:
            relative_path: 相对路径（如：20251212/29_摄像头名称_20251212144441.jpg）
        
        Returns:
            str: 完整的URL路径（如：https://localhost:8090/profile/sacwspbj/20251212/29_摄像头名称_20251212144441.jpg）
        """
        if not self.image_url_prefix:
            # 如果没有配置URL前缀，直接返回相对路径
            return relative_path
        
        # 确保URL前缀以 / 结尾
        url_prefix = self.image_url_prefix.rstrip('/') + '/'
        
        # 确保相对路径不以 / 开头
        relative_path = relative_path.lstrip('/')
        
        # 拼接完整URL
        full_url = url_prefix + relative_path
        return full_url

