"""
单摄像头监测模块
每个摄像头一个独立的监测任务
使用背景建模法进行检测
"""
import cv2
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import config
from database import Database
from image_detection import ImageChangeDetection
from image_storage import ImageStorage
import av  # PyAV for RTSP with PTS support


class CameraMonitor:
    """单个摄像头监测类"""
    
    def __init__(self, camera_info: dict, db: Database):
        """
        初始化摄像头监测
        
        Args:
            camera_info: 摄像头配置信息
            db: 数据库连接对象
        """
        self.camera_info = camera_info
        self.camera_id = camera_info['id']
        self.camera_name = camera_info['fjmc']  # 摄像头名称，用于保存到 sxtmx 字段
        self.area_name = camera_info.get('gnslx', '')  # 区域名称，用于保存到 qymc 字段
        self.rtsp_url = camera_info['rtspssl']

        # 检查是否为USB摄像头
        self.is_usb_camera = self.rtsp_url.startswith('usb:')

        # 记录完整的URL用于调试
        if self.is_usb_camera:
            usb_device_id = self.rtsp_url.split(':')[1] if ':' in self.rtsp_url else '0'
            logger.info(f"摄像头 {self.camera_name} (ID: {self.camera_id}) USB设备ID: {usb_device_id}")
        else:
            logger.info(f"摄像头 {self.camera_name} (ID: {self.camera_id}) RTSP URL: {self.rtsp_url}")

            # 检查URL是否完整
            if not self.rtsp_url or len(self.rtsp_url.strip()) == 0:
                logger.error(f"摄像头 {self.camera_name} RTSP URL为空！")
            elif 'rtsp://' not in self.rtsp_url.lower():
                logger.warning(f"摄像头 {self.camera_name} RTSP URL格式可能不正确: {self.rtsp_url[:100]}")
        
        self.db = db
        self.detection = ImageChangeDetection()
        self.storage = ImageStorage()
        # YOLO 人员检测器（懒加载）
        self.person_detector = None
        
        self.config = config.RTSP_MONITOR_CONFIG
        self.running = False
        self.monitor_thread = None

        # 记录上次保存的时间（用于避免频繁保存，不依赖数据库查询）
        self.last_save_time = None

        # 当前活动轨迹缓存（用于30秒窗口内的轨迹合并）
        # 格式: {'record_id': int, 'jscs': int, 'last_time': datetime}
        self.current_trajectory = None

        # OpenCV VideoCapture (for USB cameras or fallback)
        self.cap = None

        # PyAV container and stream (for RTSP with PTS)
        self.av_container = None
        self.av_stream = None
        self.av_decoder = None
        self.time_base = None  # 视频流的时间基准
        self.pts_base_time = None  # PTS到真实时间的映射基准 (datetime)
        self.pts_base_offset = None  # PTS基准偏移量 (秒)
        self.use_pyav = False  # 是否使用PyAV（RTSP流使用，USB摄像头不使用）

        self.reconnect_attempts = 0
    
    def start(self):
        """启动监测"""
        if self.running:
            logger.warning(f"摄像头 {self.camera_name} 已在监测中")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"CameraMonitor-{self.camera_id}",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"开始监测摄像头: {self.camera_name} (ID: {self.camera_id})")
    
    def stop(self):
        """停止监测"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self._release_capture()
        logger.info(f"停止监测摄像头: {self.camera_name} (ID: {self.camera_id})")
    
    def _monitor_loop(self):
        """监测主循环"""
        while self.running and self.reconnect_attempts < self.config['max_reconnect_attempts']:
            try:
                # 连接RTSP流
                if not self._connect_rtsp():
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts < self.config['max_reconnect_attempts']:
                        logger.warning(
                            f"摄像头 {self.camera_name} 连接失败，"
                            f"{self.config['reconnect_interval']}秒后重试 "
                            f"(尝试 {self.reconnect_attempts}/{self.config['max_reconnect_attempts']})"
                        )
                        time.sleep(self.config['reconnect_interval'])
                    continue
                
                # 连接成功，重置重连计数
                self.reconnect_attempts = 0
                
                # 跳过前几帧，让摄像头稳定
                logger.info(f"摄像头 {self.camera_name} 正在初始化，跳过前3帧...")
                for _ in range(3):
                    ret, frame, frame_time = self._read_frame_with_pts()
                    if ret and frame is not None:
                        pass  # 背景建模不需要保存前一帧
                    time.sleep(0.5)
                
                # 背景建模需要更多时间学习背景
                # 重要：学习阶段应该确保画面中没有运动物体
                # 如果是重新连接（背景模型已存在），缩短学习时间
                if self.detection.bg_initialized and self.detection.bg_subtractor is not None:
                    # 重新连接，背景模型已存在，只需短暂学习适应
                    logger.info(f"摄像头 {self.camera_name} 重新连接，背景模型已存在，快速适应中（10秒）...")
                    learning_time = 10
                else:
                    # 首次连接，需要完整学习
                    logger.info(f"摄像头 {self.camera_name} 背景建模学习中，请等待30秒...")
                    logger.warning(f"⚠️  重要提示：学习阶段请确保画面中没有运动物体，否则会影响背景模型！")
                    learning_time = 30
                
                # 初始化背景建模器
                if not self.detection.bg_initialized:
                    bg_type = self.config.get('bg_subtractor_type', 'MOG2')
                    if bg_type == 'MOG2':
                        history = self.config.get('bg_history', 500)
                        var_threshold = self.config.get('bg_var_threshold', 16)
                        detect_shadows = self.config.get('bg_detect_shadows', True)
                        self.detection.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                            history=history,
                            varThreshold=var_threshold,
                            detectShadows=detect_shadows
                        )
                    else:  # KNN
                        history = self.config.get('bg_history', 500)
                        dist2_threshold = self.config.get('bg_var_threshold', 400)
                        detect_shadows = self.config.get('bg_detect_shadows', True)
                        self.detection.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                            history=history,
                            dist2Threshold=dist2_threshold,
                            detectShadows=detect_shadows
                        )
                    self.detection.bg_initialized = True
                    logger.info(f"背景建模器初始化完成，类型: {bg_type}")
                
                # 学习阶段使用更高的学习率，快速建立背景模型
                learning_phase_rate = 0.1  # 学习阶段使用10%的学习率
                
                for i in range(learning_time):  # 学习时间根据情况调整
                    ret, frame, frame_time = self._read_frame_with_pts()
                    if ret and frame is not None:
                        try:
                            # 使用高学习率学习背景（学习阶段不进行形态学和过滤）
                            fg_mask = self.detection.bg_subtractor.apply(frame, learningRate=learning_phase_rate)
                            
                            # 每5秒输出一次学习进度
                            if (i + 1) % 5 == 0:
                                import numpy as np
                                fg_pixels = np.count_nonzero(fg_mask)
                                total_pixels = fg_mask.size
                                fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0
                                logger.info(f"背景建模学习进度: {i+1}/{learning_time}秒, 当前前景比例: {fg_ratio*100:.3f}%")
                                
                                # 如果前景比例很高，说明学习阶段画面中有运动物体
                                if fg_ratio > 0.1:  # 超过10%
                                    logger.warning(f"⚠️  学习阶段检测到大量前景 ({fg_ratio*100:.1f}%)，可能影响背景模型质量！")
                        except Exception as e:
                            logger.error(f"背景建模学习异常: {e}")
                    
                    time.sleep(1)
                
                logger.info(f"摄像头 {self.camera_name} 背景建模学习完成，开始正常监测...")
                
                frame_count = 0
                # 开始监测循环
                while self.running:
                    try:
                        # 读取一帧及其时间戳
                        ret, frame, frame_time = self._read_frame_with_pts()

                        if not ret or frame is None:
                            logger.warning(f"摄像头 {self.camera_name} 获取帧失败，尝试重连")
                            break
                        
                        # 验证帧数据有效性
                        try:
                            import numpy as np
                            if not isinstance(frame, np.ndarray):
                                logger.warning(f"摄像头 {self.camera_name} 帧数据格式异常，跳过")
                                time.sleep(0.1)
                                continue
                            
                            if len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
                                logger.warning(f"摄像头 {self.camera_name} 帧尺寸异常: {frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                                time.sleep(0.1)
                                continue
                        except Exception as frame_check_error:
                            logger.warning(f"摄像头 {self.camera_name} 帧验证失败: {frame_check_error}")
                            time.sleep(0.1)
                            continue
                        
                        frame_count += 1
                        
                        # 检测画面变化（使用背景建模）
                        has_change = self.detection.detect_change(frame)
                        
                        # 获取计算好的前景比例（从缓存中获取，避免重复计算）
                        change_ratio = self.detection.last_change_ratio
                        
                        if change_ratio is not None:
                            # 输出前景信息（每次检测都输出，方便观察）
                            change_percent = change_ratio * 100
                            threshold_percent = self.detection.threshold * 100
                            consecutive_count = self.detection.consecutive_change_count
                            consecutive_threshold = self.detection.consecutive_frames_threshold
                            
                            # 如果前景比例一直为0，输出警告
                            if change_ratio == 0 and frame_count % 30 == 0:  # 每30帧输出一次
                                logger.warning(
                                    f"⚠️  [{self.camera_name}] 前景比例持续为0！"
                                    f"可能原因：1)学习阶段画面中有运动物体 2)方差阈值太高(当前: {self.config.get('bg_var_threshold', 16)}) "
                                    f"3)背景模型未正确建立。建议：降低bg_var_threshold到10-12，或重新启动程序确保学习阶段画面静止"
                                )
                            
                            if has_change:
                                logger.info(
                                    f"✓ [{self.camera_name}] 检测到画面变化！"
                                    f" 前景比例: {change_percent:.3f}% (阈值: {threshold_percent:.3f}%, "
                                    f"连续帧: {consecutive_count}/{consecutive_threshold})"
                                )
                                # 检测到变化，获取最新帧并处理
                                logger.info(f"摄像头 {self.camera_name} 检测到变化，正在处理...")
                                # 重要：清空缓冲区，读取最新的帧，避免保存旧帧
                                # RTSP流是连续连接的，缓冲区可能积累了很多旧帧
                                # 需要读取并丢弃旧帧，确保获取到最新的帧
                                # 使用PTS时间作为帧的拍摄时间（如果可用），否则使用系统时间
                                save_frame, frame_capture_time = self._get_latest_frame_with_time()
                                if save_frame is not None:
                                    # 先用YOLO再次核实当前帧中是否有人
                                    people_count_for_merge = self._count_people_in_frame(save_frame)
                                    if people_count_for_merge is not None and people_count_for_merge <= 0:
                                        logger.info(
                                            f"YOLO核实当前帧无人员，本次变化视为非人员事件，"
                                            f"不保存/合并记录 - 摄像头: {self.camera_name}"
                                        )
                                        # 重置连续帧计数，让后续检测重新开始
                                        self.detection.consecutive_change_count = 0
                                        continue  # 继续监测循环，不断开、不等待

                                    # 直接保存检测结果（新逻辑使用30秒窗口自动合并）
                                    logger.info(f"检测到人员，保存检测结果...")
                                    # 使用帧捕获时的时间（frame_capture_time），而不是保存时的时间
                                    self._save_detection_result(save_frame, frame_capture_time)

                                    # 重置连续帧计数，避免后续继续触发
                                    self.detection.consecutive_change_count = 0

                                    # 新逻辑：不断开连接，继续监测
                                    # 如果之后又检测到人，会根据时间间隔自动合并或创建新记录
                                    logger.info(f"已处理检测结果，继续监测（不断开连接）...")
                                    # 继续监测循环，不需要break
                                else:
                                    logger.warning(f"摄像头 {self.camera_name} 无法获取最新帧，跳过保存")
                            else:
                                # 判断当前帧是否超过阈值（但不一定触发检测，因为需要连续帧）
                                frame_exceeds_threshold = change_ratio > self.detection.threshold
                                
                                if frame_exceeds_threshold:
                                    # 当前帧超过阈值，但需要连续帧才触发
                                    logger.info(
                                        f"[{self.camera_name}] 前景比例: {change_percent:.3f}% "
                                        f"(阈值: {threshold_percent:.3f}%, "
                                        f"连续帧: {consecutive_count}/{consecutive_threshold}, 状态: 等待确认)"
                                    )
                                else:
                                    # 输出当前前景比例（INFO级别，方便观察）
                                    logger.info(
                                        f"[{self.camera_name}] 前景比例: {change_percent:.3f}% "
                                        f"(阈值: {threshold_percent:.3f}%, 状态: 正常)"
                                    )
                        else:
                            # 无法计算前景比例（可能是初始化中）
                            if frame_count <= 20:
                                logger.debug(f"摄像头 {self.camera_name} 背景建模初始化中，等待下一帧...")
                        
                        # 每50帧输出一次详细状态
                        if frame_count % 50 == 0:
                            logger.debug(f"摄像头 {self.camera_name} 已处理 {frame_count} 帧，监测正常...")
                        
                        # 动态采样间隔：如果检测到变化，等待更长时间；否则正常采样间隔
                        if has_change:
                            # 检测到有人，等待5秒后再检测
                            wait_time = self.config.get('detection_wait_interval', 5)
                            logger.debug(f"[{self.camera_name}] 检测到画面变化，等待 {wait_time} 秒后进行下一次检测...")
                            time.sleep(wait_time)
                        else:
                            # 正常情况，每秒检测一次
                            time.sleep(self.config['sample_interval'])
                        
                    except Exception as e:
                        logger.error(f"监测过程中发生异常 - 摄像头: {self.camera_name}: {e}")
                        # 继续监测，不中断
                        time.sleep(1)
                
                # 释放资源
                self._release_capture()
                
            except Exception as e:
                logger.error(f"摄像头 {self.camera_name} 监测异常: {e}")
                self.reconnect_attempts += 1
                
                if self.running and self.reconnect_attempts < self.config['max_reconnect_attempts']:
                    time.sleep(self.config['reconnect_interval'])
        
        if self.reconnect_attempts >= self.config['max_reconnect_attempts']:
            logger.error(f"摄像头 {self.camera_name} 重连次数已达上限，停止监测")
        
        self.running = False
    
    def _generate_alternative_urls(self, base_url: str) -> list:
        """
        生成多种RTSP URL格式，用于自动尝试不同的URL路径
        
        Args:
            base_url: 原始RTSP URL，格式如: rtsp://user:pass@ip:port/path
            
        Returns:
            list: 多种URL格式的列表
        """
        import re
        from urllib.parse import urlparse, unquote
        
        # 先尝试URL解码，处理可能的编码问题
        try:
            decoded_url = unquote(base_url)
            if decoded_url != base_url:
                logger.debug(f"URL解码: {base_url[:60]}... -> {decoded_url[:60]}...")
                base_url = decoded_url
        except Exception as e:
            logger.debug(f"URL解码失败，使用原始URL: {e}")
        
        # 解析原始URL
        try:
            parsed = urlparse(base_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
        except Exception as e:
            logger.error(f"解析RTSP URL失败: {e}, URL: {base_url[:100]}")
            # 如果解析失败，只返回原始URL
            return [base_url]
        
        # 提取通道号（如果存在）
        channel_match = re.search(r'channel[=:](\d+)', base_url, re.IGNORECASE)
        channel = int(channel_match.group(1)) if channel_match else 1
        
        # 检查原始URL是否已经包含subtype参数
        has_subtype = 'subtype' in base_url.lower()
        
        alternative_urls = []
        
        # 1. 原始URL（保持原样，优先尝试）
        alternative_urls.append(base_url)
        
        # 如果原始URL已经包含subtype参数，确保它是完整的
        if has_subtype and 'subtype=' in base_url:
            # 检查subtype参数是否完整
            subtype_match = re.search(r'subtype[=:](\d+)', base_url, re.IGNORECASE)
            if not subtype_match:
                # subtype参数不完整，尝试补充
                if '&subtype' in base_url or '?subtype' in base_url:
                    # 已经有subtype但值不完整，尝试添加值
                    base_url_with_subtype = re.sub(r'[?&]subtype[=:][^&]*', f'&subtype=1', base_url)
                    if base_url_with_subtype != base_url:
                        alternative_urls.append(base_url_with_subtype)
        
        # 2. 海康威视格式变体
        # /cam/realmonitor?channel=1&subtype=0 (主码流)
        alternative_urls.append(f"{base}/cam/realmonitor?channel={channel}&subtype=0")
        alternative_urls.append(f"{base}/cam/realmonitor?channel={channel}&subtype=1")  # 子码流
        
        # 3. 海康威视新格式
        # /Streaming/Channels/101 (通道1主码流), /Streaming/Channels/201 (通道2主码流)
        alternative_urls.append(f"{base}/Streaming/Channels/{channel}01")
        alternative_urls.append(f"{base}/Streaming/Channels/{channel}02")  # 子码流
        
        # 4. 尝试通道1（如果原始是通道2）
        if channel != 1:
            alternative_urls.append(f"{base}/cam/realmonitor?channel=1&subtype=0")
            alternative_urls.append(f"{base}/cam/realmonitor?channel=1&subtype=1")
            alternative_urls.append(f"{base}/Streaming/Channels/101")
            alternative_urls.append(f"{base}/Streaming/Channels/102")
        
        # 5. 大华格式
        alternative_urls.append(f"{base}/h264/ch{channel}/main/av_stream")
        alternative_urls.append(f"{base}/h264/ch{channel}/sub/av_stream")
        
        # 6. 通用格式
        alternative_urls.append(f"{base}/live")
        alternative_urls.append(f"{base}/stream1")
        
        # 去重并保持顺序
        seen = set()
        unique_urls = []
        for url in alternative_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        logger.debug(f"为摄像头 {self.camera_name} 生成了 {len(unique_urls)} 种URL格式")
        return unique_urls
    
    def _prepare_rtsp_url(self, url: str, transport: str = 'tcp') -> str:
        """
        准备RTSP URL，添加必要的参数以提高连接稳定性
        
        Args:
            url: 原始RTSP URL
            transport: 传输协议，'tcp' 或 'udp'，默认'tcp'
            
        Returns:
            str: 处理后的RTSP URL
        """
        # 如果URL已经包含rtsp_transport参数，检查是否需要替换
        if '?rtsp_transport=' in url or '&rtsp_transport=' in url:
            # 如果需要使用不同的传输协议，替换它
            if transport != 'tcp':
                import re
                url = re.sub(r'[?&]rtsp_transport=[^&]*', f'?rtsp_transport={transport}' if '?' not in url[:url.find('rtsp_transport')] else f'&rtsp_transport={transport}', url)
            return url
        
        # 添加传输协议参数
        # 先检查URL是否已有其他参数
        separator = '&' if '?' in url else '?'
        # 使用指定传输协议
        url_with_transport = f"{url}{separator}rtsp_transport={transport}"
        
        logger.debug(f"RTSP URL处理 ({transport}): {url[:50]}... -> {url_with_transport[:60]}...")
        return url_with_transport

    def _connect_usb_camera(self) -> bool:
        """
        连接USB摄像头

        Returns:
            bool: 连接成功返回True
        """
        try:
            # 从rtsp_url中提取USB设备ID（格式: usb:0）
            usb_device_id = int(self.rtsp_url.split(':')[1]) if ':' in self.rtsp_url else 0

            logger.info(f"正在连接USB摄像头: {self.camera_name} (设备ID: {usb_device_id})")

            # 创建VideoCapture对象
            self.cap = cv2.VideoCapture(usb_device_id)

            # 等待摄像头初始化
            time.sleep(1.0)

            if not self.cap.isOpened():
                logger.error(f"USB摄像头打开失败: {self.camera_name} (设备ID: {usb_device_id})")
                self._release_capture()
                return False

            # 设置摄像头参数（可选）
            try:
                # 设置分辨率（可根据需要调整）
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # 设置缓冲区大小
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                logger.debug(f"设置USB摄像头参数失败（将使用默认参数）: {e}")

            # 测试读取帧
            success_count = 0
            test_frames = 5

            for i in range(test_frames):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    success_count += 1
                    if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        logger.debug(
                            f"USB摄像头测试帧 {i+1}/{test_frames} 读取成功 - "
                            f"摄像头: {self.camera_name}, 尺寸: {frame.shape[1]}x{frame.shape[0]}"
                        )
                    else:
                        logger.warning(f"USB摄像头测试帧 {i+1} 尺寸异常: {frame.shape if frame is not None else 'None'}")
                else:
                    logger.debug(f"USB摄像头测试帧 {i+1}/{test_frames} 读取失败")
                time.sleep(0.1)

            # 至少需要3帧成功
            if success_count >= 3:
                logger.info(
                    f"USB摄像头连接成功 - 摄像头: {self.camera_name} (设备ID: {usb_device_id}), "
                    f"成功读取 {success_count}/{test_frames} 测试帧"
                )
                return True
            else:
                logger.error(
                    f"USB摄像头连接失败 - 摄像头: {self.camera_name} (设备ID: {usb_device_id}), "
                    f"仅成功读取 {success_count}/{test_frames} 测试帧"
                )
                self._release_capture()
                return False

        except ValueError as e:
            logger.error(f"USB摄像头设备ID格式错误: {self.rtsp_url}, 错误: {e}")
            self._release_capture()
            return False
        except Exception as e:
            logger.error(f"连接USB摄像头异常: {e}", exc_info=True)
            self._release_capture()
            return False

    def _connect_rtsp(self) -> bool:
        """
        连接RTSP流或USB摄像头，自动尝试多种URL格式

        Returns:
            bool: 连接成功返回True
        """
        try:
            # 释放之前的连接
            self._release_capture()

            # 如果是USB摄像头，使用专门的连接方法
            if self.is_usb_camera:
                return self._connect_usb_camera()
            
            # 最多尝试两轮：第一轮使用当前 rtsp_url，失败后通过接口刷新并再尝试一轮
            for round_idx in range(2):
                # 生成多种URL格式
                alternative_urls = self._generate_alternative_urls(self.rtsp_url)
                logger.info(
                    f"为摄像头 {self.camera_name} 生成 {len(alternative_urls)} 种URL格式进行尝试 "
                    f"(第 {round_idx + 1}/2 轮)"
                )
                
                # 尝试连接，首先使用TCP传输
                connection_success = False
                transport_methods = ['tcp', 'udp']  # 先尝试TCP，失败后尝试UDP
                
                for url_variant in alternative_urls:
                    if connection_success:
                        break
                    
                    # 首先尝试不带rtsp_transport参数的原始URL（某些摄像头可能不需要）
                    if url_variant == self.rtsp_url:
                        try:
                            logger.debug(f"尝试原始URL（不带rtsp_transport参数）: {url_variant[:80]}...")
                            self.cap = cv2.VideoCapture(url_variant, cv2.CAP_FFMPEG if hasattr(cv2, 'CAP_FFMPEG') else 1900)
                            time.sleep(2.0)
                            if self.cap.isOpened():
                                ret, test_frame = self.cap.read()
                                if ret and test_frame is not None:
                                    connection_success = True
                                    logger.info(f"RTSP连接成功（原始URL，不带transport参数） - 摄像头: {self.camera_name}")
                                    break
                                self._release_capture()
                        except Exception as e:
                            logger.debug(f"原始URL连接失败: {e}")
                            self._release_capture()
                        
                    for transport in transport_methods:
                        if connection_success:
                            break
                        
                        # 准备RTSP URL，添加传输协议参数
                        rtsp_url = self._prepare_rtsp_url(url_variant, transport)
                        
                        try:
                            # 强制使用FFMPEG后端（避免OpenCV误识别为图像序列）
                            # CAP_FFMPEG = 1900 (OpenCV常量)
                            if hasattr(cv2, 'CAP_FFMPEG'):
                                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                                logger.debug(f"尝试URL格式 ({transport}): {rtsp_url[:80]}...")
                            else:
                                # 如果常量不存在，尝试使用数值
                                self.cap = cv2.VideoCapture(rtsp_url, 1900)  # 1900 = CAP_FFMPEG
                                logger.debug(f"尝试URL格式 ({transport}): {rtsp_url[:80]}...")
                            
                            # 设置RTSP相关属性，提高连接稳定性
                            try:
                                # 设置缓冲区大小（减少延迟）
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                # 设置超时时间（增加到10秒）
                                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                            except (AttributeError, cv2.error):
                                # 某些OpenCV版本可能不支持这些属性
                                pass
                                
                        except (AttributeError, TypeError, cv2.error) as e:
                            # 如果FFMPEG后端不可用，尝试其他方法
                            logger.warning(f"无法使用FFMPEG后端，尝试其他方法 ({transport}): {e}")
                            try:
                                # 尝试使用GSTREAMER后端（如果可用）
                                if hasattr(cv2, 'CAP_GSTREAMER'):
                                    self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_GSTREAMER)
                                    logger.debug(f"尝试使用GSTREAMER后端连接RTSP ({transport}): {self.camera_name}")
                                else:
                                    # 直接创建，让OpenCV自动选择
                                    self.cap = cv2.VideoCapture(rtsp_url)
                                    logger.warning(f"使用默认后端创建VideoCapture ({transport}): {self.camera_name}")
                            except Exception as e2:
                                # 最后尝试直接创建
                                logger.warning(f"后端选择失败，使用默认方式 ({transport}): {e2}")
                                self.cap = cv2.VideoCapture(rtsp_url)
                        
                        # 检查是否成功打开（给更多时间让RTSP连接建立）
                        # RTSP连接需要时间进行握手和协商
                        time.sleep(2.0)  # 增加到2秒，给RTSP连接更多时间
                        
                        if self.cap.isOpened():
                            # 多次尝试读取帧验证连接是否真的可用
                            # RTSP流可能需要几帧才能稳定
                            ret = False
                            test_frame = None
                            for attempt in range(5):  # 尝试读取5次
                                ret, test_frame = self.cap.read()
                                if ret and test_frame is not None:
                                    logger.debug(
                                        f"测试帧 {attempt+1}/5 读取成功 - 摄像头: {self.camera_name}, "
                                        f"尺寸: {test_frame.shape[1]}x{test_frame.shape[0]}"
                                    )
                                    break
                                time.sleep(0.2)  # 每次尝试间隔0.2秒
                            
                            if ret and test_frame is not None:
                                connection_success = True
                                # 如果使用的不是原始URL，更新rtsp_url
                                if url_variant != self.rtsp_url:
                                    logger.info(
                                        f"RTSP连接成功！使用了备用URL格式 ({transport}传输) - 摄像头: {self.camera_name}"
                                    )
                                    logger.info(f"  原始URL: {self.rtsp_url[:80]}...")
                                    logger.info(f"  成功URL: {rtsp_url[:80]}...")
                                else:
                                    logger.info(
                                        f"RTSP连接成功 ({transport}传输) - 摄像头: {self.camera_name}, "
                                        f"URL: {rtsp_url[:70]}..."
                                    )
                                break
                            else:
                                logger.debug(
                                    f"RTSP连接已打开但无法读取帧 ({transport}): {self.camera_name}, "
                                    f"URL: {rtsp_url[:60]}..."
                                )
                                self._release_capture()
                        else:
                            # 输出完整的URL用于调试（仅在DEBUG级别）
                            logger.debug(f"RTSP连接失败 ({transport}传输): {self.camera_name}")
                            logger.debug(f"  完整URL: {rtsp_url}")
                            self._release_capture()
                
                if connection_success:
                    break

                # 如果这一轮连接失败，且是第一轮，则尝试通过接口刷新RTSP地址
                if not connection_success and round_idx == 0:
                    try:
                        logger.warning(
                            f"RTSP连接失败，尝试通过接口刷新摄像头配置后重试 - 摄像头: {self.camera_name} (ID: {self.camera_id})"
                        )
                        camera_info = self.db.get_camera_by_id(self.camera_id)
                        if camera_info and camera_info.get('rtspssl'):
                            new_url = camera_info['rtspssl']
                            if new_url != self.rtsp_url:
                                logger.info(
                                    "更新摄像头RTSP地址: %s (ID: %s)\n  旧URL: %s...\n  新URL: %s...",
                                    self.camera_name,
                                    self.camera_id,
                                    self.rtsp_url[:80],
                                    new_url[:80],
                                )
                                self.rtsp_url = new_url
                                # 如果接口返回了最新的名称/区域，也一并更新
                                self.camera_name = camera_info.get('fjmc', self.camera_name)
                                self.area_name = camera_info.get('gnslx', self.area_name)
                            else:
                                logger.info("接口返回的RTSP地址与当前相同，将直接重试连接")
                        else:
                            logger.warning("接口未返回有效的摄像头配置，无法刷新RTSP地址")
                    except Exception as refresh_err:
                        logger.error(f"刷新摄像头RTSP地址失败: {refresh_err}", exc_info=True)

            if not connection_success:
                logger.error(
                    f"RTSP流连接失败（已尝试多种URL格式和TCP/UDP传输，并刷新过一次接口） - 摄像头: {self.camera_name}"
                )
                logger.error(f"  最后尝试的URL: {self.rtsp_url[:80]}...")
                logger.error("  可能的原因：")
                logger.error("  1. RTSP URL路径不正确或已过期")
                logger.error("  2. 摄像头通道号配置错误")
                logger.error("  3. 摄像头未启用RTSP服务")
                logger.error("  4. 网络连接问题")
                return False
            
            # 设置超时（增加到30秒，因为RTSP连接可能需要更长时间）
            timeout_ms = max(self.config['rtsp_timeout'] * 1000, 30000)  # 至少30秒
            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            except (AttributeError, cv2.error):
                # 某些OpenCV版本可能不支持这些属性
                logger.debug(f"无法设置超时属性（可能版本不支持）: {self.camera_name}")
            
            # 设置缓冲区大小（减少延迟，但可能增加丢帧）
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except (AttributeError, cv2.error):
                logger.debug(f"无法设置缓冲区大小: {self.camera_name}")
            
            # 设置帧率（可选，帮助OpenCV理解流格式）
            try:
                self.cap.set(cv2.CAP_PROP_FPS, 25)  # RTSP通常25-30fps
            except (AttributeError, cv2.error):
                pass
            
            # 测试读取多帧，确保连接稳定
            success_count = 0
            test_frames = 5  # 增加测试帧数，确保连接稳定
            
            for i in range(test_frames):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    success_count += 1
                    # 检查帧是否有效（尺寸合理）
                    if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        logger.debug(
                            f"测试帧 {i+1}/{test_frames} 读取成功 - "
                            f"摄像头: {self.camera_name}, 尺寸: {frame.shape[1]}x{frame.shape[0]}"
                        )
                    else:
                        logger.warning(f"测试帧 {i+1} 尺寸异常: {frame.shape if frame is not None else 'None'}")
                else:
                    logger.debug(f"测试帧 {i+1}/{test_frames} 读取失败 - 摄像头: {self.camera_name}")
                time.sleep(0.1)
            
            # 至少需要3帧成功（提高要求，确保连接稳定）
            if success_count >= 3:
                logger.info(
                    f"RTSP流连接成功 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                    f"成功读取 {success_count}/{test_frames} 测试帧"
                )

                # 连接成功后，尝试使用PyAV重新打开以获取PTS支持
                if not self._init_pyav_stream():
                    logger.warning(
                        f"PyAV初始化失败，将继续使用OpenCV（无PTS支持） - 摄像头: {self.camera_name}"
                    )
                    self.use_pyav = False
                else:
                    logger.info(
                        f"PyAV初始化成功，将使用PTS精确时间戳 - 摄像头: {self.camera_name}"
                    )
                    # 关闭OpenCV连接，改用PyAV
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                    self.use_pyav = True

                return True
            else:
                logger.error(
                    f"RTSP流连接失败 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                    f"仅成功读取 {success_count}/{test_frames} 测试帧"
                )
                self._release_capture()
                return False
                
        except cv2.error as cv_error:
            logger.error(
                f"OpenCV连接RTSP流异常 - 摄像头: {self.camera_name}, "
                f"错误: {cv_error}, URL: {self.rtsp_url[:50]}..."
            )
            self._release_capture()
            return False
        except Exception as e:
            logger.error(
                f"连接RTSP流异常 - 摄像头: {self.camera_name}, "
                f"错误类型: {type(e).__name__}, 错误: {e}, URL: {self.rtsp_url[:50]}...",
                exc_info=True
            )
            self._release_capture()
            return False
    
    def _release_capture(self):
        """释放视频捕获资源"""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        # 释放PyAV资源
        self._release_pyav()

    def _release_pyav(self):
        """释放PyAV资源"""
        if self.av_container is not None:
            try:
                self.av_container.close()
            except:
                pass
            self.av_container = None
        self.av_stream = None
        self.av_decoder = None
        self.time_base = None
        self.pts_base_time = None
        self.pts_base_offset = None

    def _init_pyav_stream(self) -> bool:
        """
        初始化PyAV流以获取PTS支持

        Returns:
            bool: 初始化成功返回True
        """
        try:
            # 释放之前的PyAV资源
            self._release_pyav()

            # 打开RTSP流
            logger.debug(f"正在使用PyAV打开RTSP流: {self.camera_name}")
            self.av_container = av.open(
                self.rtsp_url,
                options={
                    'rtsp_transport': 'tcp',  # 使用TCP传输（更可靠）
                    'max_delay': '500000',  # 最大延迟500ms
                    'stimeout': '5000000',  # 套接字超时5秒
                    'buffer_size': '1024000',  # 缓冲区大小1MB
                    'rtsp_flags': 'prefer_tcp',  # 优先使用TCP
                },
                timeout=10.0  # 连接超时10秒
            )

            # 获取视频流
            if len(self.av_container.streams.video) == 0:
                logger.error(f"RTSP流中没有视频轨道: {self.camera_name}")
                self._release_pyav()
                return False

            self.av_stream = self.av_container.streams.video[0]
            self.time_base = self.av_stream.time_base

            logger.debug(
                f"PyAV视频流信息 - 摄像头: {self.camera_name}, "
                f"编码: {self.av_stream.codec_context.name}, "
                f"分辨率: {self.av_stream.width}x{self.av_stream.height}, "
                f"time_base: {self.time_base}"
            )

            # 创建解码器
            self.av_decoder = self.av_container.decode(self.av_stream)

            # 读取第一帧，建立PTS时间基准
            system_time_before = datetime.now()
            for frame in self.av_decoder:
                if frame.pts is not None:
                    # 建立时间基准：记录第一帧的PTS和对应的系统时间
                    self.pts_base_offset = float(frame.pts * self.time_base)
                    self.pts_base_time = datetime.now()

                    logger.info(
                        f"✓ PTS时间基准已建立 - 摄像头: {self.camera_name}, "
                        f"基准PTS: {self.pts_base_offset:.3f}秒, "
                        f"基准时间: {self.pts_base_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
                    )

                    # 重新创建解码器（因为已经读取了一帧）
                    self.av_decoder = self.av_container.decode(self.av_stream)

                    return True
                else:
                    logger.warning(f"第一帧PTS为空，尝试下一帧: {self.camera_name}")

            logger.error(f"无法获取有效的PTS时间戳: {self.camera_name}")
            self._release_pyav()
            return False

        except Exception as e:
            logger.error(f"初始化PyAV流失败 - 摄像头: {self.camera_name}, 错误: {e}", exc_info=True)
            self._release_pyav()
            return False

    def _pts_to_datetime(self, pts: int) -> datetime:
        """
        将PTS转换为真实时间

        Args:
            pts: Presentation Time Stamp

        Returns:
            datetime: 转换后的真实时间
        """
        if pts is None or self.time_base is None or self.pts_base_time is None:
            return datetime.now()

        # 计算当前帧的PTS秒数
        pts_seconds = float(pts * self.time_base)

        # 转换为真实时间：基准时间 + (当前PTS - 基准PTS)
        real_time = self.pts_base_time + timedelta(seconds=(pts_seconds - self.pts_base_offset))

        return real_time

    def _read_frame_with_pts(self) -> Tuple[bool, Optional[any], Optional[datetime]]:
        """
        读取一帧及其时间戳
        如果使用PyAV，返回帧和PTS时间；否则使用OpenCV返回帧和系统时间

        Returns:
            Tuple[bool, frame, timestamp]: (是否成功, 帧数据, 时间戳)
        """
        if self.use_pyav and self.av_decoder is not None:
            # 使用PyAV读取
            try:
                for frame in self.av_decoder:
                    # 转换为numpy数组 (BGR格式，兼容OpenCV)
                    img = frame.to_ndarray(format='bgr24')

                    # 获取PTS并转换为真实时间
                    frame_time = self._pts_to_datetime(frame.pts)

                    return True, img, frame_time

                # 流结束
                return False, None, None

            except Exception as e:
                # 区分错误类型
                error_str = str(e)
                if '10054' in error_str or 'Connection' in error_str or 'Broken pipe' in error_str:
                    # 网络连接错误，需要重连
                    logger.warning(
                        f"PyAV网络连接中断 - 摄像头: {self.camera_name}, "
                        f"错误: {e}, 将触发重连机制"
                    )
                    # 立即释放PyAV资源，为重连做准备
                    self._release_pyav()
                else:
                    # 其他错误（解码错误等）
                    logger.error(f"PyAV读取帧失败 - 摄像头: {self.camera_name}, 错误: {e}")

                return False, None, None
        else:
            # 使用OpenCV读取（USB摄像头或fallback）
            ret, frame = self.cap.read()
            frame_time = datetime.now() if ret else None
            return ret, frame, frame_time
    
    def _should_record(self) -> bool:
        """
        判断是否需要记录（避免频繁记录）
        优先使用本地记录的时间，避免频繁查询数据库
        
        Returns:
            bool: 需要记录返回True
        """
        try:
            # 优先使用本地记录的时间（更快，更准确）
            if self.last_save_time is not None:
                time_diff = (datetime.now() - self.last_save_time).total_seconds()
                if time_diff >= self.config['min_record_interval']:
                    return True
                else:
                    return False
            
            # 如果本地没有记录，查询数据库
            last_record_time = self.db.get_last_record_time(self.camera_id)
            
            if last_record_time is None:
                # 首次记录
                return True
            
            # 更新本地记录的时间
            self.last_save_time = last_record_time
            
            # 计算时间差
            time_diff = (datetime.now() - last_record_time).total_seconds()
            
            return time_diff >= self.config['min_record_interval']
            
        except Exception as e:
            logger.error(f"检查记录间隔失败 - 摄像头: {self.camera_name}: {e}")
            return True  # 异常时允许记录
    
    def _should_merge_record(self, last_record: Dict[str, Any], current_time: datetime) -> bool:
        """
        判断是否应该合并到上一条记录
        
        Args:
            last_record: 上一条记录（包含id, pssj, jssj等）
            current_time: 当前检测时间
            
        Returns:
            bool: 应该合并返回True，否则返回False
        """
        try:
            # 获取上一条记录的参考时间（优先使用结束时间，如果没有则使用开始时间）
            if last_record.get('jssj'):
                reference_time = last_record['jssj']
            else:
                reference_time = last_record['pssj']
            
            # 处理时间格式（可能是字符串或datetime对象）
            if isinstance(reference_time, str):
                from datetime import datetime
                reference_time = datetime.strptime(reference_time, '%Y-%m-%d %H:%M:%S')
            elif reference_time is None:
                # 如果jssj为None，使用pssj
                reference_time = last_record['pssj']
                if isinstance(reference_time, str):
                    from datetime import datetime
                    reference_time = datetime.strptime(reference_time, '%Y-%m-%d %H:%M:%S')
            
            # 计算时间差（秒）
            time_diff = (current_time - reference_time).total_seconds()
            merge_interval = self.config.get('record_merge_interval', 180)  # 默认3分钟
            
            # 如果时间间隔小于合并间隔，则应该合并
            should_merge = time_diff < merge_interval
            if should_merge:
                logger.debug(f"时间间隔 {time_diff:.1f}秒 < {merge_interval}秒，应该合并记录")
            else:
                logger.debug(f"时间间隔 {time_diff:.1f}秒 >= {merge_interval}秒，创建新记录")
            
            return should_merge
            
        except Exception as e:
            logger.error(f"判断是否合并记录失败 - 摄像头: {self.camera_name}: {e}", exc_info=True)
            return False  # 异常时不合并，创建新记录
    
    def _get_latest_frame_with_time(self) -> Tuple[Optional[any], Optional[datetime]]:
        """
        获取最新的帧及其时间戳用于保存
        清空缓冲区，读取并丢弃旧帧，确保获取到最新的帧

        重要：RTSP流是连续连接的，缓冲区可能积累了很多旧帧
        休眠期间可能积压大量帧（例如5秒×25fps=125帧）

        策略：
        1. 【PyAV有PTS】基于时间清空：持续读取直到帧时间接近当前系统时间（误差<1秒）
        2. 【OpenCV无PTS】基于数量清空：读取足够多的帧（100帧）彻底清空缓冲区

        Returns:
            Tuple[frame, timestamp]: (最新的帧数据, PTS时间戳或系统时间)，失败返回(None, None)
        """
        try:
            import numpy as np

            latest_frame = None
            latest_time = None
            now = datetime.now()

            if self.use_pyav and self.av_decoder is not None:
                # ========== 策略1: PyAV - 基于时间的智能清空 ==========
                logger.debug(f"使用PyAV基于时间清空缓冲区 - 摄像头: {self.camera_name}")

                max_read_count = 200  # 最大读取帧数，避免死循环（25fps×8秒）
                frames_read = 0

                for i in range(max_read_count):
                    ret, frame, frame_time = self._read_frame_with_pts()
                    frames_read += 1

                    if not ret or frame is None:
                        logger.warning(f"读取帧失败，使用已有的最新帧")
                        break

                    # 验证帧有效性
                    if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                        height, width = frame.shape[:2]
                        if height >= 100 and width >= 100:
                            latest_frame = frame.copy()
                            latest_time = frame_time

                            # 如果有准确的PTS时间，检查是否接近当前时间
                            if frame_time:
                                time_diff = (now - frame_time).total_seconds()

                                # 如果帧时间与当前时间差距小于1秒，认为是最新帧
                                if time_diff < 1.0:
                                    logger.info(
                                        f"✓ 获取到实时帧！读取{frames_read}帧，"
                                        f"时间差: {time_diff:.2f}秒，"
                                        f"帧时间: {frame_time.strftime('%H:%M:%S.%f')[:-3]}"
                                    )
                                    break
                                elif i % 20 == 0:  # 每20帧输出一次进度
                                    logger.debug(
                                        f"继续清空缓冲区...已读{frames_read}帧，"
                                        f"当前帧延迟: {time_diff:.2f}秒"
                                    )

                if latest_frame is not None:
                    final_diff = (now - latest_time).total_seconds() if latest_time else 0
                    logger.info(
                        f"✓ PyAV缓冲区清空完成 - 摄像头: {self.camera_name}, "
                        f"共读取{frames_read}帧，最终时间差: {final_diff:.2f}秒"
                    )
                    return latest_frame, latest_time
                else:
                    logger.warning(f"PyAV清空缓冲区失败，未获取到有效帧")
                    return None, None

            else:
                # ========== 策略2: OpenCV - 基于数量的彻底清空 ==========
                logger.debug(f"使用OpenCV基于数量清空缓冲区 - 摄像头: {self.camera_name}")

                # USB摄像头或无PTS的RTSP流，没有准确时间戳
                # 休眠5秒可能积压 5×25=125帧，读取100帧确保清空
                buffer_clear_count = 100  # 大幅增加清空数量

                for i in range(buffer_clear_count):
                    ret, frame, frame_time = self._read_frame_with_pts()
                    if ret and frame is not None:
                        # 验证帧有效性
                        if isinstance(frame, np.ndarray) and len(frame.shape) >= 2:
                            height, width = frame.shape[:2]
                            if height >= 100 and width >= 100:
                                latest_frame = frame.copy()
                                latest_time = frame_time
                            else:
                                if i == buffer_clear_count - 1:
                                    logger.warning(f"帧尺寸过小: {width}x{height}")
                        else:
                            if i == buffer_clear_count - 1:
                                logger.warning(f"帧格式无效")
                    else:
                        # 读取失败，使用已有帧
                        if latest_frame is not None:
                            break
                        if i == 0:
                            time.sleep(0.1)

                    # 每隔20帧输出进度
                    if i > 0 and i % 20 == 0:
                        logger.debug(f"OpenCV缓冲区清空进度: {i}/{buffer_clear_count}帧")

                if latest_frame is not None:
                    logger.info(
                        f"✓ OpenCV缓冲区清空完成 - 摄像头: {self.camera_name}, "
                        f"共读取{buffer_clear_count}帧"
                    )
                    return latest_frame, latest_time
                else:
                    logger.warning(f"OpenCV清空缓冲区失败，未获取到有效帧")
                    return None, None

        except Exception as e:
            logger.error(f"获取最新帧失败: {e}", exc_info=True)
            return None, None

    def _get_latest_frame(self):
        """
        获取最新的帧用于保存（兼容旧接口）

        Returns:
            numpy.ndarray: 最新的帧数据，失败返回None
        """
        frame, _ = self._get_latest_frame_with_time()
        return frame
    
    def _get_clean_frame(self):
        """
        获取一个完整、干净的帧用于保存（已废弃，使用_get_latest_frame代替）
        多次尝试读取，直到获取到有效帧
        
        Returns:
            numpy.ndarray: 有效的帧数据，失败返回None
        """
        # 直接调用_get_latest_frame
        return self._get_latest_frame()
    
    def _save_detection_result(self, frame, detection_time=None):
        """
        保存检测结果（包含YOLO人员检测）
        新逻辑：30秒内的检测合并到同一条轨迹记录

        Args:
            frame: 视频帧
            detection_time: 检测时间
        """
        try:
            from pathlib import Path

            # 1. 保存图片（返回相对路径，如：20251212/29_摄像头名称_20251212144441.jpg）
            image_relative_path = self.storage.save_image(frame, self.camera_id, self.camera_name)

            if image_relative_path is None:
                logger.warning(f"保存图片失败，跳过记录 - 摄像头: {self.camera_name}")
                return

            # 2. 使用YOLO检测图片中的人员数量
            people_count = None
            try:
                # 计算图片的本地绝对路径
                full_image_path = Path(self.storage.image_save_path) / image_relative_path

                # 懒加载 YOLO 检测器（避免每次都重新加载模型）
                if self.person_detector is None:
                    try:
                        # 优先使用自适应检测器（YOLOv11 + CPU优化）
                        if config.ADAPTIVE_DETECTION_CONFIG.get('enabled', False):
                            from core.person_detector_adaptive import AdaptivePersonDetector
                            self.person_detector = AdaptivePersonDetector(
                                model_dir=config.ADAPTIVE_DETECTION_CONFIG['model_dir'],
                                conf_threshold=config.ADAPTIVE_DETECTION_CONFIG['conf_threshold'],
                                iou_threshold=config.ADAPTIVE_DETECTION_CONFIG['iou_threshold'],
                                force_engine=config.ADAPTIVE_DETECTION_CONFIG.get('force_engine'),
                                num_threads=config.ADAPTIVE_DETECTION_CONFIG.get('num_threads')
                            )
                            logger.info(
                                f"自适应人员检测器已初始化（YOLOv11）- 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                                f"引擎: {self.person_detector.engine.value}"
                            )
                        else:
                            # 传统方式（兼容性）
                            from core.detector import PersonDetector
                            self.person_detector = PersonDetector()
                            logger.info(
                                f"YOLO人员检测器已初始化 - 摄像头: {self.camera_name} (ID: {self.camera_id})"
                            )
                    except Exception as det_init_err:
                        logger.error(
                            f"初始化YOLO人员检测器失败，跳过人数检测: {det_init_err}",
                            exc_info=True,
                        )
                        self.person_detector = None

                if self.person_detector is not None:
                    detections = self.person_detector.detect(full_image_path)
                    people_count = len(detections)
                    logger.info(
                        f"YOLO人员检测结果 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                        f"人数: {people_count}, 图片: {full_image_path}"
                    )

                    # 如果没有检测到人，则删除图片并不保存记录
                    if people_count <= 0:
                        logger.info(
                            f"YOLO未检测到人员，本次检测结果忽略，不保存数据库记录 - 摄像头: {self.camera_name}"
                        )
                        try:
                            if full_image_path.exists():
                                full_image_path.unlink()
                                logger.debug(f"已删除无效图片文件: {full_image_path}")
                        except Exception as del_err:
                            logger.warning(f"删除图片文件失败: {full_image_path}, 错误: {del_err}")
                        return
                else:
                    logger.warning(
                        f"YOLO检测器不可用，无法统计人数，将继续保存记录但不写入人数字段 - 摄像头: {self.camera_name}"
                    )
            except Exception as det_err:
                logger.error(
                    f"执行YOLO人员检测时发生错误，将继续保存记录但不写入人数字段: {det_err}",
                    exc_info=True,
                )
                people_count = None

            # 3. 将相对路径转换为完整的URL路径
            image_url = self.storage.get_image_url(image_relative_path)

            # 4. 使用传入的检测时间，如果没有则使用当前时间
            current_time = detection_time if detection_time is not None else datetime.now()

            # 5. 检查是否在30秒窗口内，判断是否应该合并到现有轨迹
            merge_interval = 30  # 30秒合并窗口
            should_merge = False

            if self.current_trajectory is not None:
                # 计算距离上次检测的时间差
                time_diff = (current_time - self.current_trajectory['last_time']).total_seconds()
                if time_diff <= merge_interval:
                    should_merge = True
                    logger.info(
                        f"检测到活动在30秒窗口内({time_diff:.1f}秒)，合并到现有轨迹 - "
                        f"摄像头: {self.camera_name}, 轨迹ID: {self.current_trajectory['record_id']}"
                    )
                else:
                    logger.info(
                        f"距离上次检测已超过30秒({time_diff:.1f}秒)，创建新轨迹 - "
                        f"摄像头: {self.camera_name}"
                    )

            if should_merge:
                # 合并到现有轨迹：更新检测次数、结束时间和人员数量（取最大值），并保存截图到新表
                record_id = self.current_trajectory['record_id']
                new_jscs = self.current_trajectory['jscs'] + 1

                # 更新轨迹记录（包含人员数量，数据库会自动取最大值）
                success = self.db.update_trajectory_detection(
                    record_id=record_id,
                    jscs=new_jscs,
                    jssj=current_time,
                    rysl=people_count
                )

                if success:
                    # 保存截图到新表
                    screenshot_id = self.db.save_trajectory_screenshot(
                        hdgj_id=record_id,
                        screenshot_url=image_url,
                        screenshot_time=current_time,
                        screenshot_order=new_jscs
                    )

                    if screenshot_id:
                        # 更新缓存中的最大人员数量
                        if people_count is not None:
                            current_max = self.current_trajectory.get('max_rysl', 0) or 0
                            self.current_trajectory['max_rysl'] = max(current_max, people_count)

                        logger.info(
                            f"✓ 已合并到轨迹 - 摄像头: {self.camera_name}, 轨迹ID: {record_id}, "
                            f"检测次数: {new_jscs}, 当前人数: {people_count}, 最大人数: {self.current_trajectory.get('max_rysl')}, "
                            f"截图ID: {screenshot_id}"
                        )

                        # 更新缓存
                        self.current_trajectory['jscs'] = new_jscs
                        self.current_trajectory['last_time'] = current_time

                        # 异步调用ReID识别人员姓名
                        self._async_identify_person(frame, record_id)
                    else:
                        logger.warning(f"保存轨迹截图失败 - 轨迹ID: {record_id}")
                else:
                    logger.warning(f"更新轨迹检测次数失败 - 轨迹ID: {record_id}")
            else:
                # 创建新轨迹记录
                record_id = self.db.save_detection_record(
                    pssj=current_time,
                    pstp=image_url,
                    qyid=self.camera_id,
                    qymc=self.area_name,  # 使用 gnslx 作为区域名称
                    sxtmx=self.camera_name,  # 使用 fjmc 作为摄像头名称
                    rysl=people_count if people_count is not None else None,
                )

                if record_id:
                    logger.info(
                        f"✓ 创建新轨迹记录 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                        f"图片URL: {image_url}, 记录ID: {record_id}, 检测次数: 1"
                    )

                    # 同时保存第一张截图到新表
                    screenshot_id = self.db.save_trajectory_screenshot(
                        hdgj_id=record_id,
                        screenshot_url=image_url,
                        screenshot_time=current_time,
                        screenshot_order=1
                    )

                    # 更新当前轨迹缓存
                    self.current_trajectory = {
                        'record_id': record_id,
                        'jscs': 1,
                        'last_time': current_time,
                        'max_rysl': people_count if people_count is not None else 0
                    }

                    # 异步调用ReID识别人员姓名
                    self._async_identify_person(frame, record_id)
                else:
                    logger.warning(
                        f"保存记录失败 - 摄像头: {self.camera_name} (ID: {self.camera_id})"
                    )

        except Exception as e:
            logger.error(
                f"保存检测结果失败 - 摄像头: {self.camera_name} (ID: {self.camera_id}): {e}"
            )

    def _count_people_in_frame(self, frame) -> Optional[int]:
        """
        使用YOLO检测当前帧中的人员数量（不做保存/数据库操作）
        
        Returns:
            int: 检测到的人员数量；检测失败返回None
        """
        try:
            # 懒加载 YOLO 检测器（与 _save_detection_result 复用同一个实例）
            if self.person_detector is None:
                try:
                    # 优先使用自适应检测器（YOLOv11 + CPU优化）
                    if config.ADAPTIVE_DETECTION_CONFIG.get('enabled', False):
                        from core.person_detector_adaptive import AdaptivePersonDetector
                        self.person_detector = AdaptivePersonDetector(
                            model_dir=config.ADAPTIVE_DETECTION_CONFIG['model_dir'],
                            conf_threshold=config.ADAPTIVE_DETECTION_CONFIG['conf_threshold'],
                            iou_threshold=config.ADAPTIVE_DETECTION_CONFIG['iou_threshold'],
                            force_engine=config.ADAPTIVE_DETECTION_CONFIG.get('force_engine'),
                            num_threads=config.ADAPTIVE_DETECTION_CONFIG.get('num_threads')
                        )
                        logger.info(
                            f"自适应人员检测器已初始化(用于合并前校验)（YOLOv11）- 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                            f"引擎: {self.person_detector.engine.value}"
                        )
                    else:
                        # 传统方式（兼容性）
                        from core.detector import PersonDetector
                        self.person_detector = PersonDetector()
                        logger.info(
                            f"YOLO人员检测器已初始化(用于合并前校验) - 摄像头: {self.camera_name} (ID: {self.camera_id})"
                        )
                except Exception as det_init_err:
                    logger.error(
                        f"初始化YOLO人员检测器失败（合并前校验），跳过人数检测: {det_init_err}",
                        exc_info=True,
                    )
                    self.person_detector = None
                    return None

            if self.person_detector is None:
                return None

            # 根据检测器类型选择推理方式
            if hasattr(self.person_detector, 'detect_image'):
                # 自适应检测器（YOLOv11 + ONNX/OpenVINO）
                detections = self.person_detector.detect_image(frame)
                count = len(detections)
            else:
                # 传统 PyTorch YOLO 检测器
                results = self.person_detector.model.predict(
                    frame,
                    conf=self.person_detector.conf_threshold,
                    iou=self.person_detector.iou_threshold,
                    classes=[0],
                    device=self.person_detector.device,
                    verbose=False,
                )
                if not results:
                    return 0
                result = results[0]
                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0

            logger.info(
                f"YOLO合并前校验结果 - 摄像头: {self.camera_name} (ID: {self.camera_id}), 人数: {count}"
            )
            return count
        except Exception as e:
            logger.error(f"YOLO合并前人数校验失败: {e}", exc_info=True)
            return None

    def _async_identify_person(self, frame, record_id: int):
        """
        异步识别人员姓名
        
        Args:
            frame: 视频帧
            record_id: 数据库记录ID
        """
        def identify():
            try:
                from reid_integration import ReIDIntegration
                reid = ReIDIntegration()
                
                if not reid.enabled:
                    logger.debug("ReID系统未启用，跳过人员识别")
                    return
                
                # 识别人员姓名
                person_name = reid.identify_person_from_frame(frame)
                
                if person_name:
                    # 更新数据库记录，添加人员姓名到ryxm字段
                    success = self.db.update_person_name(record_id, person_name)
                    if success:
                        logger.info(
                            f"✓ 识别到人员: {person_name} "
                            f"(摄像头: {self.camera_name}, 记录ID: {record_id})"
                        )
                    else:
                        logger.warning(f"更新人员姓名失败 - 记录ID: {record_id}")
                else:
                    logger.debug(f"未识别到人员 (摄像头: {self.camera_name}, 记录ID: {record_id})")
                    
            except ImportError:
                logger.debug("ReID集成模块未找到，跳过人员识别")
            except Exception as e:
                logger.error(f"ReID识别失败 (记录ID: {record_id}): {e}", exc_info=True)
        
        # 启动异步线程
        thread = threading.Thread(target=identify, daemon=True)
        thread.start()
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.running

