"""
单摄像头监测模块
每个摄像头一个独立的监测任务
使用背景建模法进行检测
"""
import cv2
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import config_loader as config
from database import Database
from image_detection import ImageChangeDetection
from image_storage import ImageStorage
from video_stream_reader import VideoStreamReader  # 独立线程读取器
from input_shaper import InputShaper  # 输入整形层
import av  # PyAV for RTSP with PTS support


class CameraMonitor:
    """单个摄像头监测类"""
    
    def __init__(self, camera_info: dict, db: Database, yolo_pool=None):
        """
        初始化摄像头监测
        """
        self.camera_info = camera_info
        self.camera_id = camera_info['id']
        self.camera_name = camera_info['fjmc']  # 摄像头名称
        self.area_name = camera_info.get('gnslx', '')  # 区域名称
        self.rtsp_url = camera_info['rtspssl']

        # 检查是否为USB摄像头
        self.is_usb_camera = self.rtsp_url.startswith('usb:')

        # 记录完整的URL用于调试
        if self.is_usb_camera:
            usb_device_id = self.rtsp_url.split(':')[1] if ':' in self.rtsp_url else '0'
            logger.info(f"摄像头 {self.camera_name} (ID: {self.camera_id}) USB设备ID: {usb_device_id}")
        else:
            logger.info(f"摄像头 {self.camera_name} (ID: {self.camera_id}) RTSP URL: {self.rtsp_url}")
            if not self.rtsp_url or len(self.rtsp_url.strip()) == 0:
                logger.error(f"摄像头 {self.camera_name} RTSP URL为空！")

        self.db = db
        self.detection = ImageChangeDetection()
        self.storage = ImageStorage()

        # YOLO检测方式
        self.yolo_pool = yolo_pool
        self.person_detector = None

        if self.yolo_pool:
            logger.info(f"摄像头 {self.camera_name} 将使用共享YOLO实例池")
        else:
            logger.info(f"摄像头 {self.camera_name} 将创建独立YOLO实例")
        
        self.config = config.RTSP_MONITOR_CONFIG
        self.running = False
        self.monitor_thread = None

        # 记录上次保存的时间
        self.last_save_time = None

        # 当前活动轨迹缓存
        self.current_trajectory = None

        # 视频捕获相关
        self.cap = None
        self.stream_reader = None  # 独立线程读取器
        self.input_shaper = None   # 输入整形层

        # PyAV 相关
        self.av_container = None
        self.av_stream = None
        self.av_decoder = None
        self.time_base = None
        self.pts_base_time = None
        self.pts_base_offset = None
        self.use_pyav = False

        # 时间校准
        self.time_offset_seconds = None
        self.time_calibration_method = None

        # 控制相关
        self.last_process_time = 0
        self.detection_interval = self.config.get('detection_wait_interval', 5)

        self.detection_mode = 'fast'
        self.no_person_count = 0
        self.slow_mode_interval = 10
        self.no_person_threshold = 30

        # 异步任务队列 (frame, time, detections)
        self.task_queue = queue.Queue(maxsize=100)
        self.worker_thread = None

        self.reconnect_attempts = 0
    
    def start(self):
        """启动监测"""
        if self.running:
            logger.warning(f"摄像头 {self.camera_name} 已在监测中")
            return

        self.running = True

        # 启动后台工作线程
        self.worker_thread = threading.Thread(
            target=self._worker_process,
            name=f"CameraWorker-{self.camera_id}",
            daemon=True
        )
        self.worker_thread.start()
        logger.info(f"后台工作线程已启动: {self.camera_name} (ID: {self.camera_id})")

        # 启动监测主线程
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

    def _worker_process(self):
        """
        后台工作线程，专门处理耗时任务（文件保存、数据库写入）
        """
        logger.info(f"后台工作线程开始运行 - 摄像头: {self.camera_name}")

        while self.running:
            try:
                # 从队列获取任务
                try:
                    # 尝试解包3个参数 (兼容旧代码)
                    task_data = self.task_queue.get(timeout=1)
                    if len(task_data) == 3:
                        frame, capture_time, detections = task_data
                    else:
                        frame, capture_time = task_data
                        detections = None
                except queue.Empty:
                    continue

                # 执行保存逻辑
                self._save_detection_result(frame, capture_time, detections=detections)

                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"后台工作线程异常 - 摄像头: {self.camera_name}: {e}", exc_info=True)

        logger.info(f"后台工作线程已停止 - 摄像头: {self.camera_name}")

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
                
                self.reconnect_attempts = 0
                
                # 初始化阶段...
                logger.info(f"摄像头 {self.camera_name} 正在初始化，快速跳过前3帧...")
                for _ in range(3):
                    ret, frame, frame_time = self._read_frame_with_pts()
                    time.sleep(0.001)
                
                # 背景建模初始化
                if self.detection.bg_initialized and self.detection.bg_subtractor is not None:
                    logger.info(f"摄像头 {self.camera_name} 重新连接，快速适应中（10秒）...")
                    learning_time = 10
                else:
                    logger.info(f"摄像头 {self.camera_name} 背景建模学习中，请等待30秒...")
                    learning_time = 30
                
                if not self.detection.bg_initialized:
                    # 简化的初始化逻辑
                    history = self.config.get('bg_history', 500)
                    var_threshold = self.config.get('bg_var_threshold', 16)
                    self.detection.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=history, varThreshold=var_threshold, detectShadows=True
                    )
                    self.detection.bg_initialized = True
                
                # 学习阶段（使用快速读取，不受节拍限制）
                learning_end_time = time.time() + learning_time
                while time.time() < learning_end_time and self.running:
                    ret, frame, frame_time = self._read_frame_for_learning()
                    if ret and frame is not None:
                        self.detection.bg_subtractor.apply(frame, learningRate=0.1)
                    time.sleep(0.001)
                
                logger.info(f"摄像头 {self.camera_name} 开始正常监测...")

                last_detection_time = 0
                use_stream_reader = self.config.get('use_video_stream_reader', False)
                last_buffer_flush_time = time.time()
                buffer_flush_interval = self.config.get('buffer_flush_interval', 30)

                # 监测循环
                while self.running:
                    try:
                        # 1. 定期清理（仅非VideoStreamReader/InputShaper模式）
                        if not use_stream_reader and self.stream_reader is None and self.input_shaper is None:
                            current_time = time.time()
                            if current_time - last_buffer_flush_time >= buffer_flush_interval:
                                self._flush_buffer_smart(max_frames=100, reason="定期清理")
                                last_buffer_flush_time = current_time

                        # 2. 读取最新帧
                        ret, frame, frame_time = self._read_frame_with_pts()

                        if not ret or frame is None:
                            logger.warning(f"摄像头 {self.camera_name} 获取帧失败，尝试重连")
                            break

                        if frame.size == 0: continue

                        # 3. 频率控制
                        current_time = time.time()
                        if self.detection_mode == 'slow':
                            if current_time - last_detection_time < self.slow_mode_interval:
                                time.sleep(0.001); continue
                        else:
                            if current_time - last_detection_time < 1.0:
                                time.sleep(0.001); continue

                        last_detection_time = current_time

                        # 4. 背景检测
                        has_change = self.detection.detect_change(frame)
                        
                        if has_change:
                            # 简单的坏帧过滤
                            try:
                                import numpy as np
                                if np.mean(frame) < 10 or np.mean(frame) > 245: continue
                            except: pass

                            logger.info(f"摄像头 {self.camera_name} 检测到变化...")

                            # 5. 【核心优化】使用 YOLO 检测列表
                            detections = self._detect_people_in_frame(frame)
                            people_count = len(detections) if detections is not None else 0

                            # 清理缓冲区（YOLO耗时后）
                            if not use_stream_reader and self.stream_reader is None and self.input_shaper is None:
                                self._flush_buffer_after_yolo(max_frames=50)

                            # 6. 有效性判断
                            if detections is None or people_count <= 0:
                                if self.detection_mode == 'slow':
                                    self.no_person_count += 1
                                    if self.no_person_count >= self.no_person_threshold:
                                        self.detection_mode = 'fast'
                                        self.no_person_count = 0
                                continue

                            # 7. 冷却检查
                            current_time = time.time()
                            if current_time - self.last_process_time < self.detection_interval:
                                continue

                            self.last_process_time = current_time

                            # 8. 【核心优化】推送到后台队列 (带检测结果)
                            logger.info(f"检测到 {people_count} 人，加入队列...")
                            if not self.task_queue.full():
                                self.task_queue.put((frame.copy(), frame_time, detections))
                            else:
                                logger.warning(f"队列已满，丢帧 - {self.camera_name}")

                            # 切换模式
                            if self.detection_mode == 'fast':
                                self.detection_mode = 'slow'
                            self.no_person_count = 0

                        time.sleep(0.005)

                    except Exception as e:
                        logger.error(f"监测循环异常: {e}")
                        time.sleep(1)
                
                self._release_capture()
                
            except Exception as e:
                logger.error(f"监测异常: {e}")
                self.reconnect_attempts += 1
                time.sleep(self.config['reconnect_interval'])
        
        self.running = False

    # =========================================================================
    # 以下为被恢复的连接和辅助方法
    # =========================================================================

    def _generate_alternative_urls(self, base_url: str) -> list:
        """生成多种RTSP URL格式"""
        import re
        from urllib.parse import urlparse, unquote
        try:
            decoded_url = unquote(base_url)
            if decoded_url != base_url: base_url = decoded_url
        except: pass
        
        urls = [base_url]
        # 简单的尝试机制
        if 'channel' in base_url and 'subtype' not in base_url:
             urls.append(f"{base_url}&subtype=0")
        return urls

    def _prepare_rtsp_url(self, url: str, transport: str = 'tcp') -> str:
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}rtsp_transport={transport}"

    def _connect_usb_camera(self) -> bool:
        """连接USB摄像头"""
        try:
            usb_device_id = int(self.rtsp_url.split(':')[1]) if ':' in self.rtsp_url else 0
            logger.info(f"正在连接USB摄像头: {self.camera_name} (ID: {usb_device_id})")

            self.cap = cv2.VideoCapture(usb_device_id)
            time.sleep(1.0)

            if not self.cap.isOpened():
                logger.error(f"USB摄像头打开失败")
                return False

            # 设置参数
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except: pass

            # 测试读取
            success = False
            for i in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    success = True
                    break
                time.sleep(0.1)

            if success:
                logger.info(f"USB摄像头连接成功")
                # 优先检查是否启用输入整形层
                input_shaper_config = config.INPUT_SHAPER_CONFIG
                if input_shaper_config.get('enabled', False):
                    logger.info(f"启用 InputShaper 输入整形层")
                    self.input_shaper = InputShaper(
                        self.cap,
                        camera_name=self.camera_name,
                        target_fps=input_shaper_config.get('target_fps', 2.0),
                        buffer_size=input_shaper_config.get('buffer_size', 30),
                        drop_strategy=input_shaper_config.get('drop_strategy', 'drop_old'),
                        empty_behavior=input_shaper_config.get('empty_behavior', 'skip')
                    )
                else:
                    # 回退到原来的 VideoStreamReader
                    use_stream_reader = self.config.get('use_video_stream_reader', False)
                    if use_stream_reader:
                        logger.info(f"启用 VideoStreamReader 独立线程")
                        self.stream_reader = VideoStreamReader(
                            self.cap,
                            camera_name=self.camera_name,
                            use_pyav=False
                        )
                return True
            else:
                logger.error(f"USB摄像头读取测试帧失败")
                self._release_capture()
                return False

        except Exception as e:
            logger.error(f"连接USB摄像头异常: {e}")
            return False

    def _connect_rtsp(self) -> bool:
        """连接RTSP流"""
        try:
            self._release_capture()

            if self.is_usb_camera:
                return self._connect_usb_camera()
            
            # 使用OpenCV连接
            logger.info(f"正在连接RTSP流: {self.rtsp_url}")
            # 强制 TCP
            rtsp_url = self._prepare_rtsp_url(self.rtsp_url, 'tcp')
            
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG if hasattr(cv2, 'CAP_FFMPEG') else 1900)
            
            # 设置缓冲区
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # 尽可能小
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            except: pass

            time.sleep(1.0)

            if self.cap.isOpened():
                # 测试读取
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"RTSP连接成功")
                    
                    # 检查是否需要 PyAV (timestamp_strategy)
                    strategy = self.config.get('timestamp_strategy', 'realtime')
                    if strategy in ['pts_auto', 'pts_fixed']:
                        if self._init_pyav_stream():
                            self.use_pyav = True
                            if self.cap: self.cap.release(); self.cap = None
                            return True
                    
                    self.use_pyav = False

                    # 优先检查是否启用输入整形层
                    input_shaper_config = config.INPUT_SHAPER_CONFIG
                    if input_shaper_config.get('enabled', False):
                        logger.info(f"启用 InputShaper 输入整形层")
                        self.input_shaper = InputShaper(
                            self.cap,
                            camera_name=self.camera_name,
                            target_fps=input_shaper_config.get('target_fps', 2.0),
                            buffer_size=input_shaper_config.get('buffer_size', 30),
                            drop_strategy=input_shaper_config.get('drop_strategy', 'drop_old'),
                            empty_behavior=input_shaper_config.get('empty_behavior', 'skip')
                        )
                    else:
                        # 回退到原来的 VideoStreamReader
                        use_stream_reader = self.config.get('use_video_stream_reader', False)
                        if use_stream_reader:
                            self.stream_reader = VideoStreamReader(
                                self.cap, camera_name=self.camera_name, use_pyav=False
                            )
                    return True
            
            logger.error(f"RTSP连接失败")
            return False

        except Exception as e:
            logger.error(f"连接RTSP异常: {e}")
            return False

    def _release_capture(self):
        if self.input_shaper: self.input_shaper.release(); self.input_shaper = None
        if self.stream_reader: self.stream_reader.release(); self.stream_reader = None
        if self.cap: self.cap.release(); self.cap = None
        self._release_pyav()

    def _release_pyav(self):
        if self.av_container: 
            try: self.av_container.close() 
            except: pass
            self.av_container = None
        self.av_decoder = None

    def _init_pyav_stream(self) -> bool:
        """初始化 PyAV"""
        try:
            self._release_pyav()
            self.av_container = av.open(
                self.rtsp_url,
                options={'rtsp_transport': 'tcp', 'buffer_size': '20480000'}, # 大缓冲抗抖动
                timeout=20.0
            )
            if len(self.av_container.streams.video) == 0: return False
            self.av_stream = self.av_container.streams.video[0]
            self.time_base = self.av_stream.time_base
            self.av_decoder = self.av_container.decode(self.av_stream)
            
            # 读取第一帧建立基准
            for frame in self.av_decoder:
                if frame.pts is not None:
                    self.pts_base_offset = float(frame.pts * self.time_base)
                    self.pts_base_time = datetime.now()
                    self.av_decoder = self.av_container.decode(self.av_stream) # 重置
                    return True
                break
            return False
        except Exception as e:
            logger.error(f"PyAV初始化失败: {e}")
            return False

    def _pts_to_datetime(self, pts) -> datetime:
        if pts is None or self.pts_base_time is None: return datetime.now()
        pts_sec = float(pts * self.time_base)
        return self.pts_base_time + timedelta(seconds=(pts_sec - self.pts_base_offset))

    def _read_frame_for_learning(self) -> Tuple[bool, Optional[any], Optional[datetime]]:
        """用于背景学习阶段的快速帧读取（不受节拍限制）"""
        if self.input_shaper:
            return self.input_shaper.read_nowait()
        return self._read_frame_with_pts()

    def _read_frame_with_pts(self) -> Tuple[bool, Optional[any], Optional[datetime]]:
        # 优先使用输入整形层
        if self.input_shaper:
            return self.input_shaper.read()

        if self.use_pyav and self.av_decoder:
            try:
                for frame in self.av_decoder:
                    img = frame.to_ndarray(format='bgr24')
                    return True, img, self._pts_to_datetime(frame.pts)
            except:
                self.use_pyav = False # 降级
                return False, None, None

        if self.stream_reader:
            return self.stream_reader.read()

        if self.cap:
            ret, frame = self.cap.read()
            return ret, frame, datetime.now()

        return False, None, None

    def _flush_buffer_smart(self, max_frames=100, reason=""):
        if self.use_pyav or self.stream_reader or self.input_shaper: return 0
        count = 0
        if self.cap:
            for _ in range(max_frames):
                if not self.cap.grab(): break
                count += 1
        if count > 0: logger.debug(f"[{self.camera_name}] {reason} 丢弃 {count} 帧")
        return count

    def _flush_buffer_after_yolo(self, max_frames=50):
        self._flush_buffer_smart(max_frames, "YOLO后清理")

    # =========================================================================
    # 核心业务逻辑
    # =========================================================================

    def _save_detection_result(self, frame, detection_time=None, detections=None):
        """
        保存检测结果
        """
        try:
            current_time = detection_time if detection_time is not None else datetime.now()
            
            # 1. 获取人数
            people_count = 0
            if detections is not None:
                people_count = len(detections)
            else:
                # 回退：如果没有传入，则现场检测
                new_detections = self._detect_people_in_frame(frame)
                if new_detections: people_count = len(new_detections)

            if people_count <= 0: return

            logger.info(f"正在保存证据 ({people_count}人)...")
            
            # 2. 保存图片
            image_relative_path = self.storage.save_image(
                frame, self.camera_id, self.camera_name, capture_time=current_time
            )
            if not image_relative_path: return
            image_url = self.storage.get_image_url(image_relative_path)

            # 3. 轨迹合并
            merge_interval = self.config.get('trajectory_merge_interval', 30)
            should_merge = False

            if self.current_trajectory:
                time_diff = (current_time - self.current_trajectory['last_time']).total_seconds()
                if time_diff <= merge_interval: should_merge = True
            
            if should_merge:
                record_id = self.current_trajectory['record_id']
                new_jscs = self.current_trajectory['jscs'] + 1
                
                success = self.db.update_trajectory_detection(
                    record_id=record_id, jscs=new_jscs, jssj=current_time, rysl=people_count
                )
                
                if success:
                    self.db.save_trajectory_screenshot(
                        track_id=record_id, screenshot_url=image_url, 
                        screenshot_time=current_time, screenshot_order=new_jscs
                    )
                    # 更新缓存
                    self.current_trajectory['max_rysl'] = max(self.current_trajectory.get('max_rysl', 0), people_count)
                    self.current_trajectory['jscs'] = new_jscs
                    self.current_trajectory['last_time'] = current_time
                    logger.info(f"✓ 合并轨迹 - ID: {record_id}, 次数: {new_jscs}")
            else:
                # 新轨迹
                record_id = self.db.save_detection_record(
                    pssj=current_time, pstp=image_url, qyid=self.camera_id, 
                    qymc=self.area_name, sxtmx=self.camera_name, rysl=people_count
                )
                if record_id:
                    self.db.save_trajectory_screenshot(
                        track_id=record_id, screenshot_url=image_url, 
                        screenshot_time=current_time, screenshot_order=1
                    )
                    self.current_trajectory = {
                        'record_id': record_id, 'jscs': 1, 
                        'last_time': current_time, 'max_rysl': people_count
                    }
                    logger.info(f"✓ 新轨迹 - ID: {record_id}")

        except Exception as e:
            logger.error(f"保存失败: {e}")

    def _detect_people_in_frame(self, frame) -> Optional[List[Dict]]:
        """
        检测人员，返回列表
        """
        try:
            # 优先使用进程池
            if self.yolo_pool is not None:
                try:
                    # 【修正】直接使用 config.YOLO_POOL_CONFIG 读取，确保读取到 30秒
                    timeout = config.YOLO_POOL_CONFIG.get('timeout', 30)
                    return self.yolo_pool.detect(frame, timeout=timeout)
                except Exception as pool_err:
                    logger.error(f"YOLO池检测异常: {pool_err}")
                    return None

            # 回退本地检测逻辑...
            if self.person_detector is None:
                try:
                    from core.detector import PersonDetector
                    self.person_detector = PersonDetector()
                except: return None

            if hasattr(self.person_detector, 'detect_image'):
                return self.person_detector.detect_image(frame)
            
            # 兼容旧predict
            results = self.person_detector.model.predict(
                frame, conf=0.45, classes=[0], verbose=False
            )
            return [{} for _ in results[0].boxes] if results else []

        except Exception as e:
            logger.error(f"本地检测失败: {e}")
            return None

    def is_running(self) -> bool:
        return self.running