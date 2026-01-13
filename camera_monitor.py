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
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import config_loader as config
from database import Database
from image_detection import ImageChangeDetection
from image_storage import ImageStorage
from video_stream_reader import VideoStreamReader  # 独立线程读取器
import av  # PyAV for RTSP with PTS support


class CameraMonitor:
    """单个摄像头监测类"""
    
    def __init__(self, camera_info: dict, db: Database, yolo_pool=None):
        """
        初始化摄像头监测

        Args:
            camera_info: 摄像头配置信息
            db: 数据库连接对象
            yolo_pool: YOLO实例池（可选）。如果提供，则使用共享池；否则创建独立实例
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

        # YOLO检测方式：使用共享池或独立实例
        self.yolo_pool = yolo_pool  # 共享YOLO实例池（如果启用）
        self.person_detector = None  # 独立YOLO实例（如果未使用池）

        if self.yolo_pool:
            logger.info(f"摄像头 {self.camera_name} 将使用共享YOLO实例池")
        else:
            logger.info(f"摄像头 {self.camera_name} 将创建独立YOLO实例")
        
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

        # VideoStreamReader (独立线程读取器，解决缓冲区积压问题)
        self.stream_reader = None

        # PyAV container and stream (for RTSP with PTS)
        self.av_container = None
        self.av_stream = None
        self.av_decoder = None
        self.time_base = None  # 视频流的时间基准
        self.pts_base_time = None  # PTS到真实时间的映射基准 (datetime)
        self.pts_base_offset = None  # PTS基准偏移量 (秒)
        self.use_pyav = False  # 是否使用PyAV（RTSP流使用，USB摄像头不使用）

        # ==================== 时间校准相关（已简化为使用系统时间）====================
        self.time_offset_seconds = None  # 时间偏移量（秒），已弃用
        self.time_calibration_method = None  # 实际使用的校准方法，已弃用

        # 时间间隔控制（避免暴力sleep）
        self.last_process_time = 0  # 上次处理时间（时间戳）
        self.detection_interval = self.config.get('detection_wait_interval', 5)  # 检测间隔（秒）

        # 自适应检测模式（智能频率控制）
        self.detection_mode = 'fast'  # 检测模式：'fast'(快速/持续检测) 或 'slow'(慢速/10秒一次)
        self.no_person_count = 0  # 慢速模式下连续未检测到人的次数
        self.slow_mode_interval = 10  # 慢速模式的检测间隔（秒）
        self.no_person_threshold = 30  # 连续N次未检测到人后切回快速模式

        # 异步任务队列（避免I/O阻塞视频读取）
        self.task_queue = queue.Queue(maxsize=100)  # 限制队列大小，防止内存溢出
        self.worker_thread = None  # 后台工作线程（在start时启动）

        self.reconnect_attempts = 0
    
    def start(self):
        """启动监测"""
        if self.running:
            logger.warning(f"摄像头 {self.camera_name} 已在监测中")
            return

        self.running = True

        # 启动后台工作线程（处理耗时的I/O操作）
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
        后台工作线程，专门处理耗时任务（文件保存、YOLO检测、数据库写入）
        这样主线程可以持续读取视频帧，避免缓冲区积压
        """
        logger.info(f"后台工作线程开始运行 - 摄像头: {self.camera_name}")

        while self.running:
            try:
                # 从队列获取任务，超时1秒（避免线程卡死）
                try:
                    frame, capture_time = self.task_queue.get(timeout=1)
                except queue.Empty:
                    # 队列为空，继续等待
                    continue

                # 执行原有的耗时处理逻辑
                self._save_detection_result(frame, capture_time)

                # 标记任务完成
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
                
                # 连接成功，重置重连计数
                self.reconnect_attempts = 0
                
                # 跳过前几帧，让摄像头稳定（持续读取避免缓冲区积压）
                logger.info(f"摄像头 {self.camera_name} 正在初始化，快速跳过前3帧...")
                for _ in range(3):
                    ret, frame, frame_time = self._read_frame_with_pts()
                    if ret and frame is not None:
                        pass  # 背景建模不需要保存前一帧
                    time.sleep(0.001)  # 极小延迟，避免缓冲区积压
                
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

                # 改为基于时间控制，持续读取避免缓冲区积压
                learning_start_time = time.time()
                learning_end_time = learning_start_time + learning_time
                frame_count_in_learning = 0
                last_progress_log_time = learning_start_time

                logger.info(f"开始背景建模学习阶段，目标时长: {learning_time}秒，持续读取避免缓冲区积压")

                while time.time() < learning_end_time and self.running:
                    ret, frame, frame_time = self._read_frame_with_pts()
                    if ret and frame is not None:
                        try:
                            # 使用高学习率学习背景（学习阶段不进行形态学和过滤）
                            fg_mask = self.detection.bg_subtractor.apply(frame, learningRate=learning_phase_rate)
                            frame_count_in_learning += 1

                            # 每5秒输出一次学习进度
                            current_time = time.time()
                            if current_time - last_progress_log_time >= 5.0:
                                elapsed = current_time - learning_start_time
                                import numpy as np
                                fg_pixels = np.count_nonzero(fg_mask)
                                total_pixels = fg_mask.size
                                fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0
                                logger.info(
                                    f"背景建模学习进度: {elapsed:.1f}/{learning_time}秒, "
                                    f"已处理 {frame_count_in_learning} 帧, "
                                    f"当前前景比例: {fg_ratio*100:.3f}%"
                                )

                                # 如果前景比例很高，说明学习阶段画面中有运动物体
                                if fg_ratio > 0.1:  # 超过10%
                                    logger.warning(f"⚠️  学习阶段检测到大量前景 ({fg_ratio*100:.1f}%)，可能影响背景模型质量！")

                                last_progress_log_time = current_time
                        except Exception as e:
                            logger.error(f"背景建模学习异常: {e}")

                    # 极小延迟，避免CPU空转，但不阻塞流读取
                    time.sleep(0.001)
                
                logger.info(f"摄像头 {self.camera_name} 背景建模学习完成，开始正常监测...")

                frame_count = 0
                last_detection_time = 0  # 上次背景检测的时间（控制检测频率）

                # 开始监测循环
                # 缓冲区清理策略：根据配置决定是否启用
                use_stream_reader = self.config.get('use_video_stream_reader', False)
                last_buffer_flush_time = time.time()
                buffer_flush_interval = self.config.get('buffer_flush_interval', 30)
                buffer_flush_max_frames = self.config.get('buffer_flush_max_frames', 100)

                # 只在不使用VideoStreamReader时才显示清理配置
                if not use_stream_reader and self.stream_reader is None:
                    logger.info(
                        f"\n{'='*60}\n"
                        f"[{self.camera_name}] 缓冲区管理已启用（定期清理策略）\n"
                        f"  - 定期清理间隔: {buffer_flush_interval}秒\n"
                        f"  - 安全上限: {buffer_flush_max_frames}帧\n"
                        f"  - YOLO后清理上限: {self.config.get('buffer_flush_after_yolo_frames', 50)}帧\n"
                        f"  - 预计每{buffer_flush_interval}秒会看到一次清理日志\n"
                        f"{'='*60}"
                    )
                else:
                    logger.info(
                        f"\n{'='*60}\n"
                        f"[{self.camera_name}] 使用VideoStreamReader独立线程模式\n"
                        f"  - 独立线程持续读取最新帧\n"
                        f"  - 无需定期清理缓冲区\n"
                        f"  - 实时性最高（<1ms延迟）\n"
                        f"{'='*60}"
                    )

                while self.running:
                    try:
                        # === 自适应缓冲区清理策略 ===
                        # 仅在不使用VideoStreamReader时才执行定期清理
                        if not use_stream_reader and self.stream_reader is None:
                            current_time = time.time()
                            if current_time - last_buffer_flush_time >= buffer_flush_interval:
                                flushed = self._flush_buffer_smart(
                                    max_frames=buffer_flush_max_frames,
                                    reason="定期清理"
                                )
                                last_buffer_flush_time = current_time

                        # 持续读取帧，绝不阻塞！
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
                        current_time = time.time()

                        # 自适应检测频率控制（智能模式切换）
                        # 快速模式：每1秒检测（适合无人/有变化但不是人的场景）
                        # 慢速模式：每10秒检测（节省CPU，适合确认有人的场景）

                        if self.detection_mode == 'slow':
                            # 慢速模式：10秒检测一次
                            if current_time - last_detection_time < self.slow_mode_interval:
                                time.sleep(0.001)  # 极短休眠让出CPU
                                continue
                        else:
                            # 快速模式：每1秒检测一次（平衡性能和实时性）
                            if current_time - last_detection_time < 1.0:
                                time.sleep(0.001)
                                continue

                        last_detection_time = current_time

                        # 执行背景检测
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
                            # ==================== 1. 新增：坏帧/花屏过滤器 ====================
                            # POC报错会导致解码出全黑、全绿或全灰的纯色图片，这些会被误判为"剧烈运动"
                            # 正常摄像头的画面是有纹理的，标准差(std)通常较高
                            # 坏帧通常是纯色，标准差极低 (<10)
                                try:
                                    import numpy as np
                                    if frame is None or frame.size == 0:
                                        continue
                                    
                                    # 计算均值和标准差
                                    frame_mean = np.mean(frame)
                                    frame_std = np.std(frame)
                                    
                                    # 规则1: 极暗(黑屏)或极亮(白屏)
                                    if frame_mean < 10 or frame_mean > 245:
                                        logger.warning(f"检测到异常坏帧(纯色)，均值:{frame_mean:.1f}，跳过 - {self.camera_name}")
                                        continue
                                        
                                    # 规则2: 画面太平坦(全绿/全灰花屏)，正常画面std至少>20
                                    if frame_std < 10:
                                        logger.warning(f"检测到异常坏帧(无纹理)，标准差:{frame_std:.1f}，跳过 - {self.camera_name}")
                                        continue
                                except Exception as e:
                                    logger.error(f"坏帧检测出错: {e}")
                                # ================================================================

                                logger.info(
                                    f"✓ [{self.camera_name}] 检测到画面变化！"
                                    f" 前景比例: {change_percent:.3f}% (阈值: {threshold_percent:.3f}%, "
                                    f"连续帧: {consecutive_count}/{consecutive_threshold})"
                                )

                                # 检测到变化，直接使用当前实时帧
                                logger.info(f"摄像头 {self.camera_name} 检测到变化，正在处理...")

                                # 先用YOLO核实当前帧中是否有人（主线程执行，会阻塞200-500ms）
                                people_count_for_merge = self._count_people_in_frame(frame)

                                # 🚀 关键优化：YOLO检测完成后立即清空缓冲区
                                if not use_stream_reader and self.stream_reader is None:
                                    yolo_flush_frames = self.config.get('buffer_flush_after_yolo_frames', 50)
                                    self._flush_buffer_after_yolo(max_frames=yolo_flush_frames)

                                # ==================== 2. 修正：严格的 YOLO 校验逻辑 ====================
                                # 原代码: if people_count_for_merge is not None and people_count_for_merge <= 0:
                                # 致命问题: 当 YOLO 因为花屏报错返回 None 时，原代码会跳过 continue，导致"默认保存"
                                
                                # 新逻辑: 只要不是"明确有人"，一律跳过 (默认拒绝)
                                if people_count_for_merge is None or people_count_for_merge <= 0:
                                    if people_count_for_merge is None:
                                        # YOLO 没跑通（通常是因为花屏帧导致推理失败）
                                        logger.warning(
                                            f"YOLO检测异常(可能因花屏/坏帧)，为防止误报，跳过本次保存 - {self.camera_name}"
                                        )
                                    else:
                                        # YOLO 跑通了，确实没人
                                        logger.info(
                                            f"YOLO核实当前帧无人员(0人)，本次变化视为非人员事件，"
                                            f"不保存/合并记录 - 摄像头: {self.camera_name}"
                                        )
                                    
                                    # 重置连续帧计数，让后续检测重新开始
                                    self.detection.consecutive_change_count = 0

                                    # 慢速模式下未检测到人，计数+1
                                    if self.detection_mode == 'slow':
                                        self.no_person_count += 1
                                        logger.debug(
                                            f"[慢速模式] 连续{self.no_person_count}次未检测到人 "
                                            f"(阈值: {self.no_person_threshold}次)"
                                        )

                                        # 连续30次未检测到人，切回快速模式
                                        if self.no_person_count >= self.no_person_threshold:
                                            self.detection_mode = 'fast'
                                            self.no_person_count = 0
                                            logger.info(
                                                f"🔄 [{self.camera_name}] 连续{self.no_person_threshold}次未检测到人，"
                                                f"切换到快速检测模式"
                                            )

                                    continue  # 🚀 关键：遇到坏帧或无人，跳过保存，继续下一次循环

                                # YOLO确认有人，检查冷却时间（避免同一人频繁保存）
                                current_time = time.time()
                                time_since_last_process = current_time - self.last_process_time

                                if time_since_last_process < self.detection_interval:
                                    logger.debug(
                                        f"[{self.camera_name}] 检测到人员，但距离上次保存仅 {time_since_last_process:.1f}秒，"
                                        f"等待 {self.detection_interval}秒冷却，跳过本次保存"
                                    )
                                    # 继续检测，不进入慢速模式（因为确实有人）
                                    continue

                                # 更新保存时间
                                self.last_process_time = current_time

                                # 异步保存检测结果：将任务放入队列，由后台线程处理
                                logger.info(f"检测到人员，添加到处理队列...")
                                if not self.task_queue.full():
                                    # frame.copy() 非常重要！因为frame会被下一帧覆盖
                                    self.task_queue.put((frame.copy(), frame_time))
                                    logger.info(f"✓ 任务已加入队列 - 摄像头: {self.camera_name}")
                                else:
                                    logger.warning(f"任务队列已满，跳过当前帧 - {self.camera_name}")

                                # 检测到人员，切换到慢速模式（节省CPU）
                                if self.detection_mode == 'fast':
                                    self.detection_mode = 'slow'
                                    logger.info(
                                        f"🔄 [{self.camera_name}] 检测到人员，"
                                        f"切换到慢速检测模式（{self.slow_mode_interval}秒/次）"
                                    )

                                # 重置未检测到人的计数
                                self.no_person_count = 0

                                # 重置连续帧计数，避免后续继续触发
                                self.detection.consecutive_change_count = 0

                                # 新逻辑：不断开连接，继续监测
                                # 如果之后又检测到人，会根据时间间隔自动合并或创建新记录
                                logger.info(f"继续监测（慢速模式，主线程继续读取视频）...")
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

                        # 极短休眠让出CPU，避免100%占用，但不阻塞缓冲区
                        # 关键：这里只是礼让CPU，不是控制采样率
                        time.sleep(0.005)  # 5ms，既能让出CPU，又不影响实时性

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
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            except Exception as e:
                logger.debug(f"设置USB摄像头参数失败（将使用默认参数）: {e}")

            # 测试读取帧（快速验证，避免缓冲区积压）
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
                time.sleep(0.001)  # 极小延迟，避免缓冲区积压

            # 至少需要3帧成功
            if success_count >= 3:
                logger.info(
                    f"USB摄像头连接成功 - 摄像头: {self.camera_name} (设备ID: {usb_device_id}), "
                    f"成功读取 {success_count}/{test_frames} 测试帧"
                )

                # ============ 使用 VideoStreamReader 包装 VideoCapture ============
                # 根据配置决定是否使用独立线程读取器
                use_stream_reader = self.config.get('use_video_stream_reader', False)

                if use_stream_reader:
                    logger.info(
                        f"[{self.camera_name}] 使用 VideoStreamReader 包装 USB 摄像头 - "
                        f"独立线程持续读取最新帧，彻底解决缓冲区积压"
                    )
                    self.stream_reader = VideoStreamReader(
                        self.cap,
                        camera_name=self.camera_name,
                        use_pyav=False
                    )
                else:
                    logger.info(
                        f"[{self.camera_name}] USB摄像头使用定期清理模式 - "
                        f"每{self.config.get('buffer_flush_interval', 10)}秒清空缓冲区"
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
                
                # 使用TCP传输
                connection_success = False
                transport_methods = ['tcp']
                
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
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
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
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            except (AttributeError, cv2.error):
                logger.debug(f"无法设置缓冲区大小: {self.camera_name}")
            
            # 设置帧率（可选，帮助OpenCV理解流格式）
            try:
                self.cap.set(cv2.CAP_PROP_FPS, 25)  # RTSP通常25-30fps
            except (AttributeError, cv2.error):
                pass
            
            # 测试读取多帧，确保连接稳定（快速验证，避免缓冲区积压）
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
                time.sleep(0.001)  # 极小延迟，避免缓冲区积压
            
            # 至少需要3帧成功（提高要求，确保连接稳定）
            if success_count >= 3:
                logger.info(
                    f"RTSP流连接成功 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                    f"成功读取 {success_count}/{test_frames} 测试帧"
                )

                # 根据时间戳策略决定是否需要初始化PyAV
                timestamp_strategy = self.config.get('timestamp_strategy', 'realtime')

                if timestamp_strategy in ['pts_auto', 'pts_fixed']:
                    # 只有需要PTS时才初始化PyAV
                    logger.debug(f"时间戳策略为 {timestamp_strategy}，尝试初始化PyAV获取PTS支持")
                    if not self._init_pyav_stream():
                        logger.warning(
                            f"PyAV初始化失败，将继续使用OpenCV（无PTS支持） - 摄像头: {self.camera_name}"
                        )
                        self.use_pyav = False
                    else:
                        logger.info(f"PyAV初始化成功 - 摄像头: {self.camera_name}")
                        # 关闭OpenCV连接，改用PyAV
                        if self.cap is not None:
                            try:
                                self.cap.release()
                            except:
                                pass
                            self.cap = None
                        self.use_pyav = True
                else:
                    # realtime 策略不需要 PyAV，直接使用系统时间
                    logger.debug(
                        f"时间戳策略为 {timestamp_strategy}，使用系统时间，跳过PyAV初始化"
                    )
                    self.use_pyav = False

                # ============ 使用 VideoStreamReader 包装 VideoCapture ============
                # 根据配置决定是否使用独立线程读取器
                use_stream_reader = self.config.get('use_video_stream_reader', False)

                if not self.use_pyav and self.cap is not None and use_stream_reader:
                    logger.info(
                        f"[{self.camera_name}] 使用 VideoStreamReader 包装 VideoCapture - "
                        f"独立线程持续读取最新帧，彻底解决缓冲区积压"
                    )
                    # 用 VideoStreamReader 包装当前的 VideoCapture
                    self.stream_reader = VideoStreamReader(
                        self.cap,
                        camera_name=self.camera_name,
                        use_pyav=False
                    )
                    # 注意：VideoStreamReader 会接管 self.cap，不要再直接使用 self.cap.read()
                elif not self.use_pyav and self.cap is not None and not use_stream_reader:
                    logger.info(
                        f"[{self.camera_name}] 使用定期清理模式 - "
                        f"每{self.config.get('buffer_flush_interval', 10)}秒清空缓冲区"
                    )

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
        # 先释放 VideoStreamReader（它会释放内部的 VideoCapture）
        if self.stream_reader is not None:
            try:
                self.stream_reader.release()
            except:
                pass
            self.stream_reader = None

        # 如果 stream_reader 没有接管 cap，则手动释放 cap
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
        # 注意：time_offset_seconds 不重置，在重连时可以继续使用

    def _calibrate_time_from_rtcp(self) -> Optional[float]:
        """
        尝试从RTCP获取NTP时间戳并计算时间偏移

        Returns:
            float: 时间偏移量（秒），如果获取失败返回None
        """
        try:
            # 方法1: 检查container的start_time_realtime属性
            if hasattr(self.av_container, 'start_time_realtime') and self.av_container.start_time_realtime:
                # start_time_realtime 通常是微秒级的Unix时间戳
                ntp_timestamp_us = self.av_container.start_time_realtime
                ntp_datetime = datetime.fromtimestamp(ntp_timestamp_us / 1_000_000)

                # 计算偏移：系统时间 - NTP时间
                offset = (datetime.now() - ntp_datetime).total_seconds()

                logger.info(
                    f"✓ 从RTCP获取到NTP时间 - 摄像头: {self.camera_name}, "
                    f"NTP时间: {ntp_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, "
                    f"时间偏移: {offset:.2f}秒"
                )
                return offset

            # 方法2: 从stream metadata获取
            if self.av_stream and hasattr(self.av_stream, 'metadata') and self.av_stream.metadata:
                metadata = self.av_stream.metadata

                # 检查常见的时间相关metadata字段
                for key in ['creation_time', 'timecode', 'start_time']:
                    if key in metadata:
                        logger.debug(f"发现stream metadata[{key}]: {metadata[key]}")
                        # 尝试解析时间
                        try:
                            # ISO格式：2024-12-12T14:30:45.123Z
                            time_str = metadata[key]
                            if 'T' in time_str:
                                # 移除时区标识
                                time_str = time_str.replace('Z', '').split('+')[0].split('-')[0]
                                stream_time = datetime.strptime(time_str.split('.')[0], '%Y-%m-%dT%H:%M:%S')

                                offset = (datetime.now() - stream_time).total_seconds()
                                logger.info(
                                    f"✓ 从stream metadata获取到时间 - 摄像头: {self.camera_name}, "
                                    f"Stream时间: {stream_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                                    f"时间偏移: {offset:.2f}秒"
                                )
                                return offset
                        except Exception as parse_err:
                            logger.debug(f"解析metadata时间失败: {parse_err}")

            logger.debug(f"未能从RTCP/metadata获取时间信息 - 摄像头: {self.camera_name}")
            return None

        except Exception as e:
            logger.debug(f"RTCP时间校准失败 - 摄像头: {self.camera_name}, 错误: {e}")
            return None

    def _auto_calibrate_time_offset(self) -> float:
        """
        自动计算时间偏移量
        综合多种方法，按优先级尝试

        Returns:
            float: 时间偏移量（秒），默认返回0
        """
        calibration_method = self.config.get('time_calibration_method', 'auto')

        # 1. 如果配置了手动偏移，直接使用
        manual_offset = self.config.get('pts_time_offset')
        if manual_offset is not None:
            logger.info(
                f"使用手动配置的时间偏移 - 摄像头: {self.camera_name}, "
                f"偏移: {manual_offset:.2f}秒"
            )
            self.time_calibration_method = 'manual'
            return manual_offset

        # 2. 尝试从RTCP/metadata自动获取
        if calibration_method in ['auto', 'rtcp', 'stream_metadata']:
            rtcp_offset = self._calibrate_time_from_rtcp()
            if rtcp_offset is not None:
                self.time_calibration_method = 'rtcp'
                return rtcp_offset

        # 3. 默认：估算RTSP流延迟（基于缓冲区清空时间）
        # 这是一个粗略估算，假设第一帧到达时的延迟代表整体延迟
        logger.warning(
            f"⚠️ 无法自动校准时间偏移 - 摄像头: {self.camera_name}, "
            f"将使用默认值0秒（可能存在延迟）"
        )
        logger.warning(
            f"💡 建议：在config.py中设置 'pts_time_offset' 参数来手动校准时间偏移"
        )
        self.time_calibration_method = 'none'
        return 0.0

    def _init_pyav_stream(self) -> bool:
        """
        初始化PyAV流以获取PTS支持

        注意：此函数失败不影响OpenCV连接，可以继续使用OpenCV读取帧

        Returns:
            bool: 初始化成功返回True
        """
        try:
            # 释放之前的PyAV资源（不影响OpenCV的self.cap）
            self._release_pyav()

            # 打开RTSP流
            logger.debug(f"正在使用PyAV打开RTSP流: {self.camera_name}")
            # [camera_monitor.py]
            self.av_container = av.open(
            self.rtsp_url,
            options={
                'rtsp_transport': 'tcp',
                'max_delay': '3000000',     # [修改] 从 500000 改为 3000000 (3秒)，允许更大的网络抖动
                'stimeout': '10000000',     # [修改] socket超时增加到 10秒
                'buffer_size': '10240000',  # [修改] 接收缓冲区从 1MB 增加到 10MB
                'rtsp_flags': 'prefer_tcp',
            },
            timeout=20.0  # [修改] 连接超时增加到 20秒
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

                    logger.debug(f"PTS时间基准已建立 - {self.camera_name}")

                    # 时间校准已简化（直接使用系统时间）
                    if self.time_offset_seconds is None:
                        self.time_offset_seconds = 0  # 不再需要复杂校准
                        logger.debug(
                            f"PTS时间基准已建立，使用简化校准 - 摄像头: {self.camera_name}"
                        )

                    # 重新创建解码器（因为已经读取了一帧）
                    self.av_decoder = self.av_container.decode(self.av_stream)

                    return True
                else:
                    logger.debug(f"第一帧PTS为空，尝试下一帧: {self.camera_name}")

            logger.debug(f"无法获取有效的PTS时间戳，使用系统时间: {self.camera_name}")
            self._release_pyav()
            return False

        except Exception as e:
            logger.error(f"初始化PyAV流失败 - 摄像头: {self.camera_name}, 错误: {e}", exc_info=True)
            self._release_pyav()
            return False

    def _pts_to_datetime(self, pts: int) -> datetime:
        """
        将PTS转换为真实时间（应用时间偏移校准）

        Args:
            pts: Presentation Time Stamp

        Returns:
            datetime: 转换后的真实时间（已校准）
        """
        if pts is None or self.time_base is None or self.pts_base_time is None:
            return datetime.now()

        # 计算当前帧的PTS秒数
        pts_seconds = float(pts * self.time_base)

        # 转换为真实时间：基准时间 + (当前PTS - 基准PTS)
        real_time = self.pts_base_time + timedelta(seconds=(pts_seconds - self.pts_base_offset))

        # ==================== 应用时间偏移校准 ====================
        # 如果存在时间偏移，修正RTSP流延迟
        if self.time_offset_seconds is not None and self.time_offset_seconds != 0:
            # 减去偏移量，得到摄像头的真实拍摄时间
            # 例如：如果RTSP流延迟5秒，time_offset_seconds=5，则 real_time -= 5秒
            real_time -= timedelta(seconds=self.time_offset_seconds)

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
                    # 重置use_pyav标志，下次重连时会重新尝试PyAV
                    self.use_pyav = False
                else:
                    # 其他错误（解码错误等）
                    logger.error(
                        f"PyAV读取帧失败 - 摄像头: {self.camera_name}, "
                        f"错误: {e}, 将触发重连"
                    )
                    # 同样释放资源并重置标志
                    self._release_pyav()
                    self.use_pyav = False

                return False, None, None
        else:
            # 使用OpenCV读取（USB摄像头或fallback）
            # 优先使用 VideoStreamReader（独立线程模式）
            if self.stream_reader is not None:
                try:
                    ret, frame, frame_time = self.stream_reader.read()
                    # stream_reader.read() 已经返回时间戳，直接使用
                    return ret, frame, frame_time
                except Exception as e:
                    logger.error(
                        f"VideoStreamReader读取帧失败 - 摄像头: {self.camera_name}, "
                        f"错误: {e}"
                    )
                    return False, None, None

            # 降级方案：直接从 VideoCapture 读取（不推荐，会有缓冲区积压）
            if self.cap is None:
                logger.error(
                    f"OpenCV VideoCapture为空 - 摄像头: {self.camera_name}, "
                    f"可能是连接已断开，需要重连"
                )
                return False, None, None

            try:
                ret, frame = self.cap.read()
                frame_time = datetime.now() if ret else None
                return ret, frame, frame_time
            except Exception as e:
                logger.error(
                    f"OpenCV读取帧失败 - 摄像头: {self.camera_name}, "
                    f"错误: {e}"
                )
                return False, None, None

    def _flush_buffer_after_yolo(self, max_frames: int = 50) -> int:
        """
        YOLO检测后清空缓冲区到底，追上最新进度

        在主线程执行YOLO检测时（耗时200-500ms），视频流持续产生帧导致缓冲区积压。
        此方法会持续grab()直到缓冲区为空，确保后续读取的是最新帧。

        关键逻辑：
        1. 持续调用grab()直到返回False（缓冲区空了）
        2. max_frames作为安全上限，防止异常情况

        Args:
            max_frames: 安全上限（防止无限循环），默认50帧

        Returns:
            int: 清空的帧数
        """
        flushed_count = 0

        try:
            if self.use_pyav and self.av_decoder is not None:
                # PyAV 模式：快速消费解码器中的帧，直到清空
                for _ in range(max_frames):
                    try:
                        frame = next(self.av_decoder, None)
                        if frame is None:
                            # 缓冲区已空
                            break
                        flushed_count += 1
                    except StopIteration:
                        break
                    except Exception:
                        break

            elif self.cap is not None and self.cap.isOpened():
                # OpenCV 模式：快速读取并丢弃，直到清空
                for _ in range(max_frames):
                    ret = self.cap.grab()  # grab() 比 read() 快，只解码不返回
                    if not ret:
                        # 缓冲区已空
                        break
                    flushed_count += 1

            if flushed_count > 0:
                if flushed_count >= max_frames:
                    logger.warning(
                        f"⚠️ [{self.camera_name}] YOLO检测后缓冲区清理已达上限 {max_frames} 帧 "
                        f"(约 {flushed_count/25:.2f}秒延迟)，可能仍未清空！建议增加buffer_flush_after_yolo_frames"
                    )
                else:
                    logger.debug(
                        f"[{self.camera_name}] YOLO检测后清空缓冲区，丢弃 {flushed_count} 帧旧数据 "
                        f"(约 {flushed_count/25:.2f}秒延迟) - 已清空"
                    )

        except Exception as e:
            logger.warning(f"[{self.camera_name}] 清空缓冲区异常: {e}")

        return flushed_count

    def _flush_buffer_smart(self, max_frames: int = 100, reason: str = "智能清理") -> int:
        """
        智能缓冲区清理 - 清空到底，确保彻底清除积压

        多摄像头场景下，缓冲区容易累积延迟。此方法会持续清空直到缓冲区为空，
        确保 realtime 时间戳机制的准确性（datetime.now() 对应的是最新帧）。

        关键逻辑：
        1. 持续调用grab()直到返回False（缓冲区空了）
        2. max_frames作为安全上限，防止异常情况下的死循环
        3. 清空后，下一帧就是"最新鲜"的帧

        Args:
            max_frames: 安全上限（防止死循环），默认100帧（约4秒）
            reason: 清理原因（用于日志）

        Returns:
            int: 实际清空的帧数
        """
        flushed_count = 0

        try:
            if self.use_pyav and self.av_decoder is not None:
                # PyAV 模式：快速消费解码器中的帧，直到清空
                for _ in range(max_frames):
                    try:
                        frame = next(self.av_decoder, None)
                        if frame is None:
                            # 缓冲区已空
                            break
                        flushed_count += 1
                    except StopIteration:
                        # 流结束或缓冲区已空
                        break
                    except Exception:
                        break

            elif self.cap is not None and self.cap.isOpened():
                # OpenCV 模式：使用grab()快速跳过，直到清空
                for _ in range(max_frames):
                    ret = self.cap.grab()  # grab() 只解码不返回，速度极快
                    if not ret:
                        # 缓冲区已空，停止清理
                        break
                    flushed_count += 1

            # 根据清理结果给出不同的日志
            if flushed_count > 0:
                # 计算延迟时间（假设25fps）
                delay_seconds = flushed_count / 25.0

                # 如果清理了很多帧，说明积压严重，需要警告
                if flushed_count >= max_frames:
                    logger.warning(
                        f"⚠️ [{self.camera_name}] 缓冲区严重积压！{reason}已达上限 {max_frames} 帧 "
                        f"(约 {delay_seconds:.2f}秒延迟)，可能仍未清空！"
                        f"建议：1)减少定期清理间隔 2)增加max_frames上限"
                    )
                elif flushed_count > 30:
                    # 积压超过1秒，给出提示
                    logger.warning(
                        f"⚠️ [{self.camera_name}] 缓冲区积压较多！{reason}丢弃 {flushed_count} 帧 "
                        f"(约 {delay_seconds:.2f}秒延迟) - 已清空，下一帧即最新"
                    )
                else:
                    # 正常清理
                    logger.info(
                        f"✓ [{self.camera_name}] {reason}丢弃 {flushed_count} 帧 "
                        f"(约 {delay_seconds:.2f}秒延迟) - 缓冲区已清空，realtime时间戳准确"
                    )
            else:
                # 缓冲区无积压，也输出INFO级别日志（方便观察清理是否在工作）
                logger.info(f"✓ [{self.camera_name}] {reason}，缓冲区无积压（理想状态）- 清理机制正常工作")

        except Exception as e:
            logger.warning(f"[{self.camera_name}] 智能缓冲区清理异常: {e}")

        return flushed_count

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
    
    def _save_detection_result(self, frame, detection_time=None):
        """
        保存检测结果（包含YOLO人员检测）
        新逻辑：
        1. 先在内存中进行YOLO检测
        2. 只有检测到人才保存图片和数据库记录
        3. 30秒内的检测合并到同一条轨迹记录

        Args:
            frame: 视频帧(numpy array)
            detection_time: 检测时间
        """
        try:
            from pathlib import Path

            # 0. 首先确定准确的检测时间（用于文件名和数据库）
            current_time = detection_time if detection_time is not None else datetime.now()

            # 1. 先在内存中进行YOLO检测（避免无效的磁盘I/O）
            people_count = None
            try:
                # 方式1：使用共享YOLO实例池
                if self.yolo_pool is not None:
                    try:
                        detections = self.yolo_pool.detect(
                            frame,
                            timeout=config.YOLO_POOL_CONFIG.get('timeout', 30)
                        )
                        people_count = len(detections)
                        logger.info(
                            f"✓ YOLO池检测结果 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                            f"人数: {people_count}"
                        )

                        # 如果没有检测到人，直接返回，不保存任何文件
                        if people_count <= 0:
                            logger.debug(
                                f"YOLO未检测到人员，跳过保存 - 摄像头: {self.camera_name}"
                            )
                            return

                    except Exception as pool_err:
                        logger.error(
                            f"使用YOLO池检测失败 - 摄像头: {self.camera_name}, 错误: {pool_err}"
                        )
                        people_count = None

                # 方式2：使用独立YOLO实例（向后兼容）
                elif self.person_detector is not None or self.yolo_pool is None:
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

                    # 直接在内存中检测（不需要先保存到硬盘）
                    if self.person_detector is not None:
                        # 根据检测器类型选择检测方法
                        if hasattr(self.person_detector, 'detect_image'):
                            # 自适应检测器（YOLOv11）- 直接传入frame
                            detections = self.person_detector.detect_image(frame)
                        elif hasattr(self.person_detector, 'get_person_count'):
                            # 某些检测器可能有get_person_count方法
                            people_count = self.person_detector.get_person_count(frame)
                            detections = [{}] * people_count if people_count > 0 else []
                        else:
                            # 传统YOLO检测器 - 使用model.predict
                            results = self.person_detector.model.predict(
                                frame,
                                conf=self.person_detector.conf_threshold,
                                iou=self.person_detector.iou_threshold,
                                classes=[0],  # 只检测人
                                device=self.person_detector.device,
                                verbose=False,
                            )
                            if results:
                                boxes = results[0].boxes
                                detections = [{}] * len(boxes) if boxes is not None else []
                            else:
                                detections = []

                        people_count = len(detections)
                        logger.info(
                            f"✓ 内存YOLO检测结果 - 摄像头: {self.camera_name} (ID: {self.camera_id}), "
                            f"人数: {people_count}"
                        )

                        # 如果没有检测到人，直接返回，不保存任何文件
                        if people_count <= 0:
                            logger.debug(
                                f"YOLO未检测到人员，跳过保存 - 摄像头: {self.camera_name}"
                            )
                            return
                    else:
                        logger.warning(
                            f"YOLO检测器不可用，无法统计人数，将继续保存记录但不写入人数字段 - 摄像头: {self.camera_name}"
                        )
                        people_count = None

            except Exception as det_err:
                logger.error(
                    f"执行YOLO人员检测时发生错误，将继续保存记录但不写入人数字段: {det_err}",
                    exc_info=True,
                )
                people_count = None

            # 2. 只有检测到人才保存图片（节省磁盘I/O和存储空间）
            logger.info(f"检测到 {people_count} 人，正在保存证据...")
            image_relative_path = self.storage.save_image(
                frame,
                self.camera_id,
                self.camera_name,
                capture_time=current_time
            )

            if image_relative_path is None:
                logger.warning(f"保存图片失败，跳过记录 - 摄像头: {self.camera_name}")
                return

            # 3. 将相对路径转换为完整的URL路径
            image_url = self.storage.get_image_url(image_relative_path)

            # 4. 检查是否在合并窗口内，判断是否应该合并到现有轨迹
            # 从配置文件读取合并间隔，默认30秒
            merge_interval = self.config.get('trajectory_merge_interval', 30)
            should_merge = False

            if self.current_trajectory is not None:
                # 计算距离上次检测的时间差
                time_diff = (current_time - self.current_trajectory['last_time']).total_seconds()
                if time_diff <= merge_interval:
                    should_merge = True
                    logger.info(
                        f"检测到活动在{merge_interval}秒窗口内({time_diff:.1f}秒)，合并到现有轨迹 - "
                        f"摄像头: {self.camera_name}, 轨迹ID: {self.current_trajectory['record_id']}"
                    )
                else:
                    logger.info(
                        f"距离上次检测已超过{merge_interval}秒({time_diff:.1f}秒)，创建新轨迹 - "
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
                        track_id=record_id,
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
                        track_id=record_id,
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

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.running

