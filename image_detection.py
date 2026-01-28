"""
图像变化检测模块
使用背景建模法（Background Subtraction）检测画面变化
"""
import random
import cv2
import numpy as np
from loguru import logger
import config_loader as config


class ImageChangeDetection:
    """图像变化检测类 - 仅使用背景建模法"""
    
    def __init__(self):
        self.config = config.RTSP_MONITOR_CONFIG
        self.threshold = self.config['change_threshold']
        self.morphology_operation = self.config.get('morphology_operation', 1)
        self.min_change_area = self.config.get('min_change_area', 100)
        self.consecutive_frames_threshold = self.config.get('consecutive_frames_threshold', 2)
        
        # 用于连续帧检测的计数器
        self.consecutive_change_count = 0
        
        # 缓存最后一次计算的前景比例（避免重复计算）
        self.last_change_ratio = None
        
        # 背景建模器（延迟初始化）
        self.bg_subtractor = None
        self.bg_initialized = False
        
        logger.info("初始化背景建模检测器")
    
    def detect_change(self, current_frame, previous_frame=None):
        """
        检测画面是否发生变化（使用连续帧检测减少误报）
        
        Args:
            current_frame: 当前帧（numpy数组或OpenCV Mat）
            previous_frame: 前一帧（可选，背景建模不需要）
        
        Returns:
            bool: True表示有变化，False表示无变化
        """
        # 使用背景建模计算前景比例
        change_ratio = self._calculate_background_subtraction(current_frame)
        
        # 缓存计算结果
        self.last_change_ratio = change_ratio
        
        if change_ratio is None:
            self.consecutive_change_count = 0
            return False
        
        # 判断当前帧是否有变化
        has_change_this_frame = change_ratio > self.threshold
        
        if has_change_this_frame:
            # 当前帧有变化，增加计数
            self.consecutive_change_count += 1
        else:
            # 当前帧无变化，重置计数
            self.consecutive_change_count = 0
        
        # 只有连续N帧都有变化才认为真的有变化
        return self.consecutive_change_count >= self.consecutive_frames_threshold
    
    def calculate_change_ratio(self, current_frame, previous_frame=None):
        """
        计算画面变化比例（兼容旧接口）
        
        Args:
            current_frame: 当前帧
            previous_frame: 前一帧（可选，背景建模不需要）
        
        Returns:
            float: 变化比例 (0.0 - 1.0)，如果无法计算返回None
        """
        # 先调用detect_change计算变化比例（会缓存到last_change_ratio）
        self.detect_change(current_frame, previous_frame)
        return self.last_change_ratio
    
    def _calculate_background_subtraction(self, current_frame):
        """
        背景建模法：计算前景（运动物体）比例
        
        Returns:
            float: 前景比例 (0.0 - 1.0)，如果无法计算返回None
        """
        try:
            # 确保背景建模器已初始化（应该在camera_monitor中初始化）
            if not self.bg_initialized or self.bg_subtractor is None:
                logger.warning("背景建模器未初始化，尝试初始化...")
                bg_type = self.config.get('bg_subtractor_type', 'MOG2')
                
                if bg_type == 'MOG2':
                    history = self.config.get('bg_history', 500)
                    var_threshold = self.config.get('bg_var_threshold', 16)
                    detect_shadows = self.config.get('bg_detect_shadows', True)
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=history,
                        varThreshold=var_threshold,
                        detectShadows=detect_shadows
                    )
                else:  # KNN
                    history = self.config.get('bg_history', 500)
                    dist2_threshold = self.config.get('bg_var_threshold', 400)
                    detect_shadows = self.config.get('bg_detect_shadows', True)
                    self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                        history=history,
                        dist2Threshold=dist2_threshold,
                        detectShadows=detect_shadows
                    )
                
                self.bg_initialized = True
                logger.info(f"背景建模器初始化完成，类型: {bg_type}")
            
            # 确保是numpy数组
            if not isinstance(current_frame, np.ndarray):
                current_frame = np.array(current_frame)
            
            # 应用背景减除
            # 重要：如果检测到前景，使用极低的学习率（甚至0），避免把前景学习成背景
            # 这样可以防止前景长时间停留后被学习成背景，导致后续检测不到
            base_learning_rate = self.config.get('bg_learning_rate', 0.001)
            
            # 先使用0学习率检测前景（不更新背景模型，只检测）
            temp_mask = self.bg_subtractor.apply(current_frame, learningRate=0)
            temp_fg_ratio = np.count_nonzero(temp_mask) / temp_mask.size if temp_mask.size > 0 else 0
            
            # 根据前景比例决定是否更新背景模型
            if temp_fg_ratio > self.threshold:  # 如果检测到前景
                # 有运动物体，暂停背景学习，避免把前景学习成背景
                # 直接使用第一次检测的结果，不再更新背景模型
                fg_mask = temp_mask
                logger.debug(f"检测到前景 ({temp_fg_ratio*100:.3f}%)，暂停背景学习，避免前景被学习成背景")
            else:
                # 无运动物体，正常学习率更新背景模型
                # 重新应用一次，这次会更新背景模型
                fg_mask = self.bg_subtractor.apply(current_frame, learningRate=base_learning_rate)
            
            # 调试输出：定期输出前景掩码统计信息（处理前）
            fg_pixels_before = np.count_nonzero(fg_mask)
            total_pixels = fg_mask.size
            fg_ratio_before = fg_pixels_before / total_pixels if total_pixels > 0 else 0
            
            # 如果前景比例很低，输出警告（可能有问题）
            if fg_ratio_before < 0.0001 and random.random() < 0.2:  # 20%概率输出
                logger.warning(f"⚠️  背景建模原始前景比例极低: {fg_ratio_before*100:.4f}%，可能原因：1)学习阶段画面中有运动物体 2)方差阈值太高 3)背景模型未正确建立")
            
            # 形态学操作去除小噪声点
            fg_mask = self._apply_morphology(fg_mask)
            
            # 过滤小的变化区域
            fg_mask = self._filter_small_areas(fg_mask)
            
            # 调试输出：处理后
            fg_pixels_after = np.count_nonzero(fg_mask)
            fg_ratio_after = fg_pixels_after / total_pixels if total_pixels > 0 else 0
            
            # 如果处理后前景比例很低，输出详细信息
            if fg_ratio_after < 0.0001 and random.random() < 0.2:  # 20%概率输出
                logger.debug(f"背景建模 - 原始: {fg_pixels_before}/{total_pixels} ({fg_ratio_before*100:.3f}%), "
                           f"处理后: {fg_pixels_after}/{total_pixels} ({fg_ratio_after*100:.3f}%)")
            
            # 如果原始前景比例很高但处理后很低，说明过滤太激进
            if fg_ratio_before > 0.01 and fg_ratio_after < 0.001:
                logger.warning(
                    f"⚠️  过滤可能过于激进：原始前景 {fg_ratio_before*100:.2f}% -> 处理后 {fg_ratio_after*100:.3f}%"
                    f"（形态学操作: {self.morphology_operation}, 最小区域: {self.min_change_area}）"
                    f"建议：降低min_change_area或减小形态学核大小"
                )
            
            # 计算前景比例
            return self._calculate_ratio(fg_mask)
            
        except Exception as e:
            logger.error(f"背景建模计算异常: {e}", exc_info=True)
            return None
    
    def _apply_morphology(self, thresh):
        """应用形态学操作去除小噪声点"""
        if self.morphology_operation > 0:
            # 使用较小的核（3x3而不是5x5），减少过滤强度
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if self.morphology_operation == 1 or self.morphology_operation == 3:
                # 开运算：去除小的噪声点
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            if self.morphology_operation == 2 or self.morphology_operation == 3:
                # 闭运算：填充小的空洞
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh
    
    def _filter_small_areas(self, thresh):
        """过滤小的变化区域（只保留较大的变化区域）"""
        if self.min_change_area > 0:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            thresh_filtered = np.zeros_like(thresh)
            total_filtered = 0
            kept_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_change_area:
                    cv2.drawContours(thresh_filtered, [contour], -1, 255, -1)
                    kept_areas.append(area)
                else:
                    total_filtered += area
            
            # 调试输出：如果过滤掉了大量区域，输出警告
            if total_filtered > 0 and random.random() < 0.1:
                filtered_ratio = total_filtered / thresh.size if thresh.size > 0 else 0
                if filtered_ratio > 0.01:  # 如果过滤掉了超过1%的区域
                    logger.debug(f"过滤了小区域，过滤面积: {total_filtered}像素 ({filtered_ratio*100:.3f}%)，保留区域数: {len(kept_areas)}")
            
            return thresh_filtered
        return thresh
    
    def _calculate_ratio(self, mask):
        """计算前景比例"""
        total_pixels = mask.size
        changed_pixels = np.count_nonzero(mask)
        change_ratio = changed_pixels / total_pixels if total_pixels > 0 else 0
        return float(change_ratio)
    
    def reset(self):
        """重置检测器（用于重新初始化）"""
        self.bg_subtractor = None
        self.bg_initialized = False
        self.consecutive_change_count = 0
        self.last_change_ratio = None
        logger.info("背景建模检测器已重置")

