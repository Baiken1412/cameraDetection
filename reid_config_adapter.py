"""
ReID配置适配器
将 bjjcspbj 的配置映射到 ReID 系统需要的 Config 类格式
"""
import os
import sys
from pathlib import Path
import torch

# 导入 bjjcspbj 的配置
sys.path.insert(0, str(Path(__file__).parent))
import config as bjjcspbj_config


class ReIDConfigAdapter:
    """ReID配置适配器，使用 bjjcspbj 的配置"""
    
    # ========== 路径配置 ==========
    @property
    def PROJECT_ROOT(self):
        """项目根目录"""
        return Path(__file__).parent.absolute()
    
    @property
    def INPUT_IMAGE_DIR(self):
        """输入图片目录 - 使用 RTSP_MONITOR_CONFIG 中的 image_save_path"""
        # 优先使用 RTSP_MONITOR_CONFIG 中的 image_save_path
        rtsp_config = bjjcspbj_config.RTSP_MONITOR_CONFIG
        image_save_path = rtsp_config.get('image_save_path', 'data/input_images')
        
        # 如果配置了绝对路径，直接使用
        if os.path.isabs(image_save_path):
            return image_save_path
        
        # 否则使用相对路径
        return str(self.PROJECT_ROOT / image_save_path)
    
    @property
    def OUTPUT_DIR(self):
        """输出目录"""
        return str(Path(self.INPUT_IMAGE_DIR).parent)
    
    @property
    def CROPPED_DIR(self):
        """裁剪后的人员图片目录"""
        reid_config = bjjcspbj_config.REID_CONFIG
        path = reid_config.get('cropped_dir', 'data/cropped_persons')
        if os.path.isabs(path):
            return path
        return str(self.PROJECT_ROOT / path)
    
    @property
    def FEATURES_DIR(self):
        """特征文件目录"""
        reid_config = bjjcspbj_config.REID_CONFIG
        path = reid_config.get('features_dir', 'data/features')
        if os.path.isabs(path):
            return path
        return str(self.PROJECT_ROOT / path)
    
    @property
    def RESULTS_DIR(self):
        """结果保存目录"""
        reid_config = bjjcspbj_config.REID_CONFIG
        path = reid_config.get('results_dir', 'data/results')
        if os.path.isabs(path):
            return path
        return str(self.PROJECT_ROOT / path)
    
    @property
    def MODELS_DIR(self):
        """模型目录 - 使用本地models目录"""
        # 在打包后的exe环境中，优先使用exe所在目录
        if getattr(sys, 'frozen', False):
            # PyInstaller打包后的exe
            exe_dir = Path(sys.executable).parent
            models_dir = exe_dir / 'models'
        else:
            # 开发环境
            models_dir = Path(__file__).parent / 'models'
        models_dir.mkdir(exist_ok=True)
        return str(models_dir)
    
    @property
    def LOGS_DIR(self):
        """日志目录"""
        return str(Path(__file__).parent / 'logs')
    
    @property
    def LOG_FILE(self):
        """日志文件路径"""
        return str(Path(self.LOGS_DIR) / 'reid_process.log')
    
    # ========== 模型配置 ==========
    @property
    def YOLO_MODEL_NAME(self):
        return "yolov8m.pt"  # 从 yolov8x 切换到 yolov8m (4-6倍速度提升)
    
    @property
    def YOLO_MODEL_PATH(self):
        return str(Path(self.MODELS_DIR) / self.YOLO_MODEL_NAME)
    
    @property
    def REID_MODEL_NAME(self):
        return bjjcspbj_config.REID_CONFIG.get('reid_model_name', 'osnet_x1_0')
    
    @property
    def REID_PRETRAINED(self):
        return bjjcspbj_config.REID_CONFIG.get('reid_pretrained', True)
    
    # ========== 检测参数 ==========
    @property
    def DETECTION_CONF_THRESHOLD(self):
        return bjjcspbj_config.REID_CONFIG.get('detection_conf_threshold', 0.5)
    
    @property
    def DETECTION_IOU_THRESHOLD(self):
        return bjjcspbj_config.REID_CONFIG.get('detection_iou_threshold', 0.5)
    
    @property
    def DETECTION_BATCH_SIZE(self):
        return 8
    
    # ========== 特征提取参数 ==========
    @property
    def FEATURE_DIM(self):
        return 512
    
    @property
    def FEATURE_BATCH_SIZE(self):
        return 32
    
    @property
    def INPUT_SIZE(self):
        return (256, 128)
    
    @property
    def IMAGENET_MEAN(self):
        return [0.485, 0.456, 0.406]
    
    @property
    def IMAGENET_STD(self):
        return [0.229, 0.224, 0.225]
    
    # ========== 输入图片格式 ==========
    @property
    def SUPPORTED_IMAGE_FORMATS(self):
        """
        支持的输入图片扩展名列表
        对应原 ReID Config 中的 SUPPORTED_IMAGE_FORMATS，用于 scan_images 过滤文件
        """
        # 常见图片格式，后端保存的抓拍图片通常是 .jpg
        return [".jpg", ".jpeg", ".png", ".bmp"]
    
    # ========== 聚类参数 ==========
    @property
    def SIMILARITY_THRESHOLD(self):
        return bjjcspbj_config.REID_CONFIG.get('similarity_threshold', 0.7)
    
    @SIMILARITY_THRESHOLD.setter
    def SIMILARITY_THRESHOLD(self, value):
        """允许动态修改相似度阈值"""
        bjjcspbj_config.REID_CONFIG['similarity_threshold'] = value
    
    @property
    def SIMILARITY_METRIC(self):
        return bjjcspbj_config.REID_CONFIG.get('similarity_metric', 'cosine')
    
    # ========== 设备配置 ==========
    @property
    def DEVICE(self):
        device = bjjcspbj_config.REID_CONFIG.get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return device
    
    # ========== 数据库配置 ==========
    @property
    def USE_DATABASE(self):
        return True  # bjjcspbj 始终使用数据库
    
    @property
    def DB_HOST(self):
        return bjjcspbj_config.REID_DATABASE_CONFIG['host']
    
    @property
    def DB_PORT(self):
        return bjjcspbj_config.REID_DATABASE_CONFIG['port']
    
    @property
    def DB_USER(self):
        return bjjcspbj_config.REID_DATABASE_CONFIG['user']
    
    @property
    def DB_PASSWORD(self):
        return bjjcspbj_config.REID_DATABASE_CONFIG['password']
    
    @property
    def DB_NAME(self):
        return bjjcspbj_config.REID_DATABASE_CONFIG['database']
    
    @property
    def DB_CHARSET(self):
        return bjjcspbj_config.REID_DATABASE_CONFIG['charset']
    
    # ========== Flask配置 ==========
    @property
    def FLASK_HOST(self):
        return bjjcspbj_config.REID_CONFIG.get('flask_host', '0.0.0.0')
    
    @property
    def FLASK_PORT(self):
        return bjjcspbj_config.REID_CONFIG.get('flask_port', 5000)
    
    @property
    def FLASK_DEBUG(self):
        return bjjcspbj_config.REID_CONFIG.get('flask_debug', True)
    
    @property
    def SECRET_KEY(self):
        return bjjcspbj_config.REID_CONFIG.get('secret_key', 'bjjcspbj-reid-secret-key-2024')
    
    # ========== 其他配置 ==========
    @property
    def MANUAL_CONFIRMATION_FILE(self):
        return str(Path(self.RESULTS_DIR) / 'manual_confirmations.json')
    
    @property
    def ANCHOR_FILE(self):
        return str(Path(self.RESULTS_DIR) / 'anchors.json')
    
    @property
    def PENDING_QUEUE_FILE(self):
        return str(Path(self.RESULTS_DIR) / 'pending_queue.json')
    
    def ensure_dirs(self):
        """确保所有必要的目录存在"""
        dirs = [
            self.INPUT_IMAGE_DIR,
            self.CROPPED_DIR,
            self.FEATURES_DIR,
            self.RESULTS_DIR,
            self.LOGS_DIR,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        return True
    
    @classmethod
    def get_device_info(cls):
        """获取设备信息"""
        instance = cls()
        if instance.DEVICE == 'cuda':
            return f"CUDA - {torch.cuda.get_device_name(0)}"
        return "CPU"


# 创建全局实例，作为 Config 使用
Config = ReIDConfigAdapter()

