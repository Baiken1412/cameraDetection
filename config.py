"""
配置文件
"""
import os
from pathlib import Path

# 数据库配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '1412',
    'database': 'caseappdb',
    'charset': 'utf8mb4'
}


# RTSP监测配置
RTSP_MONITOR_CONFIG = {
    # 采样间隔（秒）- 正常情况下每秒检测一次
    'sample_interval': 1,

    # 检测到变化后的等待间隔（秒）- 如果检测到有人，则等待此时间后再检测
    'detection_wait_interval': 10,

    # 前景阈值（0.001表示0.1%的前景，降低阈值提高敏感度）
    # 如果检测不到变化，可以进一步降低到0.0005（0.05%）
    'change_threshold': 0.02,

    # ==================== 时间校准配置 ====================
    # 时间戳策略
    # 'realtime': 使用实时系统时间（推荐，最准确）- 清空缓冲区后使用当前系统时间
    # 'pts_auto': 使用PTS时间+自动校准 - 尝试从RTCP获取偏移，失败则使用PTS
    # 'pts_fixed': 使用PTS时间+固定偏移 - 需手动配置pts_time_offset
    'timestamp_strategy': 'realtime',

    # PTS固定偏移（仅当timestamp_strategy='pts_fixed'时使用）
    'pts_time_offset': None,

    # 是否启用时间诊断日志（用于调试）
    'enable_time_diagnosis': True,

    # ==================== 缓冲区管理配置 ====================
    # 是否启用独立线程读取模式（VideoStreamReader）
    # true: 每个摄像头独立线程持续读取，实时性最高，但CPU占用增加50%
    # false: 使用定期清理模式，CPU占用低，实时性稍差
    # 建议：
    #   - 摄像头数量 <= 5：启用（true）- 实时性优先
    #   - 摄像头数量 > 5：禁用（false）- 性能优先
    'use_video_stream_reader': True,  # 改为False，使用简单的定期清理方案

    # 智能缓冲区清理间隔（秒）- 仅当 use_video_stream_reader=False 时生效
    # 多摄像头场景下，定期清空缓冲区可避免时间错位
    # 建议值：
    #   - 10秒：高实时性要求场景（强制时间同步）
    #   - 20秒：平衡模式（默认，适合大多数场景）
    #   - 30秒：人员稀少场景
    # 注意：USB摄像头可以设置更长间隔（缓冲区积压少）
    'buffer_flush_interval': 10,  # 改为10秒，更频繁清理

    # 智能清理的安全上限（帧数）
    # 注意：这是**安全上限**，不是目标值！
    # 实际清理会持续grab()直到缓冲区为空，max_frames只是防止死循环
    # 建议值：100-200帧（25fps下约4-8秒）
    # 如果日志显示"已达上限"警告，说明积压严重，需要增加此值或缩短清理间隔
    'buffer_flush_max_frames': 150,  # 增加到150帧，防止"已达上限"警告

    # YOLO检测后清理的安全上限（帧数）
    # YOLO耗时200-500ms，期间积压约5-12帧
    # 但为了安全起见，设置更大的上限
    # 建议值：50-100帧
    'buffer_flush_after_yolo_frames': 100,  # 增加到100帧，确保彻底清空

    # 重连间隔（秒）
    'reconnect_interval': 5,
    
    # 最大重连次数
    'max_reconnect_attempts': 10,
    
    # 图片保存路径（本地文件系统路径，绝对路径）
    'image_save_path': 'D:/appdata/uploadPath/caseapp',  # 例如: 'D:/ruoyi/uploadPath/caseapp'

    # 图片访问URL前缀（用于存储到数据库的URL路径）
    # 本地开发环境：使用localhost + Java服务端口8090（HTTPS）
    'image_url_prefix': '',

    # 是否保存图片到文件系统（True:保存文件，False:Base64存数据库）
    'save_image_to_file': True,

    # JPEG 保存质量（1-100，默认95）
    'jpeg_quality': 95,

    # 图片最大宽度（像素），超过此宽度的图片将等比缩放
    'max_image_width': 1920,

    # 最小记录间隔（秒），保存完当前图片后，等待此时间后再保存下一张
    'min_record_interval': 10,  # 改为10秒，支持更频繁的记录合并

    # 检测到人员后等待时间（秒），保存图片后等待此时间再开始检测
    # 注意：此参数已弃用，改为持续监测模式
    'detection_wait_after_save': 120,  # 2分钟 = 120秒（已弃用）
    
    # 记录合并间隔（秒），如果新检测到的人员与上一条记录的时间间隔小于此值，则合并记录
    # 设为 0 禁用合并，每次检测都创建独立轨迹记录
    'record_merge_interval': 0,  # 禁用合并功能

    # 轨迹合并时间窗口（秒）
    # 如果在此时间窗口内再次检测到人员，会合并到同一轨迹中，而不是创建新轨迹
    # 调整建议：
    #   - 30秒：适合正常走动场景（默认）
    #   - 60秒：适合人员停留或缓慢移动场景
    #   - 15秒：适合快速通过场景
    'trajectory_merge_interval': 60,

    # RTSP连接超时（秒）
    'rtsp_timeout': 30,  # 增加到30秒，RTSP连接可能需要更长时间
    
    # 形态学操作（用于去除小噪声点）
    # 0=不使用, 1=开运算, 2=闭运算, 3=开运算+闭运算
    'morphology_operation': 1,
    
    # 最小变化区域面积（像素数），小于此值的变化区域将被忽略
    # 降低此值可以提高敏感度，但可能增加噪声
    # 如果原始前景比例较高但处理后为0，说明过滤太激进，应该降低此值
    'min_change_area': 500,  # 从500降低到150，减少过滤

    # 摄像头配置获取接口（代替直接查询 app_roomip 表）
    # 说明：系统通过此接口获取所有摄像头的 RTSP 实时流地址
    # 方法：POST，返回格式示例：
    # {
    #   "msg": "操作成功",
    #   "code": 0,
    #   "data": [ { id, fjmc, ip, dk, tdh, xh, zh, mm, sblx, gnslx, rtspssl } ]
    # }
    # 本地开发环境：使用localhost + Java服务端口8090（HTTPS，需要Java系统已启动）
    'camera_api_url': 'https://localhost:8090/caseapp/track/rtspStream',
    # 是否验证 HTTPS 证书（本地开发使用HTTP，不需要验证）
    'camera_api_verify_ssl': False,
    # 接口请求超时时间（秒）
    'camera_api_timeout': 10,  # 本地开发增加超时时间，避免调试时超时
    
    # 连续帧检测：连续N帧都有变化才认为有变化（减少误报）
    'consecutive_frames_threshold': 2,
    
    # 背景建模参数
    'bg_subtractor_type': 'MOG2',  # 'MOG2' 或 'KNN'
    'bg_learning_rate': 0.001,  # 背景学习率（0-1，检测阶段使用，越小越稳定但适应慢）
    'bg_history': 500,  # 历史帧数（MOG2使用）
    'bg_var_threshold': 10,  # 方差阈值（MOG2使用，越小越敏感，默认16，如果检测不到可以降低到10-12）
    'bg_detect_shadows': True,  # 是否检测阴影（MOG2使用，True可减少阴影误报）
    
    # 可视化页面显示近期记录的时间范围（分钟），只显示这个时间范围内的检测记录
    'recent_records_minutes': 30,  # 默认显示最近30分钟的记录

    # USB摄像头支持
    'enable_usb_camera': False,  # 是否启用USB摄像头
    'usb_camera_id': 0,  # USB摄像头设备ID（通常是0，如果有多个摄像头可以是1,2等）
    'usb_camera_name': 'USB摄像头',  # USB摄像头名称
    'usb_camera_area': 'USB监控区域'  # USB摄像头监控区域名称
}

# ReID人员识别配置
# 注意：使用自适应检测后，ReID功能已弃用（专注于人员检测和计数）
REID_CONFIG = {
    # 是否启用ReID人员识别功能
    'enabled': False,  # 已弃用，使用ADAPTIVE_DETECTION_CONFIG代替
    
    # 注意：ReID系统已整合到caseapp项目中，不再需要外部路径
    # 核心模块位于：core/, pipeline/, utils/ 目录
    
    # 人员匹配阈值（0.0-1.0），相似度超过此值才认为匹配
    'match_threshold': 0.7,  # 默认0.7，可根据实际情况调整
    
    # ReID训练相关配置
    'input_image_dir': 'data/input_images',  # 输入图片目录
    'cropped_dir': 'data/cropped_persons',  # 裁剪后的人员图片目录
    'results_dir': 'data/results',  # 结果保存目录
    'features_dir': 'data/features',  # 特征文件目录
    'similarity_threshold': 0.7,  # 相似度阈值
    'detection_conf_threshold': 0.5,  # 检测置信度阈值
    'detection_iou_threshold': 0.5,  # 检测IOU阈值
    'reid_model_name': 'osnet_x1_0',  # ReID模型名称
    'reid_pretrained': True,  # 是否使用预训练模型
    'similarity_metric': 'cosine',  # 相似度计算方法
    'device': 'cuda',  # 计算设备: 'cuda' 或 'cpu'
    'flask_host': '0.0.0.0',  # Flask服务地址
    'flask_port': 5000,  # Flask服务端口
    'flask_debug': True,  # Flask调试模式
    'secret_key': 'caseapp-reid-secret-key-2024',  # Flask密钥
}

# ReID数据库配置（已与主库合并，使用与 DATABASE_CONFIG 相同的数据源）
# 说明：为兼容旧代码保留 REID_DATABASE_CONFIG 名称，但实际指向同一数据库。
REID_DATABASE_CONFIG = {
    'host': DATABASE_CONFIG['host'],
    'port': DATABASE_CONFIG['port'],
    'user': DATABASE_CONFIG['user'],
    'password': DATABASE_CONFIG['password'],
    'database': DATABASE_CONFIG['database'],
    'charset': DATABASE_CONFIG['charset'],
}

# 日志配置
LOG_CONFIG = {
    'level': 'DEBUG',  # 改为DEBUG以便查看详细检测信息
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
    'file': 'logs/rtsp_monitor.log',
    'rotation': '10 MB',
    'retention': '30 days'
}

# 自适应人员检测配置（YOLOv11 + 自动选择推理引擎）
ADAPTIVE_DETECTION_CONFIG = {
    # 启用自适应检测（自动检测CPU类型，选择最优推理引擎）
    'enabled': True,  # True=启用自适应检测, False=使用传统方式

    # 模型目录
    'model_dir': 'models',

    # 检测参数
    'conf_threshold': 0.5,  # 置信度阈值（0.0-1.0）
    'iou_threshold': 0.4,   # NMS IOU阈值（0.0-1.0）

    # 推理引擎配置（自动检测）
    # None=自动检测CPU类型并选择引擎
    # 'openvino'=强制使用OpenVINO（Intel CPU）
    # 'onnx'=强制使用ONNX Runtime（AMD/通用CPU）
    'force_engine': None,

    # CPU线程数配置
    # None=自动（CPU核心数-2）
    # 手动指定线程数（如：8, 12, 16）
    'num_threads': None,
}

# 确保目录存在
def ensure_directories():
    """确保必要的目录存在"""
    # 使用绝对路径确保目录存在
    image_path = RTSP_MONITOR_CONFIG['image_save_path']
    if not os.path.isabs(image_path):
        # 如果是相对路径，转换为绝对路径
        script_dir = Path(__file__).parent.absolute()
        image_path = script_dir / image_path
    Path(image_path).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)

