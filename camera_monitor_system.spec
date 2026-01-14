# -*- mode: python ; coding: utf-8 -*-
"""
RTSP视频流监测系统 - PyInstaller配置文件
"""

from PyInstaller.utils.hooks import collect_all
import os
import onnxruntime  # <---【关键修改1】导入库以获取真实路径

block_cipher = None

# 1. 使用 collect_all 自动收集 onnxruntime 的基础依赖
ort_datas, ort_binaries, ort_hiddenimports = collect_all('onnxruntime')

# 2. 【关键修改2】强制添加 onnxruntime 完整包路径
# 这可以解决部分 DLL 加载路径不匹配或文件遗漏的问题
ort_path = os.path.dirname(onnxruntime.__file__)
# 格式: (源路径, 目标路径) -> 将整个文件夹拷贝到顶层 onnxruntime 目录
force_ort_datas = [(ort_path, 'onnxruntime')]

# 3. 定义原本需要的资源文件
added_files = [
    ('models', 'models'),
    ('static', 'static'),
    ('templates', 'templates'),
    ('data', 'data'),
    ('*.sql', '.'),
    ('config.json', '.'),  # 配置文件模板
    ('使用说明.txt', '.'),  # 使用说明文档
]

# 4. 合并所有数据文件：原有文件 + 自动收集的ORT数据 + 强制拷贝的ORT数据
all_datas = added_files + ort_datas + force_ort_datas

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=ort_binaries,
    datas=all_datas,  # <--- 使用合并后的 datas
    hiddenimports=[
        'av',
        'cv2',
        'numpy',
        'PIL',
        'pymysql',
        'loguru',
        'cryptography',
        'flask',
        'docx',
        'markdown',
        'onnxruntime',
        'openvino',
        # 核心模块
        'config_loader',
        'database',
        'camera_monitor',
        'license_manager',
        'video_stream_reader',
        'image_storage',
        'image_detection',
        # core 包
        'core.detector',
        'core.detector_onnx',
        'core.feature_extractor',
        'core.feature_extractor_onnx',
        'core.ensemble_extractor',
        'core.similarity',
        'core.clustering',
        'core.pending_queue',
        'core.manual_confirmation',
        'core.db_confirmation_manager',
        'core.anchor_manager',
        'core.person_detector_adaptive',
        # utils 包
        'utils.cpu_detector',
        'utils.device_manager',
        'utils.file_handler',
        'utils.image_processor',
        # pipeline 包
        'pipeline.reid_pipeline',
        # ReID 相关
        'reid_config',
        'reid_config_adapter',
        'reid_database',
        'reid_integration',
        'reid_web_app',
        'person_management_api',
        'person_manager',
    ] + ort_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CameraMonitorSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 保持开启以查看报错信息
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    # 下面这两行再次添加是为了确保它们暴露在根目录，方便用户修改
    [('config.json', 'config.json', 'DATA')],
    [('使用说明.txt', '使用说明.txt', 'DATA')],
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CameraMonitorSystem',
)