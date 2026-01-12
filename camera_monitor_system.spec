# -*- mode: python ; coding: utf-8 -*-
"""
RTSP视频流监测系统 - PyInstaller配置文件
"""

block_cipher = None

# 需要包含的数据文件
added_files = [
    ('models', 'models'),
    ('static', 'static'),
    ('templates', 'templates'),
    ('data', 'data'),
    ('*.sql', '.'),
    ('config.json', '.'),  # 配置文件模板，首次运行自动复制到exe同级
    ('使用说明.txt', '.'),  # 使用说明文档
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
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
    ],
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
    console=True,  # 显示控制台窗口以查看日志
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 如果有 icon 文件，可以在这里指定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    [('config.json', 'config.json', 'DATA')],  # 配置文件放到exe同级
    [('使用说明.txt', '使用说明.txt', 'DATA')],  # 使用说明放到exe同级
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CameraMonitorSystem',
)
