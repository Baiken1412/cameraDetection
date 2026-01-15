# -*- mode: python ; coding: utf-8 -*-
"""
RTSP视频流监测系统 - 最终修复版 V2 (修复 ValueError)
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all
import onnxruntime
import openvino

block_cipher = None

# ========================================================
# 1. 资源收集逻辑 (保持不变，这是正确的)
# ========================================================
# --- OpenVINO ---
ov_path = os.path.dirname(openvino.__file__)
print(f"OpenVINO Path: {ov_path}")
ov_datas, ov_binaries, ov_hiddenimports = collect_all('openvino')

# --- ONNX Runtime ---
ort_path = os.path.dirname(onnxruntime.__file__)
ort_datas, ort_binaries, ort_hiddenimports = collect_all('onnxruntime')

# --- 手动暴力收集 DLL (构造二元组) ---
extra_binaries = []
extra_datas = []

# OpenVINO 遍历
for root, dirs, files in os.walk(ov_path):
    for file in files:
        if file.endswith(('.dll', '.pyd', '.so', '.dylib', '.bin', '.xml')):
            source = os.path.join(root, file)
            rel_dir = os.path.relpath(root, os.path.dirname(ov_path))
            # 策略 A: 保持结构
            extra_binaries.append((source, os.path.join('openvino', rel_dir)))
            # 策略 B: 铺平到根目录
            extra_binaries.append((source, '.'))
            # 策略 C: libs 目录
            if 'libs' in rel_dir:
                 extra_binaries.append((source, 'libs'))

extra_datas.append((ov_path, 'openvino'))

# ONNX Runtime 遍历
for root, dirs, files in os.walk(ort_path):
    for file in files:
        if file.endswith(('.dll', '.pyd', '.so')):
            source = os.path.join(root, file)
            extra_binaries.append((source, '.'))
            extra_binaries.append((source, os.path.join('onnxruntime', os.path.relpath(root, os.path.dirname(ort_path)))))

# ========================================================
# 2. 合并所有配置
# ========================================================
added_files = [
    ('models', 'models'),
    ('static', 'static'),
    ('templates', 'templates'),
    ('data', 'data'),
    ('*.sql', '.'),
    ('config.json', '.'),
    ('使用说明.txt', '.'),
]

# 所有的二元组 (source, dest)
all_datas = added_files + ov_datas + extra_datas + ort_datas
# 【关键修改】这里把手动收集的 binaries 也合并进来，交给 Analysis 处理
all_binaries = ov_binaries + extra_binaries + ort_binaries

my_hidden_imports = [
    'av', 'cv2', 'numpy', 'PIL', 'pymysql', 'loguru', 
    'cryptography', 'flask', 'docx', 'markdown',
    'openvino', 'openvino.runtime', 'openvino.frontend',
    'openvino.frontend.pytorch', 'openvino.frontend.tensorflow', 'openvino.frontend.onnx',
    'config_loader', 'database', 'camera_monitor', 'license_manager',
    'video_stream_reader', 'image_storage', 'image_detection',
    'core.detector', 'core.detector_onnx', 'core.feature_extractor',
    'core.feature_extractor_onnx', 'core.ensemble_extractor',
    'core.similarity', 'core.clustering', 'core.pending_queue',
    'core.manual_confirmation', 'core.db_confirmation_manager',
    'core.anchor_manager', 'core.person_detector_adaptive',
    'utils.cpu_detector', 'utils.device_manager', 'utils.file_handler',
    'utils.image_processor', 'pipeline.reid_pipeline',
    'reid_config', 'reid_config_adapter', 'reid_database',
    'reid_integration', 'reid_web_app', 'person_management_api',
    'person_manager'
]

# ========================================================
# 3. Analysis (核心修正)
# ========================================================
a = Analysis(
    ['main.py'],
    pathex=[],
    # 【关键修改】在这里传入 all_binaries，Analysis 会负责格式转换
    binaries=all_binaries, 
    datas=all_datas,
    hiddenimports=ov_hiddenimports + ort_hiddenimports + my_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['config'], 
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ========================================================
# 4. EXE (核心修正)
# ========================================================
exe = EXE(
    pyz,
    a.scripts,
    # 【关键修改】只使用 a.binaries，因为 Analysis 已经包含了我们合并的内容并做了标准化
    a.binaries, 
    a.zipfiles,
    a.datas, 
    [],
    name='CameraMonitorSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)