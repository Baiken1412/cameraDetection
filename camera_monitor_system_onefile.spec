# -*- mode: python ; coding: utf-8 -*-
"""
RTSP视频流监测系统 - 打包配置
改为文件夹模式（one-dir），解决 onefile 模式 struct.error TOC 超限问题
输出目录：dist/CameraMonitorSystem/
运行方式：双击 dist/CameraMonitorSystem/CameraMonitorSystem.exe
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all
import onnxruntime
import openvino

block_cipher = None

# ========================================================
# 1. 收集 OpenVINO 和 ONNX Runtime 的资源
#    去掉原来的"三重添加"策略，只保留结构化添加（策略A），避免 TOC 超限
# ========================================================
ov_path = os.path.dirname(openvino.__file__)
print(f"OpenVINO Path: {ov_path}")
ov_datas, ov_binaries, ov_hiddenimports = collect_all('openvino')

ort_path = os.path.dirname(onnxruntime.__file__)
ort_datas, ort_binaries, ort_hiddenimports = collect_all('onnxruntime')

# 手动补充 OpenVINO DLL（仅保留结构化路径，不再铺平到根目录）
extra_binaries = []
extra_datas = []

for root, dirs, files in os.walk(ov_path):
    for file in files:
        if file.endswith(('.dll', '.pyd', '.so', '.dylib', '.bin', '.xml')):
            source = os.path.join(root, file)
            rel_dir = os.path.relpath(root, os.path.dirname(ov_path))
            extra_binaries.append((source, os.path.join('openvino', rel_dir)))

extra_datas.append((ov_path, 'openvino'))

# 手动补充 ONNX Runtime DLL（仅结构化路径）
# 排除 CUDA/TensorRT provider DLL，这些 DLL 在无 GPU 机器上初始化会崩溃
_ort_gpu_skip = ('cuda', 'tensorrt', 'trt', 'dnnl', 'openvino')
for root, dirs, files in os.walk(ort_path):
    for file in files:
        if file.endswith(('.dll', '.pyd', '.so')):
            if any(x in file.lower() for x in _ort_gpu_skip):
                continue
            source = os.path.join(root, file)
            rel_dir = os.path.relpath(root, os.path.dirname(ort_path))
            extra_binaries.append((source, os.path.join('onnxruntime', rel_dir)))

# collect_all 收集的 ort_binaries 里也可能含 GPU provider，同样过滤
ort_binaries = [(s, d) for s, d in ort_binaries
                if not any(x in s.lower() for x in _ort_gpu_skip)]

# ========================================================
# 2. 项目自身资源文件
# ========================================================
added_files = [
    ('models',      'models'),
    ('static',      'static'),
    ('templates',   'templates'),
    ('data',        'data'),
    ('*.sql',       '.'),
    ('config.json', '.'),   # 配置文件打包进去，部署时只需复制一个文件夹
    ('快速上手.txt', '.'),
]

all_datas    = added_files + ov_datas + extra_datas + ort_datas
all_binaries = ov_binaries + extra_binaries + ort_binaries

# ========================================================
# 3. 隐式导入
# ========================================================
my_hidden_imports = [
    # 视频/图像处理
    'av', 'cv2', 'numpy', 'PIL',
    # 数据库 / Web / 日志
    'pymysql', 'loguru', 'cryptography', 'flask', 'docx', 'markdown',
    # 推理引擎
    'openvino', 'openvino.runtime', 'openvino.frontend',
    'openvino.frontend.onnx',
    # 本项目核心模块
    'config_loader', 'database', 'camera_monitor', 'license_manager',
    'video_stream_reader', 'input_shaper', 'image_storage', 'image_detection',
    'core.detector', 'core.person_detector_adaptive',
    'core.yolo_pool', 'core.yolo_process_pool',
    'utils.cpu_detector', 'utils.device_manager', 'utils.file_handler',
    'utils.image_processor',
    # ReID 相关（运行时按需加载，保留声明避免导入失败）
    'reid_config', 'reid_config_adapter', 'reid_database',
    'reid_integration', 'reid_web_app', 'person_management_api',
    'person_manager',
]

# ========================================================
# 4. Analysis
# ========================================================
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=ov_hiddenimports + ort_hiddenimports + my_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_onnxruntime.py'],
    excludes=[
        'config',
        # 以下库在运行时不需要，排除掉可大幅缩减体积
        'torch', 'torchvision', 'torchaudio', 'torchreid',
        'tensorflow', 'tensorflow_core', 'tensorflow_estimator', 'keras',
        'bitsandbytes',
        'polars',
        'pandas',
        'scipy',
        'sklearn', 'scikit_learn',
        'matplotlib',
        'sympy',
        'IPython', 'ipykernel', 'jupyter',
        'pyarrow',
        'transformers',
        'huggingface_hub',
        'diffusers',
        'accelerate',
        'nncf',
        'imageio',
        'skimage',
        'kaleido',
        'plotly',
        'bokeh',
        # onnx/onnxslim 仅用于模型导出，运行时不需要
        # 排除可避免 PyInstaller 扫描 onnx.reference 时的 access violation
        'onnx', 'onnxslim',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ========================================================
# 5. EXE（文件夹模式：exclude_binaries=True，binaries/datas 交给 COLLECT）
# ========================================================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # 关键：文件夹模式不把依赖嵌入 exe
    name='CameraMonitorSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ========================================================
# 6. COLLECT（把所有 DLL/数据文件收集到输出文件夹）
# ========================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CameraMonitorSystem',     # 输出到 dist/CameraMonitorSystem/
)

# ========================================================
# 7. 打包后清理：删除 GPU provider DLL
#    这些 DLL 在无 CUDA/TensorRT 机器上会导致 onnxruntime 初始化崩溃
#    即使在 binaries 中过滤，hook 也会重新加回来，故在此强制删除
# ========================================================
import glob as _glob
_ort_capi = os.path.join(DISTPATH, 'CameraMonitorSystem', '_internal', 'onnxruntime', 'capi')
for _dll in ['onnxruntime_providers_cuda.dll',
             'onnxruntime_providers_tensorrt.dll',
             'onnxruntime_providers_migraphx.dll',
             'onnxruntime_providers_rocm.dll']:
    _p = os.path.join(_ort_capi, _dll)
    if os.path.exists(_p):
        os.remove(_p)
        print(f"[post-build] Removed GPU provider DLL: {_dll}")
