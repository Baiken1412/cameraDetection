# -*- mode: python ; coding: utf-8 -*-
# 新版打包配置 - 单文件 exe，config.json 外置可修改

import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# 收集需要的包
datas = []
binaries = []
hiddenimports = []

# models 目录（YOLO 模型文件）
datas += [('models', 'models')]

# config.json 作为默认模板打包进去（首次运行会复制到 exe 同级）
datas += [('config.json', '.')]

# onnxruntime 已通过 force_engine=openvino 绕过，不再收集（节省 ~400MB）

# 收集 openvino（如果有）
try:
    tmp_ret = collect_all('openvino')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass

# 收集 cv2
try:
    tmp_ret = collect_all('cv2')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception:
    pass

hiddenimports += [
    # pkg_resources / jaraco 依赖
    'jaraco.text',
    'jaraco.functools',
    'jaraco.context',
    'jaraco.collections',
    'more_itertools',
    'multiprocessing',
    'multiprocessing.pool',
    'multiprocessing.managers',
    'multiprocessing.process',
    'multiprocessing.queues',
    'multiprocessing.synchronize',
    'queue',
    'loguru',
    'pymysql',
    'requests',
    'urllib3',
    'PIL',
    'PIL.Image',
    'numpy',
    'cv2',
    'config_loader',
    'config',
    'database',
    'camera_monitor',
    'license_manager',
    'image_storage',
    'image_detection',
    'video_stream_reader',
    'input_shaper',
    'core.person_detector_adaptive',
    'core.yolo_pool',
    'core.yolo_process_pool',
    'core.detector',
    'core.detector_onnx',
    'core.detector_openvino',
    'core.feature_extractor',
    'core.feature_extractor_onnx',
    'core.similarity',
    'core.anchor_manager',
    'core.clustering',
    'core.db_confirmation_manager',
    'core.pending_queue',
    'core.ensemble_extractor',
    'core.manual_confirmation',
    'pipeline.reid_pipeline',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=['rthook_onnxruntime.py'] if __import__('os').path.exists('rthook_onnxruntime.py') else [],
    excludes=[
        # UI / 开发工具
        'tkinter', 'tkinter.ttk',
        'IPython', 'jupyter', 'notebook', 'ipykernel', 'ipywidgets',
        'pytest', 'unittest',
        'setuptools', 'distutils', 'pip',
        # 数据科学（项目未使用）
        'tensorflow', 'tensorflow_core', 'tensorboard',
        'torch', 'torchvision', 'torchaudio',
        'sklearn', 'scikit_learn',
        'scipy',
        'pandas',
        'pyarrow',
        'polars',
        'skimage', 'scikit_image',
        'matplotlib',
        'plotly', 'kaleido',
        'sympy',
        'statsmodels',
        # NLP / 向量库（项目未使用）
        'transformers', 'tokenizers', 'huggingface_hub',
        'hf_xet', 'hf_transfer',
        'chromadb', 'hnswlib', 'faiss',
        'bitsandbytes',
        'jieba', 'jieba.analyse',
        'nltk', 'spacy',
        'sentence_transformers',
        # 其他大型库
        'onnxruntime',
        'grpc', 'grpcio',
        'google.protobuf', 'google.cloud',
        'boto3', 'botocore',
        'sqlalchemy', 'alembic',
        'celery', 'redis',
        'wx', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'docutils', 'sphinx',
        'lxml',
        'bs4', 'scrapy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='camera_monitor_system',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
