# PyInstaller runtime hook
# 在每个子进程启动时注册 onnxruntime / openvino 的 DLL 目录
# 解决 multiprocessing 子进程中 DLL 加载失败的问题
import os
import sys

# 强制 UTF-8 模式，解决路径含中文时 OpenVINO C++ 代码 UnicodeDecodeError
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    import ctypes

    base = sys._MEIPASS

    dll_search_dirs = [
        os.path.join(base, 'onnxruntime', 'capi'),
        os.path.join(base, 'openvino', 'libs'),
        os.path.join(base, 'openvino'),
        base,
    ]

    # 注册 DLL 搜索目录
    for dll_dir in dll_search_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

    # 预加载 onnxruntime 的核心 DLL（仅当 onnxruntime 存在时）
    ort_capi = os.path.join(base, 'onnxruntime', 'capi')
    if os.path.isdir(ort_capi):
        for dll_name in [
            'onnxruntime_providers_shared.dll',
            'onnxruntime.dll',
        ]:
            dll_path = os.path.join(ort_capi, dll_name)
            if os.path.exists(dll_path):
                try:
                    ctypes.CDLL(dll_path)
                except Exception:
                    pass  # onnxruntime 不可用时静默跳过
