"""
ReID系统配置模块
为core模块提供Config类，适配bjjcspbj的配置
"""
# 使用绝对导入，明确指定模块名避免与config.py冲突
# reid_config_adapter.py 在文件末尾已经创建了 Config = ReIDConfigAdapter() 实例
try:
    # 方法1: 直接导入（如果路径正确）
    import reid_config_adapter
    if hasattr(reid_config_adapter, 'Config'):
        Config = reid_config_adapter.Config
    else:
        # 如果Config不存在，创建实例
        Config = reid_config_adapter.ReIDConfigAdapter()
except ImportError:
    # 方法2: 如果直接导入失败，使用importlib
    import importlib.util
    from pathlib import Path
    
    adapter_path = Path(__file__).parent / 'reid_config_adapter.py'
    if adapter_path.exists():
        spec = importlib.util.spec_from_file_location("reid_config_adapter", adapter_path)
        adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter_module)
        if hasattr(adapter_module, 'Config'):
            Config = adapter_module.Config
        else:
            Config = adapter_module.ReIDConfigAdapter()
    else:
        # 如果文件不存在，创建占位符
        class ConfigPlaceholder:
            def __getattr__(self, name):
                raise ImportError(f"reid_config_adapter.py 文件不存在: {adapter_path}")
        Config = ConfigPlaceholder()

# 导出Config，供core模块使用
__all__ = ['Config']

