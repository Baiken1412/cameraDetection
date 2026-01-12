"""
配置加载器 - 从外部 config.json 加载配置
支持打包后的可执行文件从外部读取配置
"""
import json
import os
import sys
import shutil
from pathlib import Path

def get_base_path():
    """获取程序基础路径"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后
        return Path(sys.executable).parent
    else:
        # 开发环境
        return Path(__file__).parent

def get_internal_path():
    """获取 _internal 目录路径"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，_internal 在 exe 同级
        return Path(sys.executable).parent / '_internal'
    else:
        # 开发环境，没有 _internal
        return None

def get_config_path():
    """
    获取配置文件路径
    优先级：
    1. exe 同级目录的 config.json（用户可修改）
    2. _internal 目录的 config.json（默认模板）
    """
    base_path = get_base_path()
    user_config = base_path / 'config.json'

    # 如果用户配置存在，直接使用
    if user_config.exists():
        return user_config

    # 如果是打包后的程序，尝试从 _internal 复制默认配置
    internal_path = get_internal_path()
    if internal_path:
        internal_config = internal_path / 'config.json'
        if internal_config.exists():
            try:
                # 复制默认配置到 exe 同级目录
                shutil.copy2(internal_config, user_config)
                print(f"已从模板创建配置文件: {user_config}")
                return user_config
            except Exception as e:
                print(f"警告: 无法复制配置文件: {e}")
                # 如果复制失败，直接使用 _internal 中的配置
                return internal_config

    # 开发环境或找不到配置文件
    return user_config

def load_config():
    """
    加载配置文件
    """
    config_path = get_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请确保 config.json 文件位于程序所在目录"
        )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"成功加载配置文件: {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    except Exception as e:
        raise Exception(f"加载配置文件失败: {e}")

# 加载配置
_config = load_config()

# 导出配置变量（保持与原 config.py 相同的接口）
DATABASE_CONFIG = _config.get('DATABASE_CONFIG', {})
RTSP_MONITOR_CONFIG = _config.get('RTSP_MONITOR_CONFIG', {})
REID_CONFIG = _config.get('REID_CONFIG', {})
REID_DATABASE_CONFIG = _config.get('REID_DATABASE_CONFIG', {})
LOG_CONFIG = _config.get('LOG_CONFIG', {})
ADAPTIVE_DETECTION_CONFIG = _config.get('ADAPTIVE_DETECTION_CONFIG', {})

def ensure_directories():
    """确保必要的目录存在"""
    # 使用绝对路径确保目录存在
    image_path = RTSP_MONITOR_CONFIG.get('image_save_path', '')
    if image_path:
        if not os.path.isabs(image_path):
            # 如果是相对路径，转换为绝对路径
            if getattr(sys, 'frozen', False):
                base_path = Path(sys.executable).parent
            else:
                base_path = Path(__file__).parent
            image_path = base_path / image_path
        Path(image_path).mkdir(parents=True, exist_ok=True)

    Path('logs').mkdir(parents=True, exist_ok=True)
