"""
pytest 配置文件
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line(
        "markers", "slow: 标记为慢速测试（默认跳过，使用 --slow 运行）"
    )


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--slow", action="store_true", default=False, help="运行慢速测试"
    )


def pytest_collection_modifyitems(config, items):
    """根据选项跳过慢速测试"""
    if config.getoption("--slow"):
        return

    skip_slow = pytest.mark.skip(reason="需要 --slow 选项运行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
