"""
设备管理工具模块
"""
import logging
import torch

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    获取计算设备

    Args:
        prefer_gpu: 是否优先使用GPU

    Returns:
        torch.device: 设备对象
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")

    return device


def get_device_info() -> dict:
    """
    获取设备详细信息

    Returns:
        dict: 设备信息
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cpu'
    }

    if info['cuda_available']:
        info['current_device'] = 'cuda'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda

        # GPU内存信息
        info['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        info['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
        info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB

    return info


def print_device_info():
    """打印设备信息"""
    info = get_device_info()

    logger.info("=" * 50)
    logger.info("设备信息:")
    logger.info(f"CUDA可用: {info['cuda_available']}")

    if info['cuda_available']:
        logger.info(f"GPU名称: {info['gpu_name']}")
        logger.info(f"CUDA版本: {info['cuda_version']}")
        logger.info(f"GPU数量: {info['device_count']}")
        logger.info(f"GPU内存: {info['gpu_memory_total']:.2f} GB (总计)")
        logger.info(f"         {info['gpu_memory_allocated']:.2f} GB (已分配)")
        logger.info(f"         {info['gpu_memory_reserved']:.2f} GB (已保留)")
    else:
        logger.info("GPU不可用，将使用CPU")

    logger.info("=" * 50)


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU缓存已清理")
