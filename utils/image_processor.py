"""
图像处理工具模块
"""
import logging
from typing import Union, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def crop_image(
    image: np.ndarray,
    bbox: list,
    padding: float = 0.0
) -> np.ndarray:
    """
    裁剪图像

    Args:
        image: 原始图像
        bbox: 边界框 [x1, y1, x2, y2]
        padding: 填充比例（0.1表示10%填充）

    Returns:
        np.ndarray: 裁剪后的图像
    """
    x1, y1, x2, y2 = map(int, bbox)

    # 添加填充
    if padding > 0:
        width = x2 - x1
        height = y2 - y1

        x1 = max(0, int(x1 - width * padding))
        y1 = max(0, int(y1 - height * padding))
        x2 = min(image.shape[1], int(x2 + width * padding))
        y2 = min(image.shape[0], int(y2 + height * padding))

    crop = image[y1:y2, x1:x2]
    return crop


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    keep_aspect_ratio: bool = False
) -> np.ndarray:
    """
    调整图像大小

    Args:
        image: 输入图像
        size: 目标大小 (width, height)
        keep_aspect_ratio: 是否保持宽高比

    Returns:
        np.ndarray: 调整后的图像
    """
    if keep_aspect_ratio:
        # 保持宽高比，填充黑边
        h, w = image.shape[:2]
        target_w, target_h = size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建黑色背景
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 居中放置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas
    else:
        # 直接拉伸
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return resized


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    加载图像（支持中文路径）

    Args:
        image_path: 图像路径

    Returns:
        np.ndarray: BGR格式的图像
    """
    image_path = str(image_path)

    # 使用 numpy + cv2.imdecode 来支持中文路径
    # OpenCV 的 cv2.imread 在 Windows 上无法正确处理中文路径
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"无法加载图像: {image_path}, 错误: {e}")

    if image is None:
        raise ValueError(f"图像解码失败: {image_path}")

    return image


def save_image(image: np.ndarray, save_path: Union[str, Path], quality: int = 95) -> bool:
    """
    保存图像（支持中文路径）

    Args:
        image: 图像数组
        save_path: 保存路径
        quality: JPEG质量 (0-100)，默认95

    Returns:
        bool: 保存是否成功
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 cv2.imencode + 文件写入 来支持中文路径
    # OpenCV 的 cv2.imwrite 在 Windows 上无法正确处理中文路径
    try:
        # 根据文件扩展名选择编码格式
        ext = save_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_image = cv2.imencode('.jpg', image, encode_param)
        elif ext == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            success, encoded_image = cv2.imencode('.png', image, encode_param)
        else:
            # 默认使用JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_image = cv2.imencode('.jpg', image, encode_param)

        if not success:
            logger.error(f"图像编码失败: {save_path}")
            return False

        # 写入文件
        with open(str(save_path), 'wb') as f:
            f.write(encoded_image.tobytes())

        return True
    except Exception as e:
        logger.error(f"保存图像失败: {save_path}, 错误: {e}")
        return False


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR转RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB转BGR"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """NumPy数组转PIL Image"""
    if len(image.shape) == 2:
        # 灰度图
        return Image.fromarray(image)
    else:
        # BGR -> RGB
        rgb_image = bgr_to_rgb(image)
        return Image.fromarray(rgb_image)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL Image转NumPy数组（BGR格式）"""
    rgb_array = np.array(image)
    if len(rgb_array.shape) == 2:
        # 灰度图
        return rgb_array
    else:
        # RGB -> BGR
        bgr_array = rgb_to_bgr(rgb_array)
        return bgr_array
