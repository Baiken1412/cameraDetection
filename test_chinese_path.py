"""
测试 OpenCV 中文路径支持
验证修复后的工具函数是否正确处理中文路径
"""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2

# 导入修复后的工具函数
from utils.image_processor import load_image, save_image


def test_chinese_path():
    """测试中文路径的图片读写"""
    print("=" * 60)
    print("测试 OpenCV 中文路径支持")
    print("=" * 60)

    # 创建测试图片（彩色渐变图）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 创建渐变效果
    for i in range(480):
        test_image[i, :, 0] = int(255 * i / 480)  # Blue
        test_image[i, :, 1] = 128  # Green
        test_image[i, :, 2] = int(255 * (480 - i) / 480)  # Red

    # 创建临时目录进行测试
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试1: 纯英文路径
        print("\n[测试1] 纯英文路径")
        english_path = Path(tmpdir) / "test_english" / "image.jpg"
        english_path.parent.mkdir(parents=True, exist_ok=True)

        success = save_image(test_image, english_path)
        print(f"  保存: {english_path}")
        print(f"  结果: {'成功' if success else '失败'}")

        if success:
            try:
                loaded = load_image(english_path)
                print(f"  读取: 成功 (尺寸: {loaded.shape})")
            except Exception as e:
                print(f"  读取: 失败 - {e}")

        # 测试2: 含中文的路径
        print("\n[测试2] 含中文的路径")
        chinese_path = Path(tmpdir) / "大门区域摄像头1" / "测试图片_20260102.jpg"
        chinese_path.parent.mkdir(parents=True, exist_ok=True)

        success = save_image(test_image, chinese_path, quality=85)
        print(f"  保存: {chinese_path}")
        print(f"  结果: {'成功' if success else '失败'}")

        if success:
            try:
                loaded = load_image(chinese_path)
                print(f"  读取: 成功 (尺寸: {loaded.shape})")

                # 验证图片内容是否正确
                if loaded.shape == test_image.shape:
                    # 由于JPEG压缩有损，只检查大致相似度
                    diff = np.abs(loaded.astype(float) - test_image.astype(float)).mean()
                    print(f"  差异: {diff:.2f} (JPEG压缩导致的误差)")
                    if diff < 10:  # 允许少量误差
                        print(f"  验证: 图片内容正确")
                    else:
                        print(f"  验证: 图片内容差异较大")
            except Exception as e:
                print(f"  读取: 失败 - {e}")

        # 测试3: 对比原生 OpenCV（预期失败）
        print("\n[测试3] 原生 OpenCV 对比（预期在中文路径下失败）")
        print(f"  路径: {chinese_path}")

        # 原生 cv2.imread
        img_cv2 = cv2.imread(str(chinese_path))
        if img_cv2 is None:
            print(f"  cv2.imread: 失败 [X] (这是预期的，因为中文路径不被支持)")
        else:
            print(f"  cv2.imread: 成功 [OK] (罕见，可能系统配置支持中文路径)")

        # 修复后的工具函数
        try:
            img_fixed = load_image(chinese_path)
            print(f"  load_image: 成功 [OK] (修复生效)")
        except Exception as e:
            print(f"  load_image: 失败 [X] - {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_chinese_path()
