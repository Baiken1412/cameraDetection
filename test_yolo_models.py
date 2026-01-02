"""
YOLOv8x vs YOLOv8m 性能对比测试
测试速度、精度、内存占用
"""
import time
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import psutil
import numpy as np


def get_memory_usage():
    """获取当前进程内存占用 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def download_model_if_needed(model_name):
    """如果模型不存在，自动下载"""
    model_path = Path(model_name)

    if not model_path.exists():
        print(f"⚠️  模型文件不存在: {model_name}")
        print(f"📥 正在下载 {model_name}...")
        try:
            # YOLO会自动下载模型
            model = YOLO(model_name)
            print(f"✅ {model_name} 下载完成")
            return model
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    else:
        print(f"✅ 找到模型文件: {model_name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return YOLO(model_name)


def test_model_on_images(model, image_paths, model_name, warmup=3):
    """测试模型在多张图片上的性能"""
    print(f"\n{'='*60}")
    print(f"🧪 测试模型: {model_name}")
    print(f"{'='*60}")

    # 获取设备信息
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  设备: {device.upper()}")
    if device == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 记录初始内存
    mem_before = get_memory_usage()
    print(f"📊 初始内存: {mem_before:.1f} MB")

    # 预热（让模型加载到GPU/CPU缓存）
    if image_paths:
        print(f"\n🔥 预热 {warmup} 次...")
        for i in range(warmup):
            _ = model.predict(image_paths[0], conf=0.5, verbose=False, device=device)

    # 测试推理时间
    print(f"\n⏱️  开始性能测试...")
    inference_times = []
    detection_results = []

    for idx, img_path in enumerate(image_paths, 1):
        if not Path(img_path).exists():
            print(f"  ⚠️  图片不存在，跳过: {img_path}")
            continue

        # 计时
        start_time = time.time()
        results = model.predict(
            img_path,
            conf=0.5,
            iou=0.4,
            classes=[0],  # 只检测person
            device=device,
            verbose=False
        )
        inference_time = (time.time() - start_time) * 1000  # 转为毫秒

        # 统计检测结果
        num_people = len(results[0].boxes) if results and results[0].boxes is not None else 0
        inference_times.append(inference_time)
        detection_results.append(num_people)

        print(f"  [{idx}/{len(image_paths)}] {Path(img_path).name[:40]:40s} | "
              f"用时: {inference_time:6.1f}ms | 检测到: {num_people} 人")

    # 统计结果
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    if inference_times:
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        total_people = sum(detection_results)

        print(f"\n📈 性能统计:")
        print(f"  ├─ 平均推理时间: {avg_time:.1f} ms")
        print(f"  ├─ 最快推理时间: {min_time:.1f} ms")
        print(f"  ├─ 最慢推理时间: {max_time:.1f} ms")
        print(f"  ├─ 总检测人数: {total_people} 人")
        print(f"  ├─ 内存占用增加: {mem_used:.1f} MB")
        print(f"  └─ 模型加载后内存: {mem_after:.1f} MB")

        return {
            'model_name': model_name,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_people': total_people,
            'memory_used': mem_used,
            'inference_times': inference_times,
            'detection_results': detection_results
        }
    else:
        print("  ⚠️  没有有效的测试图片")
        return None


def compare_models(yolov8x_stats, yolov8m_stats):
    """对比两个模型的性能"""
    print(f"\n{'='*60}")
    print(f"📊 YOLOv8x vs YOLOv8m 对比")
    print(f"{'='*60}")

    if not yolov8x_stats or not yolov8m_stats:
        print("⚠️  缺少测试数据，无法对比")
        return

    # 计算提升百分比
    speed_improvement = ((yolov8x_stats['avg_time'] - yolov8m_stats['avg_time']) /
                         yolov8x_stats['avg_time'] * 100)
    memory_reduction = ((yolov8x_stats['memory_used'] - yolov8m_stats['memory_used']) /
                        yolov8x_stats['memory_used'] * 100)

    print(f"\n⏱️  推理速度:")
    print(f"  YOLOv8x: {yolov8x_stats['avg_time']:.1f} ms")
    print(f"  YOLOv8m: {yolov8m_stats['avg_time']:.1f} ms")
    print(f"  ⚡ 提升: {speed_improvement:+.1f}% {'(更快 ✅)' if speed_improvement > 0 else '(更慢 ❌)'}")
    print(f"  ⚡ 加速倍数: {yolov8x_stats['avg_time'] / yolov8m_stats['avg_time']:.2f}x")

    print(f"\n💾 内存占用:")
    print(f"  YOLOv8x: {yolov8x_stats['memory_used']:.1f} MB")
    print(f"  YOLOv8m: {yolov8m_stats['memory_used']:.1f} MB")
    print(f"  📉 减少: {memory_reduction:+.1f}% {'(更少 ✅)' if memory_reduction > 0 else '(更多 ❌)'}")

    print(f"\n🎯 检测结果:")
    print(f"  YOLOv8x: 共检测到 {yolov8x_stats['total_people']} 人")
    print(f"  YOLOv8m: 共检测到 {yolov8m_stats['total_people']} 人")
    people_diff = yolov8m_stats['total_people'] - yolov8x_stats['total_people']
    if people_diff == 0:
        print(f"  ✅ 检测结果一致")
    else:
        print(f"  ⚠️  差异: {people_diff:+d} 人 ({abs(people_diff/max(yolov8x_stats['total_people'], 1)*100):.1f}%)")

    # 建议
    print(f"\n💡 建议:")
    if speed_improvement > 50 and abs(people_diff) <= 1:
        print(f"  ✅ 强烈推荐切换到 YOLOv8m")
        print(f"     - 速度提升 {speed_improvement:.0f}%，检测结果几乎一致")
    elif speed_improvement > 30:
        print(f"  ✅ 推荐切换到 YOLOv8m")
        print(f"     - 速度提升明显，适合实时监控")
    elif speed_improvement > 0:
        print(f"  ⚖️  可以考虑切换到 YOLOv8m")
        print(f"     - 速度稍快，根据精度需求决定")
    else:
        print(f"  🤔 建议保持 YOLOv8x")
        print(f"     - YOLOv8m 没有明显速度优势")


def find_test_images(base_path, max_images=10):
    """查找测试图片"""
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"⚠️  路径不存在: {base_path}")
        return []

    # 查找最新的日期目录
    date_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
                       reverse=True)

    images = []
    for date_dir in date_dirs[:3]:  # 最多查看最近3天
        jpg_files = list(date_dir.glob("*.jpg"))
        images.extend(jpg_files)
        if len(images) >= max_images:
            break

    return images[:max_images]


def main():
    print("=" * 60)
    print("🚀 YOLOv8x vs YOLOv8m 性能对比测试")
    print("=" * 60)

    # 1. 查找测试图片
    image_save_path = Path("D:/ruoyi/uploadPath/caseapp")
    print(f"\n📁 查找测试图片: {image_save_path}")

    test_images = find_test_images(image_save_path, max_images=10)

    if not test_images:
        print(f"\n⚠️  没有找到测试图片！")
        print(f"请确保路径正确，或手动指定图片路径")

        # 提供手动输入选项
        manual_path = input("\n是否手动指定图片路径？(留空跳过): ").strip()
        if manual_path and Path(manual_path).exists():
            if Path(manual_path).is_dir():
                test_images = list(Path(manual_path).glob("*.jpg"))[:10]
            else:
                test_images = [Path(manual_path)]

        if not test_images:
            print("❌ 无法继续测试，退出")
            return

    print(f"✅ 找到 {len(test_images)} 张测试图片")
    for i, img in enumerate(test_images[:5], 1):
        print(f"  {i}. {img.name}")
    if len(test_images) > 5:
        print(f"  ... 还有 {len(test_images) - 5} 张")

    # 2. 加载模型
    print(f"\n📦 加载模型...")

    yolov8x_model = download_model_if_needed("yolov8x.pt")
    yolov8m_model = download_model_if_needed("yolov8m.pt")

    if not yolov8x_model or not yolov8m_model:
        print("\n❌ 模型加载失败，请检查网络连接或手动下载模型")
        print("\n手动下载地址:")
        print("  YOLOv8m: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt")
        print("  下载后放到当前目录: D:\\work\\spbj\\spbj\\yolov8m.pt")
        return

    # 3. 测试 YOLOv8x
    test_images_str = [str(img) for img in test_images]
    yolov8x_stats = test_model_on_images(yolov8x_model, test_images_str, "YOLOv8x")

    # 清理内存
    del yolov8x_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)

    # 4. 测试 YOLOv8m
    yolov8m_stats = test_model_on_images(yolov8m_model, test_images_str, "YOLOv8m")

    # 5. 对比结果
    compare_models(yolov8x_stats, yolov8m_stats)

    print(f"\n{'='*60}")
    print(f"✅ 测试完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
