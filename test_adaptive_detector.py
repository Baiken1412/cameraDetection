"""
测试自适应人员检测器
验证 CPU 自动检测和推理引擎选择
"""
import logging
import time
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_cpu_detection():
    """测试 CPU 自动检测"""
    print("=" * 70)
    print("测试 1: CPU 自动检测")
    print("=" * 70)

    from utils.cpu_detector import CPUDetector

    detector = CPUDetector()
    cpu_model, engine, threads = detector.detect()

    print(f"\n检测结果:")
    print(f"  CPU 型号: {cpu_model}")
    print(f"  推荐引擎: {engine.value}")
    print(f"  推荐线程: {threads}")

    # 检查引擎可用性
    print(f"\n引擎可用性:")
    from utils.cpu_detector import InferenceEngine
    openvino_ok = CPUDetector.check_engine_availability(InferenceEngine.OPENVINO)
    onnx_ok = CPUDetector.check_engine_availability(InferenceEngine.ONNX_RUNTIME)
    print(f"  OpenVINO: {'可用' if openvino_ok else '不可用'}")
    print(f"  ONNX Runtime: {'可用' if onnx_ok else '不可用'}")

    return engine, onnx_ok or openvino_ok


def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 70)
    print("测试 2: 加载自适应检测器")
    print("=" * 70)

    try:
        from core.person_detector_adaptive import AdaptivePersonDetector

        print("\n初始化检测器...")
        detector = AdaptivePersonDetector(
            model_dir='models',
            conf_threshold=0.5,
            iou_threshold=0.4
        )

        print(f"\n[OK] 检测器初始化成功")
        print(f"  使用引擎: {detector.engine.value}")
        print(f"  线程数: {detector.num_threads}")
        print(f"  输入尺寸: {detector.input_width}x{detector.input_height}")

        return detector

    except Exception as e:
        print(f"\n[FAIL] 检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(detector):
    """测试推理性能"""
    print("\n" + "=" * 70)
    print("测试 3: 推理性能测试")
    print("=" * 70)

    # 创建测试图像（随机图像）
    print("\n生成测试图像...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 预热（第一次推理可能较慢）
    print("预热推理引擎...")
    _ = detector.detect_image(test_image)

    # 性能测试
    print("\n开始性能测试（10 次推理）...")
    times = []
    for i in range(10):
        start = time.time()
        detections = detector.detect_image(test_image)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  第 {i+1} 次: {elapsed:.3f} 秒, 检测到 {len(detections)} 人")

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    print(f"\n性能统计:")
    print(f"  平均耗时: {avg_time:.3f} 秒")
    print(f"  FPS: {fps:.1f}")
    print(f"  最快: {min(times):.3f} 秒")
    print(f"  最慢: {max(times):.3f} 秒")

    # 性能评估
    if avg_time < 0.5:
        print(f"\n[优秀] 推理速度非常快！")
    elif avg_time < 1.0:
        print(f"\n[良好] 推理速度符合预期")
    elif avg_time < 2.0:
        print(f"\n[一般] 推理速度可接受")
    else:
        print(f"\n[慢] 推理速度较慢，建议:")
        print(f"  1. 检查是否使用了正确的模型（yolo11n.onnx）")
        print(f"  2. 增加线程数")
        print(f"  3. 降低输入分辨率")


def test_real_image(detector):
    """测试真实图像（如果有的话）"""
    print("\n" + "=" * 70)
    print("测试 4: 真实图像检测（可选）")
    print("=" * 70)

    # 查找测试图像
    test_images = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))

    if not test_images:
        print("\n[跳过] 未找到测试图像（.jpg 或 .png）")
        return

    test_image_path = test_images[0]
    print(f"\n使用测试图像: {test_image_path}")

    # 读取图像
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"[错误] 无法读取图像")
        return

    # 检测
    print("执行检测...")
    start = time.time()
    detections = detector.detect_image(image)
    elapsed = time.time() - start

    print(f"\n检测结果:")
    print(f"  耗时: {elapsed:.3f} 秒")
    print(f"  检测到 {len(detections)} 个人员")

    if len(detections) > 0:
        print(f"\n详细结果:")
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            conf = det['conf']
            print(f"  人员 {i}: 置信度={conf:.2f}, 位置={[int(x) for x in bbox]}")

        # 可视化
        vis_image = detector.visualize(image, detections)
        output_path = 'test_result.jpg'
        cv2.imwrite(output_path, vis_image)
        print(f"\n[OK] 可视化结果已保存: {output_path}")


def main():
    """主测试流程"""
    print("\n")
    print("=" * 70)
    print("自适应人员检测器 - 完整测试")
    print("=" * 70)

    # 测试 1: CPU 检测
    engine, engine_ok = test_cpu_detection()

    if not engine_ok:
        print("\n[错误] 推理引擎不可用，请安装:")
        if engine.value == 'openvino':
            print("  pip install openvino")
        else:
            print("  pip install onnxruntime")
        return

    # 测试 2: 模型加载
    detector = test_model_loading()

    if detector is None:
        print("\n[错误] 模型加载失败，请检查:")
        print("  1. models/ 目录下是否有 yolo11n.onnx 或 yolo11n.xml")
        print("  2. 运行 python convert_yolo11_models.py 转换模型")
        return

    # 测试 3: 推理性能
    test_inference(detector)

    # 测试 4: 真实图像（可选）
    test_real_image(detector)

    # 总结
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print(f"\n系统配置:")
    print(f"  推理引擎: {detector.engine.value}")
    print(f"  模型: YOLOv11n")
    print(f"  线程数: {detector.num_threads}")
    print(f"\n部署建议:")
    print(f"  1. 确保服务器安装了推理引擎: {detector.engine.value}")
    print(f"  2. 上传模型文件到 models/ 目录")
    print(f"  3. config.py 中 ADAPTIVE_DETECTION_CONFIG['enabled'] = True")
    print(f"  4. 运行 python main.py 启动系统")


if __name__ == "__main__":
    main()
