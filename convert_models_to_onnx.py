"""
模型转换脚本：将 PyTorch 模型转换为 ONNX 格式
ONNX 在 CPU 上推理速度比 PyTorch 快 5-10 倍
"""
import sys
from pathlib import Path
import torch
import onnx

def convert_yolo_to_onnx():
    """转换 YOLO 模型为 ONNX 格式"""
    print("\n" + "=" * 70)
    print("转换 YOLO 模型为 ONNX 格式")
    print("=" * 70)

    try:
        from ultralytics import YOLO

        # 模型路径
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        # 推荐使用最轻量的模型（CPU 上速度最快）
        yolo_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

        for model_name in yolo_models:
            model_path = models_dir / model_name
            onnx_path = models_dir / model_name.replace('.pt', '.onnx')

            if not model_path.exists():
                print(f"[跳过] {model_name} 不存在")
                continue

            if onnx_path.exists():
                print(f"[OK] {onnx_path.name} 已存在，跳过转换")
                continue

            print(f"\n转换 {model_name}...")
            try:
                # 加载模型
                model = YOLO(str(model_path))

                # 导出为 ONNX（优化配置）
                model.export(
                    format='onnx',
                    imgsz=640,  # 输入尺寸
                    dynamic=False,  # 固定尺寸（更快）
                    simplify=True,  # 简化模型
                    opset=12  # ONNX opset 版本
                )

                print(f"[OK] 转换成功: {onnx_path}")

                # 验证 ONNX 模型
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print(f"[OK] ONNX 模型验证通过")

            except Exception as e:
                print(f"[FAIL] 转换失败: {e}")

    except ImportError:
        print("[FAIL] ultralytics 未安装，请运行: pip install ultralytics")
        return False

    return True


def convert_reid_to_onnx():
    """转换 ReID 模型为 ONNX 格式"""
    print("\n" + "=" * 70)
    print("转换 ReID 模型为 ONNX 格式")
    print("=" * 70)

    try:
        import torchreid
        from torchvision import transforms

        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        # 推荐使用轻量模型（CPU 上速度最快）
        reid_models = [
            ('osnet_x0_25', 'osnet_x0_25.onnx'),
            ('osnet_x0_5', 'osnet_x0_5.onnx'),
            ('osnet_x0_75', 'osnet_x0_75.onnx'),
            ('osnet_x1_0', 'osnet_x1_0.onnx')
        ]

        for model_name, onnx_name in reid_models:
            onnx_path = models_dir / onnx_name

            if onnx_path.exists():
                print(f"[OK] {onnx_name} 已存在，跳过转换")
                continue

            print(f"\n转换 {model_name}...")
            try:
                # 加载模型
                model = torchreid.models.build_model(
                    name=model_name,
                    num_classes=1000,
                    loss='softmax',
                    pretrained=True
                )
                model.eval()

                # 创建示例输入（ReID 输入尺寸：256x128）
                dummy_input = torch.randn(1, 3, 256, 128)

                # 导出为 ONNX
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )

                print(f"[OK] 转换成功: {onnx_path}")

                # 验证 ONNX 模型
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print(f"[OK] ONNX 模型验证通过")

            except Exception as e:
                print(f"[FAIL] 转换失败: {e}")
                import traceback
                traceback.print_exc()

    except ImportError as e:
        print(f"[FAIL] torchreid 未安装: {e}")
        return False

    return True


def main():
    """主函数"""
    print("=" * 70)
    print("PyTorch 模型转换为 ONNX 格式")
    print("用于 CPU 推理加速（5-10倍速度提升）")
    print("=" * 70)

    # 转换 YOLO 模型
    yolo_ok = convert_yolo_to_onnx()

    # 转换 ReID 模型
    reid_ok = convert_reid_to_onnx()

    print("\n" + "=" * 70)
    if yolo_ok and reid_ok:
        print("模型转换完成！")
        print("\n下一步：")
        print("1. 安装 ONNX Runtime: pip install onnxruntime")
        print("2. 使用 ONNX 模型进行推理（参考 core/detector_onnx.py）")
        print("3. 修改配置文件使用 ONNX 推理引擎")
    else:
        print("部分模型转换失败，请检查错误信息")
    print("=" * 70)


if __name__ == "__main__":
    main()
