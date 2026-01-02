"""
YOLOv11 模型转换脚本
支持转换为 ONNX 和 OpenVINO IR 格式
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_yolo11_to_onnx(model_name: str = 'yolo11n.pt'):
    """
    转换 YOLOv11 模型为 ONNX 格式

    Args:
        model_name: 模型名称（yolo11n.pt, yolo11s.pt, yolo11m.pt）
    """
    print("\n" + "=" * 70)
    print(f"转换 YOLOv11 模型为 ONNX 格式: {model_name}")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics 未安装，请运行: pip install ultralytics")
        return False

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / model_name
    onnx_path = models_dir / model_name.replace('.pt', '.onnx')

    # 如果 ONNX 文件已存在
    if onnx_path.exists():
        print(f"[OK] {onnx_path.name} 已存在")
        return True

    # 如果 .pt 文件不存在，下载
    if not model_path.exists():
        print(f"[下载] 正在下载 {model_name}...")
        try:
            # YOLO 会自动下载模型
            model = YOLO(model_name)
            # 保存到 models 目录
            import shutil
            cache_path = Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics' / model_name
            if cache_path.exists():
                shutil.copy2(cache_path, model_path)
                print(f"[OK] 模型已下载到: {model_path}")
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            return False
    else:
        print(f"[OK] 找到模型文件: {model_path}")

    # 加载模型
    print(f"\n加载模型...")
    model = YOLO(str(model_path))

    # 导出为 ONNX
    print(f"导出为 ONNX 格式...")
    try:
        model.export(
            format='onnx',
            imgsz=640,
            dynamic=False,
            simplify=True,
            opset=12
        )
        print(f"[OK] ONNX 模型已保存: {onnx_path}")
        return True

    except Exception as e:
        logger.error(f"转换失败: {e}")
        return False


def convert_yolo11_to_openvino(model_name: str = 'yolo11n.pt'):
    """
    转换 YOLOv11 模型为 OpenVINO IR 格式

    Args:
        model_name: 模型名称
    """
    print("\n" + "=" * 70)
    print(f"转换 YOLOv11 模型为 OpenVINO 格式: {model_name}")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics 未安装，请运行: pip install ultralytics")
        return False

    try:
        import openvino as ov
        logger.info(f"OpenVINO 版本: {ov.__version__}")
    except ImportError:
        logger.error("OpenVINO 未安装，请运行: pip install openvino")
        print("\n提示: OpenVINO 只需要在 Intel CPU 服务器上安装")
        return False

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / model_name
    xml_path = models_dir / model_name.replace('.pt', '.xml')

    # 如果 OpenVINO 文件已存在
    if xml_path.exists():
        print(f"[OK] {xml_path.name} 已存在")
        return True

    # 确保 .pt 文件存在
    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print(f"请先运行 ONNX 转换，会自动下载模型")
        return False

    # 加载模型
    print(f"\n加载模型...")
    model = YOLO(str(model_path))

    # 导出为 OpenVINO
    print(f"导出为 OpenVINO IR 格式...")
    try:
        model.export(
            format='openvino',
            imgsz=640,
            dynamic=False,
            half=False  # FP32 精度
        )
        print(f"[OK] OpenVINO 模型已保存: {xml_path}")
        return True

    except Exception as e:
        logger.error(f"转换失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("YOLOv11 模型转换工具")
    print("支持: ONNX (AMD/通用) 和 OpenVINO (Intel 专用)")
    print("=" * 70)

    # 推荐的模型（从快到慢）
    models = [
        ('yolo11n.pt', '最快，推荐用于 CPU 服务器'),
        ('yolo11s.pt', '快速，精度稍高'),
        ('yolo11m.pt', '中等速度，较高精度（不推荐 CPU）')
    ]

    print("\n可用模型:")
    for i, (name, desc) in enumerate(models, 1):
        print(f"  {i}. {name} - {desc}")

    # 默认转换 nano 版本（最快）
    model_name = 'yolo11n.pt'

    print(f"\n将转换: {model_name}")
    print("（如需其他版本，请修改脚本中的 model_name 变量）")

    # 转换为 ONNX（通用）
    print("\n[1/2] 转换为 ONNX 格式（AMD/通用 CPU）")
    onnx_ok = convert_yolo11_to_onnx(model_name)

    # 转换为 OpenVINO（Intel 专用）
    print("\n[2/2] 转换为 OpenVINO 格式（Intel CPU 专用）")
    openvino_ok = convert_yolo11_to_openvino(model_name)

    # 总结
    print("\n" + "=" * 70)
    print("转换结果:")
    print("=" * 70)
    print(f"  ONNX: {'成功' if onnx_ok else '失败'}")
    print(f"  OpenVINO: {'成功' if openvino_ok else '失败（如果不是 Intel CPU 可忽略）'}")

    if onnx_ok or openvino_ok:
        print("\n下一步:")
        print("1. 修改 config.py 启用自适应检测")
        print("2. 系统会自动检测 CPU 类型并选择最优引擎")
        print("3. Intel CPU → OpenVINO，AMD CPU → ONNX Runtime")

        print("\n模型文件:")
        models_dir = Path('models')
        for f in models_dir.glob('yolo11*'):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")

    print("=" * 70)


if __name__ == "__main__":
    main()
