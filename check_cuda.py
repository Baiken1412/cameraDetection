"""
CUDA环境检测脚本
检查CUDA是否正确安装，以及PyTorch、YOLO、ReID模型是否能使用GPU
"""
import sys
from pathlib import Path

def check_cuda_environment():
    """检查CUDA环境"""
    print("=" * 70)
    print("CUDA环境检测")
    print("=" * 70)

    # 1. 检查PyTorch和CUDA
    print("\n[1/5] 检查PyTorch和CUDA...")
    try:
        import torch
        print(f"[OK] PyTorch版本: {torch.__version__}")
        print(f"[OK] CUDA是否可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"[OK] CUDA版本: {torch.version.cuda}")
            print(f"[OK] GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                # 显示显存信息
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    总显存: {total_memory:.2f} GB")

            # 测试GPU计算
            print("\n测试GPU计算...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("[OK] GPU计算测试成功")

        else:
            print("[FAIL] CUDA不可用，将使用CPU运行")
            print("\n可能的原因:")
            print("  1. NVIDIA驱动未安装")
            print("  2. CUDA Toolkit未安装")
            print("  3. PyTorch安装的是CPU版本")
            print("\n解决方法:")
            print("  - 安装NVIDIA驱动: https://www.nvidia.com/download/index.aspx")
            print("  - 重新安装PyTorch GPU版本: https://pytorch.org/get-started/locally/")

    except ImportError as e:
        print(f"[FAIL] PyTorch未安装: {e}")
        return False

    # 2. 检查当前项目配置
    print("\n[2/5] 检查项目配置...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import config

        device_config = config.REID_CONFIG.get('device', 'cpu')
        print(f"配置文件中的设备: {device_config}")

        if device_config == 'cuda' and not torch.cuda.is_available():
            print("[WARN] 警告: 配置为CUDA但CUDA不可用，将自动切换到CPU")
        elif device_config == 'cpu' and torch.cuda.is_available():
            print("[WARN] 提示: CUDA可用但配置为CPU，建议修改config.py中的device为'cuda'")
        else:
            print("[OK] 设备配置正确")

    except Exception as e:
        print(f"[FAIL] 读取配置失败: {e}")

    # 3. 检查YOLO模型
    print("\n[3/5] 检查YOLO模型...")
    try:
        from ultralytics import YOLO
        print(f"[OK] Ultralytics YOLO已安装")

        # 尝试加载模型
        try:
            import reid_config_adapter
            config_obj = reid_config_adapter.Config
            model_path = config_obj.YOLO_MODEL_PATH
            model_name = config_obj.YOLO_MODEL_NAME

            print(f"模型名称: {model_name}")
            print(f"模型路径: {model_path}")

            if Path(model_path).exists():
                print("[OK] YOLO模型文件存在")
                # 加载模型测试
                print("加载YOLO模型...")
                model = YOLO(model_path)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"[OK] YOLO模型加载成功，将使用设备: {device}")
            else:
                print(f"[FAIL] YOLO模型文件不存在: {model_path}")
                print(f"请下载 {model_name} 并放置到 models/ 目录")

        except Exception as e:
            print(f"[WARN]  YOLO模型检查异常: {e}")

    except ImportError:
        print("[FAIL] Ultralytics未安装，请运行: pip install ultralytics")

    # 4. 检查ReID模型
    print("\n[4/5] 检查ReID模型...")
    try:
        import torchreid
        print(f"[OK] torchreid已安装")

        try:
            import reid_config_adapter
            config_obj = reid_config_adapter.Config
            model_name = config_obj.REID_MODEL_NAME

            print(f"ReID模型: {model_name}")
            print("加载ReID模型...")

            model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                loss='softmax',
                pretrained=True
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            print(f"[OK] ReID模型加载成功，运行在: {device}")

        except Exception as e:
            print(f"[WARN]  ReID模型检查异常: {e}")

    except ImportError:
        print("[FAIL] torchreid未安装，请运行: pip install torchreid")

    # 5. 性能建议
    print("\n[5/5] 性能建议")
    print("-" * 70)

    if torch.cuda.is_available():
        print("[OK] GPU可用，性能建议:")
        print("  1. 确保config.py中 REID_CONFIG['device'] = 'cuda'")
        print("  2. 可以增大批处理大小以提高吞吐量")
        print("  3. 可以使用更大的模型获得更好精度")

        # 显存建议
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory < 4:
            print(f"  [WARN]  显存较小({total_memory:.1f}GB)，建议:")
            print("     - 使用轻量模型 (yolov8n, osnet_x0_5)")
            print("     - 减小批处理大小")
        elif total_memory < 8:
            print(f"  显存适中({total_memory:.1f}GB)，建议:")
            print("     - 使用中等模型 (yolov8s/m, osnet_x0_75)")
        else:
            print(f"  显存充足({total_memory:.1f}GB)，可以:")
            print("     - 使用大模型 (yolov8m/l, osnet_x1_0)")
            print("     - 增大批处理大小")
    else:
        print("[WARN]  CPU模式，性能建议:")
        print("  1. 使用最轻量的模型:")
        print("     - YOLO: yolov8n.pt")
        print("     - ReID: osnet_x0_5 或 osnet_x0_25")
        print("  2. 减小批处理大小 (4-8)")
        print("  3. 增加检测间隔 (2-3秒)")
        print("  4. 考虑安装CUDA以大幅提升性能")

    print("\n" + "=" * 70)
    print("检测完成")
    print("=" * 70)

    return torch.cuda.is_available()

if __name__ == "__main__":
    cuda_available = check_cuda_environment()

    if not cuda_available:
        print("\n[INFO] 如何安装CUDA:")
        print("1. 检查GPU: 运行 nvidia-smi 查看GPU型号")
        print("2. 安装NVIDIA驱动: https://www.nvidia.com/download/index.aspx")
        print("3. 安装CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("4. 重新安装PyTorch (GPU版本):")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    else:
        print("\n[OK] CUDA环境正常，可以使用GPU加速！")
        sys.exit(0)
