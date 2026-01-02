"""
检查CPU类型，推荐优化方案
"""
import platform
import subprocess

def check_cpu():
    print("=" * 70)
    print("CPU信息检测")
    print("=" * 70)

    system = platform.system()
    print(f"\n操作系统: {system}")
    print(f"架构: {platform.machine()}")

    if system == "Windows":
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True,
                text=True,
                encoding='gbk'
            )
            cpu_name = result.stdout.strip().split('\n')[1].strip()
            print(f"CPU型号: {cpu_name}")

            # 判断是否是Intel
            if 'Intel' in cpu_name:
                print("\n[推荐] 使用 OpenVINO（Intel专用优化）")
                print("预计速度提升: 5-10倍")
            elif 'AMD' in cpu_name:
                print("\n[推荐] 使用 ONNX Runtime")
                print("预计速度提升: 3-5倍")
            else:
                print("\n[推荐] 使用 ONNX Runtime（通用方案）")
        except Exception as e:
            print(f"无法获取CPU信息: {e}")

    elif system == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        print(f"CPU型号: {cpu_name}")

                        if 'Intel' in cpu_name:
                            print("\n[推荐] 使用 OpenVINO（Intel专用优化）")
                            print("预计速度提升: 5-10倍")
                        elif 'AMD' in cpu_name:
                            print("\n[推荐] 使用 ONNX Runtime")
                            print("预计速度提升: 3-5倍")
                        break
        except Exception as e:
            print(f"无法读取CPU信息: {e}")

    # 检查线程数
    import os
    cpu_count = os.cpu_count()
    print(f"\nCPU核心数: {cpu_count}")
    print(f"推荐线程数: {max(1, cpu_count - 2)}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_cpu()
