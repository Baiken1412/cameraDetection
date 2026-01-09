"""
缓冲区清理机制诊断脚本
验证定期清理是否正常工作
"""
import cv2
import time
from datetime import datetime

def test_flush_mechanism():
    """测试缓冲区清理机制"""
    print("=== 缓冲区清理机制诊断 ===\n")

    # 打开USB摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开USB摄像头")
        return

    print("✓ USB摄像头已打开")
    print("测试时长: 30秒")
    print("清理间隔: 5秒")
    print("-" * 60)

    start_time = time.time()
    last_flush_time = start_time
    flush_interval = 5  # 5秒清理一次

    while time.time() - start_time < 30:
        current_time = time.time()

        # 定期清理测试
        if current_time - last_flush_time >= flush_interval:
            # 执行清理
            flushed_count = 0
            for _ in range(100):  # 最多清理100帧
                ret = cap.grab()
                if not ret:
                    break  # 缓冲区已空
                flushed_count += 1

            elapsed = current_time - start_time
            print(
                f"[{elapsed:5.1f}s] 定期清理执行 - "
                f"清理了 {flushed_count} 帧 "
                f"({'有积压' if flushed_count > 0 else '无积压'})"
            )

            last_flush_time = current_time

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取失败")
            break

        # 短暂休眠
        time.sleep(0.01)

    cap.release()
    print("-" * 60)
    print("\n✓ 测试完成")
    print("\n如果看到了定期清理日志，说明机制正常工作")
    print("如果没看到，说明代码可能有问题")

if __name__ == "__main__":
    test_flush_mechanism()
