"""
VideoStreamReader 测试脚本
验证独立线程读取方案是否正常工作
"""
import cv2
import time
from datetime import datetime
from video_stream_reader import VideoStreamReader

def test_video_stream_reader():
    """测试 VideoStreamReader"""
    print("=" * 60)
    print("VideoStreamReader 功能测试")
    print("=" * 60)
    print()

    # 打开USB摄像头
    print("正在打开USB摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开USB摄像头")
        return

    print("✓ USB摄像头已打开")
    print()

    # 创建 VideoStreamReader
    print("正在创建 VideoStreamReader...")
    stream_reader = VideoStreamReader(cap, camera_name="测试摄像头")
    print()

    # 测试读取（模拟主线程卡顿）
    print("开始测试（30秒）...")
    print("策略：每5秒读取一次，模拟主线程卡顿")
    print("-" * 60)

    start_time = time.time()
    read_count = 0

    while time.time() - start_time < 30:
        # 读取帧
        ret, frame, frame_time = stream_reader.read()

        if ret:
            read_count += 1
            elapsed = time.time() - start_time
            time_diff = (datetime.now() - frame_time).total_seconds()

            print(
                f"[{elapsed:5.1f}s] 读取第 {read_count} 帧 - "
                f"帧时间戳: {frame_time.strftime('%H:%M:%S.%f')[:-3]}, "
                f"延迟: {time_diff*1000:.1f}ms"
            )
        else:
            print("❌ 读取失败")

        # 模拟主线程卡顿（5秒）
        time.sleep(5)

    # 释放资源
    stream_reader.release()
    print("-" * 60)
    print()
    print("✓ 测试完成")
    print()
    print("结论：")
    print("  - 如果延迟一直很小（<50ms），说明 VideoStreamReader 正常工作")
    print("  - 独立线程持续读取最新帧，主线程卡顿不影响实时性")

if __name__ == "__main__":
    test_video_stream_reader()
