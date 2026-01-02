@echo off
chcp 65001 >nul
echo ========================================
echo 验证 YOLOv8m 模型是否正确使用
echo ========================================
echo.

cd /d %~dp0

echo [1] 检查配置文件...
findstr "yolov8m" reid_config_adapter.py
if errorlevel 1 (
    echo ❌ 配置文件中未找到 yolov8m
) else (
    echo ✅ 配置文件已更新为 yolov8m
)

echo.
echo [2] 检查模型文件...
if exist "yolov8m.pt" (
    echo ✅ yolov8m.pt 已下载
    dir yolov8m.pt | findstr "yolov8m"
) else (
    echo ❌ yolov8m.pt 不存在
)

echo.
echo [3] 检查日志文件...
if exist "logs\rtsp_monitor.log" (
    echo ✅ 找到日志文件
    echo.
    echo 最近的YOLO加载记录:
    findstr /C:"加载YOLO模型" logs\rtsp_monitor.log | tail -n 3
    echo.
    echo 最近的人员检测记录:
    findstr /C:"YOLO人员检测结果" logs\rtsp_monitor.log | tail -n 5
) else (
    echo ⚠️  日志文件不存在 (服务可能未运行)
)

echo.
echo ========================================
echo 验证完成！
echo ========================================
pause
