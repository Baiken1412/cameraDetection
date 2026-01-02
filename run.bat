@echo off
chcp 65001 >nul
echo ========================================
echo RTSP视频流监测系统 - 背景建模版本
echo ========================================
echo.

cd /d %~dp0

echo 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo.
echo 启动监测系统...
echo 注意: 每个摄像头需要约20秒进行背景建模学习
echo.

python main.py

pause

