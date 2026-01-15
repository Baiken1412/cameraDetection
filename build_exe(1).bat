@echo off
chcp 65001 >nul
echo ========================================
echo 打包 RTSP 视频流监测系统（背景建模版）
echo ========================================
echo.

cd /d %~dp0

echo [1/3] 检查 Python 环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+。
    pause
    exit /b 1
)

echo.
echo [2/3] 安装依赖（如已安装可忽略提示）...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [3/3] 安装并运行 PyInstaller 打包...
python -m pip install pyinstaller
python -m PyInstaller camera_monitor_system_onefile.spec --clean

echo.
echo 打包完成！
echo 生成的单文件可执行程序：
echo   dist\CameraMonitorSystem.exe
echo.
pause