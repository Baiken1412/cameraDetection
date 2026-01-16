@echo off
chcp 65001 >nul

echo ================================
echo 打包 RTSP 视频流监测系统（背景建模版）
echo ================================
echo.

cd /d %~dp0

echo [1/3] 检查虚拟环境...

if not exist .venv\Scripts\python.exe (
    echo 错误：未找到虚拟环境 .venv
    echo 请先执行：python -m venv .venv
    pause
    exit /b 1
)

.venv\Scripts\python.exe --version
if errorlevel 1 (
    echo 错误：虚拟环境 Python 无法运行
    pause
    exit /b 1
)

echo.
echo [2/3] 安装依赖...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo [3/3] 使用 PyInstaller 打包...
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller camera_monitor_system_onefile.spec --clean

echo.
echo 打包完成！
echo 生成的 exe 位于：
echo dist\CameraMonitorSystem.exe
echo.

pause
