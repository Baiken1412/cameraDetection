@echo off
chcp 65001 >nul

echo ============================================
echo 打包 RTSP 视频流监测系统
echo 使用官方 camera_monitor_system_onefile.spec 配置
echo 生成单文件 exe，config.json 与 exe 同级
echo ============================================
echo.

cd /d %~dp0

echo [1/4] 检查 Python 运行环境...

set "PYTHON_EXE="

rem 优先使用本地虚拟环境的 Python（如果存在且可用）
if exist ".venv\Scripts\python.exe" (
    echo 检测到虚拟环境 .venv，尝试使用其中的 Python...
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if errorlevel 1 (
        echo 警告：虚拟环境 Python 无法运行，将改用系统 Python。
    ) else (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    )
)

rem 如果虚拟环境不可用，则回退到系统 Python
if "%PYTHON_EXE%"=="" (
    echo 使用系统 Python（请确保已添加到 PATH）...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo 错误：未找到可用的 Python，请先安装 Python 并配置环境变量。
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

echo.
echo [2/4] 安装依赖...
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install -r requirements.txt
rem 确保与打包 spec 对应的推理依赖已安装
"%PYTHON_EXE%" -m pip install onnxruntime openvino

echo.
echo [3/4] 安装 PyInstaller 并按 spec 打包...
"%PYTHON_EXE%" -m pip install pyinstaller

rem 使用 camera_monitor_system_onefile.spec 作为打包配置
rem 该 spec 已包含 OpenVINO/ONNX/模型/静态资源等依赖
rem 我们仍然保持 config.json 外置，方便后期维护
"%PYTHON_EXE%" -m PyInstaller camera_monitor_system_onefile.spec --clean

if errorlevel 1 (
    echo.
    echo 打包失败，请检查上方错误信息。
    pause
    exit /b 1
)

echo.
echo [4/4] 复制 config.json 到 dist 目录（与 exe 同级）...
if exist config.json (
    copy /Y config.json dist\ >nul
) else (
    echo 警告：当前目录未找到 config.json，请手动复制到 dist\ 目录。
)

echo.
echo 打包完成！
echo 生成的文件：
echo   dist\CameraMonitorSystem.exe
echo   dist\config.json  （可手动修改，便于后期维护）
echo.
pause

