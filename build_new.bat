@echo off
chcp 65001 >nul
echo ========================================
echo  摄像头监测系统 - 打包脚本 (新版)
echo ========================================

cd /d "%~dp0"

echo [1/3] 清理旧的打包输出...
if exist dist_new rmdir /s /q dist_new
if exist build_tmp rmdir /s /q build_tmp

echo [2/3] 开始 PyInstaller 打包...
pyinstaller build_new.spec --distpath dist_new --workpath build_tmp --noconfirm

if %errorlevel% neq 0 (
    echo.
    echo [错误] 打包失败！请检查错误信息。
    pause
    exit /b 1
)

echo [3/3] 复制 config.json 到输出目录...
copy /y config.json dist_new\config.json

echo.
echo ========================================
echo  打包完成！
echo  输出目录: dist_new\
echo    - camera_monitor_system.exe  （主程序）
echo    - config.json                （配置文件，可直接修改）
echo ========================================
pause
