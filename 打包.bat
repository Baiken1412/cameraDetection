@echo off
chcp 65001 >nul
echo ============================================================
echo RTSP监测系统 - 自动打包脚本
echo ============================================================
echo.

REM 清理旧的打包文件
echo [1/6] 清理旧文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist 客户软件包 rmdir /s /q 客户软件包
if exist "RTSP监测系统.spec" del "RTSP监测系统.spec"
echo ✓ 清理完成

REM 打包主程序
echo.
echo [2/6] 打包主程序...
pyinstaller --onefile --name "RTSP监测系统" main.py
if errorlevel 1 (
    echo ✗ 打包失败！
    pause
    exit /b 1
)
echo ✓ 打包完成

REM 创建发布目录
echo.
echo [3/6] 创建发布目录...
mkdir 客户软件包
echo ✓ 目录创建完成

REM 复制文件
echo.
echo [4/6] 复制必要文件...

REM 复制 exe
copy dist\RTSP监测系统.exe 客户软件包\
echo   - RTSP监测系统.exe

REM 复制配置文件
copy config.py 客户软件包\
echo   - config.py

REM 复制公钥（如果存在）
if exist public_key.pem (
    copy public_key.pem 客户软件包\
    echo   - public_key.pem
) else (
    echo   ! public_key.pem 不存在（需要先生成密钥对）
)

REM 复制模型目录
if exist models (
    xcopy models 客户软件包\models\ /E /I /Q
    echo   - models/
) else (
    echo   ! models 目录不存在
)

echo ✓ 文件复制完成

REM 创建数据目录
echo.
echo [5/6] 创建数据目录...
mkdir 客户软件包\logs
mkdir 客户软件包\data
mkdir 客户软件包\data\cropped_persons
mkdir 客户软件包\data\features
mkdir 客户软件包\data\input_images
mkdir 客户软件包\data\results
echo ✓ 目录创建完成

REM 创建使用说明
echo.
echo [6/6] 创建使用说明...
(
echo RTSP监测系统 - 使用说明
echo.
echo 1. 配置数据库
echo    编辑 config.py，修改数据库连接信息：
echo    - host: 数据库地址
echo    - user: 数据库用户名
echo    - password: 数据库密码
echo.
echo 2. 安装许可证
echo    将 license.dat 和 public_key.pem 放到此目录
echo.
echo 3. 运行程序
echo    双击 RTSP监测系统.exe
echo.
echo 4. 图片保存路径
echo    默认：D:/ruoyi/uploadPath/caseapp
echo    如需修改，请编辑 config.py 中的 image_save_path
echo.
echo 如有问题，请联系技术支持。
) > 客户软件包\使用说明.txt
echo ✓ 使用说明创建完成

REM 显示目录结构
echo.
echo ============================================================
echo 打包完成！
echo ============================================================
echo.
echo 输出目录：客户软件包\
echo.
echo 目录结构：
tree /F 客户软件包
echo.
echo ============================================================
echo 下一步：
echo 1. 为客户生成许可证：
echo    python generate_license.py --create --machine-code ^<客户机器码^> --days 365
echo.
echo 2. 将 license_xxxxxxxx.dat 改名为 license.dat，复制到客户软件包\
echo.
echo 3. 根据客户环境修改 客户软件包\config.py
echo.
echo 4. 打包成 zip 发送给客户
echo ============================================================
pause
