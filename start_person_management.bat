@echo off
chcp 65001 >nul
echo ========================================
echo 启动人员管理服务
echo ========================================
echo.

cd /d %~dp0

echo 正在启动人员管理API服务...
echo 服务地址: http://localhost:5001
echo Web界面: http://localhost:5001/person/manage
echo.

python person_management_api.py

pause

