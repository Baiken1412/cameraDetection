@echo off
chcp 65001 >nul
echo 启动 ReID Web 应用...
cd /d "%~dp0"
python reid_web_app.py
pause

