#!/bin/bash

echo "========================================"
echo "RTSP视频流监测系统 - 背景建模版本"
echo "========================================"
echo ""

cd "$(dirname "$0")"

echo "检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python，请先安装Python 3.7+"
    exit 1
fi

echo ""
echo "启动监测系统..."
echo "注意: 每个摄像头需要约20秒进行背景建模学习"
echo ""

python3 main.py

