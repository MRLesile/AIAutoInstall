#!/bin/bash

# 确保脚本在错误时停止
set -e

echo "开始部署数字化教材系统..."

# 创建必要的目录
sudo mkdir -p /var/log/digital-textbook
sudo chown ubuntu:ubuntu /var/log/digital-textbook

# 安装系统依赖
echo "安装系统依赖..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 创建并激活虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装Python依赖
echo "安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 复制systemd服务文件
echo "配置systemd服务..."
sudo cp deploy/systemd/digital-textbook.service /etc/systemd/system/

# 重新加载systemd配置
sudo systemctl daemon-reload

# 配置日志轮转
sudo tee /etc/logrotate.d/digital-textbook << EOF
/var/log/digital-textbook/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 ubuntu ubuntu
}
EOF

# 启用并启动服务
echo "启动服务..."
sudo systemctl enable digital-textbook
sudo systemctl start digital-textbook

echo "部署完成！"
echo "可以使用以下命令查看服务状态："
echo "sudo systemctl status digital-textbook"
echo "查看日志："
echo "sudo journalctl -u digital-textbook -f"