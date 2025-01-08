# 数字化教材系统部署指南

本文档描述了如何在 Ubuntu 服务器上部署数字化教材系统。

## 系统要求

- Ubuntu Server (22.04.5 LTS)
- Python 3.10.12
- 足够的磁盘空间（建议至少 10GB）
- 内存建议 8GB 以上（因为要加载模型）
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3070 Ti     Off | 00000000:01:00.0 Off |                  N/A |
|  0%   40C    P8              18W / 290W |    608MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   3192887      C   python3                                     602MiB |
+---------------------------------------------------------------------------------------+

## 项目结构
pythonProject/
├── app/
│ ├── init.py # 应用工厂和蓝图注册
│ ├── config.py # 配置常量
│ ├── extensions.py # 共享扩展实例
│ ├── Model/ # 模型文件目录
│ │ └── text2vec-base-chinese/
│ ├── routes/ # 路由模块
│ │ ├── init.py
│ │ ├── search.py # 搜索相关路由
│ │ ├── rag.py # RAG搜索路由
│ │ ├── chat.py # AI对话路由
│ │ └── upload.py # 上传相关路由
│ ├── services/ # 服务层
│ │ ├── init.py
│ │ ├── chroma_service.py # ChromaDB相关操作
│ │ ├── ai_service.py # AI对话服务
│ │ └── upload_service.py # 上传服务
│ └── utils/ # 工具类
│ ├── init.py
│ ├── auth.py # 认证相关
│ ├── decorators.py # 自定义装饰器
│ └── helpers.py # 辅助函数
├── deploy/ # 部署相关文件
│ ├── setup.sh # 部署脚本
│ └── systemd/ # systemd服务配置
│ └── digital-textbook.service
├── chroma_db/ # ChromaDB数据目录
├── run.py # 启动脚本
└── requirements.txt # 项目依赖

## 详细部署步骤

### 1. 准备工作

1.1 更新系统并安装基础工具：

bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git python3-pip python3-venv

# 1.2 克隆项目：

bash
cd /home/ubuntu
git clone <repository_url> pythonProject
cd pythonProject

# 1.3 下载模型文件：
下载 text2vec-base-chinese 模型

### 2. 安装依赖

2.1 创建并激活虚拟环境：
bash
python3 -m venv venv
source venv/bin/activate

2.2 安装依赖包：
bash
pip install --upgrade pip
pip install -r requirements.txt

依赖列表（requirements.txt）：

### 3. 配置服务

3.1 创建必要的目录：
bash
sudo mkdir -p /var/log/digital-textbook
sudo chown ubuntu:ubuntu /var/log/digital-textbook
mkdir -p chroma_db

重启服务
sudo systemctl restart digital-textbook

查看日志
sudo journalctl -u digital-textbook -f

### B. 配置文件位置

- 服务配置：`/etc/systemd/system/digital-textbook.service`
- 日志配置：`/etc/logrotate.d/digital-textbook`
- 应用配置：`app/config.py`