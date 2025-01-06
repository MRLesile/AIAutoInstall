# ChromaDB 应用部署指南

## 项目结构
```
.
├── ChromaDBApp.py      # 主应用程序
├── Model/              # 模型文件夹
├── Dockerfile         # Docker构建文件
├── docker-compose.yml # Docker编排文件
├── requirements.txt   # Python依赖文件
├── .env              # 环境变量配置
└── .dockerignore     # Docker忽略文件
```

## 部署步骤

1. 确保安装了Docker和Docker Compose
```bash
# 检查Docker版本
docker --version

# 检查Docker Compose版本
docker compose --version
```

2. 克隆项目到本地
```bash
git clone <项目地址>
cd <项目目录>
```

3. 配置环境变量
- 复制.env.example为.env
- 修改.env中的配置项

4. 构建和启动服务
```bash
# 构建镜像并启动容器
docker compose up -d

# 查看容器日志
docker compose logs -f
```

5. 验证服务
```bash
# 测试健康检查接口
curl http://localhost:5100/health
```

## 目录挂载说明
- ./chroma_db：ChromaDB数据持久化目录
- ./logs：应用日志目录

## 环境变量说明
- CHROMADB_API_KEY：API认证密钥
- DEEPSEEK_API_KEY：DeepSeek API密钥
- MODEL_PATH：模型文件路径
- CHROMADB_PERSIST_DIR：ChromaDB数据存储路径
- LOG_DIR：日志存储路径

## 常用命令
```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看日志
docker compose logs -f

# 进入容器
docker compose exec chromadb-app bash
```

## 注意事项
1. 首次启动时，需要等待模型加载完成
2. 确保挂载目录具有正确的读写权限
3. 建议定期备份chroma_db目录 




# 重新构建和运行：
## 清理现有容器和镜像
sudo docker compose down
sudo docker system prune -a

# 重新构建
sudo docker compose build --no-cache

# 启动服务
sudo docker compose up -d

# 验证Python版本：
# 进入容器
sudo docker compose exec chromadb-app bash

# 检查Python版本
python3 --version