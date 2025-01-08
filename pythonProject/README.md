# 数字化教材系统

这是一个基于 Quart 和 ChromaDB 的数字化教材系统，提供教材内容的存储、检索和智能问答功能。

## 项目结构 

pythonProject/
├── app/
│ ├── init.py # 应用工厂和蓝图注册
│ ├── config.py # 配置常量
│ ├── extensions.py # 共享扩展实例
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
├── run.py # 启动脚本
└── requirements.txt # 项目依赖

## 模块说明

### 配置相关
- `app/config.py`: 包含所有配置常量，包括API密钥、数据库路径、模型配置等

### 核心功能
- `app/extensions.py`: 包含共享的扩展实例
  - ChromaDB客户端初始化
  - Embedding模型加载
  - 日志配置等

### 路由模块
- `app/routes/search.py`: 教材内容搜索路由
- `app/routes/rag.py`: 基于RAG的智能搜索路由
- `app/routes/chat.py`: AI对话接口路由
- `app/routes/upload.py`: 教材内容上传路由

### 服务层
- `app/services/chroma_service.py`: ChromaDB操作服务
  - 向量数据库的增删改查
  - 相似度搜索等
- `app/services/ai_service.py`: AI对话服务
  - 与DeepSeek API的交互
  - 流式响应处理
- `app/services/upload_service.py`: 上传服务
  - 文件处理
  - 数据导入等

### 工具类
- `app/utils/auth.py`: 认证相关工具
  - API密钥验证
  - 权限检查等
- `app/utils/decorators.py`: 自定义装饰器
  - 错误处理
  - 请求验证
  - 缓存等
- `app/utils/helpers.py`: 辅助函数
  - URL验证
  - 文件处理
  - 通用工具函数

### 应用入口
- `app/__init__.py`: 应用工厂函数和蓝图注册
- `run.py`: 应用启动脚本

## 安装和运行

1. 安装依赖：、
   bash
pip install -r requirements.txt

2. 运行应用：
python run.py

+### 方式二：Docker 部署 //TODO
+1. 构建镜像：
+```bash
+docker build -t digital-textbook .
+```
+
+2. 运行容器：
+```bash
+docker run -d \
+  --name digital-textbook \
+  -p 5100:5100 \
+  -v $(pwd)/chroma_db:/app/chroma_db \
+  -v $(pwd)/app/Model:/app/app/Model \
+  digital-textbook
+```
+
+或者使用 docker-compose：
+```bash
+docker-compose up -d
+```

## API 认证

所有API请求需要在header中包含认证信息：Authorization: Bearer zkcm#321

## 注意事项

1. 确保已安装所有必要的依赖
2. 检查配置文件中的路径和API密钥设置
3. 确保Model目录下有正确的模型文件
4. 建议在生产环境中使用更安全的认证机制

## 维护和扩展

- 每个模块都有其明确的职责
- 遵循模块化和单一职责原则
- 使用蓝图进行路由管理
- 通过服务层隔离业务逻辑
- 工具类提供通用功能

