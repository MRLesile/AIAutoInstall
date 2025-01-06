# 使用支持CUDA 12.2的基础镜像
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 安装Python和pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl && \
    rm -rf /var/lib/apt/lists/*

# 创建Python3.10的软链接
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/Model
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH="/usr/local/bin:$PATH"

# 首先复制本地模型文件并设置权限
COPY Model/ ./Model/
RUN chmod -R 755 /app/Model

# 复制requirements.txt并安装依赖
COPY requirements.txt .
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pip setuptools wheel && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制应用代码
COPY ChromaDBApp.py .

# 创建必要的目录并设置权限
RUN mkdir -p /app/chroma_db /app/logs && \
    chmod -R 755 /app/chroma_db && \
    chmod -R 755 /app/logs

# 暴露端口
EXPOSE 5100

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5100/health || exit 1

# 启动命令
CMD ["python3", "ChromaDBApp.py"]