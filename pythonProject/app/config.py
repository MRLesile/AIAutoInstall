import os

# API配置
API_KEY = "zkcm#321"
DEEPSEEK_API_KEY = 'sk-e14549e496ec4dfebe82a04ce3edc02a'
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'

# ChromaDB配置
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DEV_COLLECTION = "dev_book_chunks"
PROD_COLLECTION = "prod_book_chunks"

# 模型配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'Model', 'text2vec-base-chinese')

# 禁用代理设置
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*' 