import os
from quart import Quart, request, jsonify, make_response, Response
from quart_cors import cors
import logging
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import json
import threading
from collections import deque
import time
from chromadb.config import Settings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import jieba
from rank_bm25 import BM25Okapi
import numpy as np
import asyncio
import aiohttp
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from quart import copy_current_app_context
from aiohttp import ClientSession, TCPConnector
from asyncio import Queue, Lock

# ============= 常量定义 =============
# API密钥应该从环境变量读取，而不是硬编码
API_KEY = os.getenv('CHROMADB_API_KEY', 'zkcm#321')  
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-e14549e496ec4dfebe82a04ce3edc02a')
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')

# 模型相关常
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 路径配置应该更灵活
PERSIST_DIRECTORY = os.getenv('CHROMADB_PERSIST_DIR', './chroma_db')
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(CURRENT_DIR, 'Model', 'text2vec-base-chinese'))

# 区分开发和生产环境的集合名称
DEV_COLLECTION = "dev_book_chunks"    # 开发环境集合
PROD_COLLECTION = "prod_book_chunks"  # 生产环境集合


# 禁用代理设置
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
# 如果需要的话，也可以显式设置代理
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''

# ============= 日志配置 =============
LOG_DIR = os.getenv('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 禁用不必要的警告
import warnings
warnings.filterwarnings('ignore')

# 初始化模型
try:
    # 设置请求配置
    import requests
    requests.packages.urllib3.disable_warnings()
    session = requests.Session()
    session.trust_env = False  # 禁用环境变量中的代理设置
    
    model = SentenceTransformer(MODEL_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_PATH,
        trust_remote_code=True
    )
    # 测试向量生成
    test_text = ["测试文本"]
    test_embedding = embedding_function(test_text)
    logger.info(f"模型测试 - 输入类型: {type(test_text)}")
    logger.info(f"模型测试 - 输出类型: {type(test_embedding)}")
    logger.info(f"模型测试 - 输出形状: {np.array(test_embedding).shape}")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise

# 初始化 ChromaDB 客户端
try:
    settings = chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=False,  # 生产环境禁用重置
        is_persistent=True,
        persist_directory=PERSIST_DIRECTORY
    )
    
    client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY,
        settings=settings
    )

    # 在 ChromaDB 客户端初始化后添加检查
    try:
        # 列出所有集合
        collections = client.list_collections()
        logger.info(f"现有集合列表: {[col.name for col in collections]}")
    except Exception as e:
        logger.error(f"获取集合列表失败: {str(e)}")

    # 如果需要配置 HNSW 参数，在创建集合时设置
    def create_collection_with_params(name, metadata=None):
        return client.create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata=metadata,
            hnsw_config={
                "ef_construction": 200,
                "ef_search": 50,
                "M": 32
            }
        )

except Exception as e:
    logger.error(f"ChromaDB 客户端初始化失败: {str(e)}")
    raise

# 初始化 Quart 应用
app = Quart(__name__)
app = cors(app)

# 创建线程池
executor = ThreadPoolExecutor(
    max_workers=min(64, (os.cpu_count() or 1) * 8),
    thread_name_prefix="ChromaWorker"
)

# ============= 工具函数 =============
# 将装饰器定义移到这里，在路由定义之前
def async_error_handler(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            logger.error(f"{f.__name__} 失败: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    return decorated_function

#验证动态地址
def validate_url(url: str) -> bool:
    """验证URL是否以http或https开头"""
    return url.startswith(('http://', 'https://')) if url else False

def check_auth(headers):
    """验证API请求的认证信息"""
    auth_header = headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    received_key = auth_header.split(' ')[1]
    return received_key == API_KEY

async def handle_options_request(allowed_methods=['GET', 'OPTIONS']):
    """处理OPTIONS预检请求的公共方法"""
    response = await make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', ','.join(allowed_methods))
    return response

async def stream_deepseek_api(prompt, conversation_id=None):
    """异步流式调用 DeepSeek API"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 2000,
                'stream': True
            }
            
            if conversation_id:
                data['conversation_id'] = conversation_id
                
            async with session.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]
                            if json_str != '[DONE]':
                                try:
                                    chunk = json.loads(json_str)
                                    if chunk['choices'][0].get('delta', {}).get('content'):
                                        yield f"data: {json.dumps({'content': chunk['choices'][0]['delta']['content']})}\n\n"
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON解析错误: {str(e)}")
                                    continue
                                    
    except Exception as e:
        error_msg = f"API调用失败: {str(e)}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

async def process_json_for_chroma(json_content: dict, book_id: str, is_test: bool = False):
    """处理 JSON 内容并添加到 ChromaDB"""
    try:
        collection_name = DEV_COLLECTION if is_test else PROD_COLLECTION
        logger.info(f"使用集合: {collection_name}")
        
        loop = asyncio.get_event_loop()
        
        # 确保集合存在
        try:
            collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=collection_name)
            )
        except Exception as e:
            logger.info(f"集合 {collection_name} 不存在，创建新集合")
            collection = await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"environment": "development" if is_test else "production"}
                )
            )
        
        # 检查并删除已存在的数据
        try:
            existing_data = await loop.run_in_executor(
                None,
                lambda: collection.get(where={"book_id": book_id})
            )
            if existing_data['ids']:
                await loop.run_in_executor(
                    None,
                    lambda: collection.delete(where={"book_id": book_id})
                )
                logger.info(f"已删除 book_id: {book_id} 的现有数据")
        except Exception as e:
            logger.error(f"查询/删除现有数据时出错: {str(e)}")
            raise
        
        # 处理新的文本内容
        documents = []    # 原始文本
        metadatas = []    # 元数据
        ids = []          # 文档ID
        
        # 处理章节
        for chapter in json_content.get('chapters', []):
            chapter_number = chapter.get('chapter_number')
            chapter_title = chapter.get('chapter_title', '')
            
            # 处理段落
            for paragraph in chapter.get('paragraphs', []):
                if paragraph.get('text'):  # 只处理有文本内容的段落
                    doc_id = f"{book_id}_c{chapter_number}_p{paragraph.get('paragraph_number')}"
                    documents.append(paragraph['text'])
                    
                    # 构建完整的元数据
                    metadata = {
                        'book_id': book_id,
                        'chapter_number': chapter_number,
                        'chapter_title': chapter_title,
                        'chapter_index': chapter.get('chapter_index', -1),
                        'paragraph_number': paragraph.get('paragraph_number'),
                        'section_index': paragraph.get('section_index', -1),
                        'section_title': paragraph.get('section_title', ''),
                        'type': 'paragraph',
                        'book_title': json_content.get('title', ''),
                        'book_author': json_content.get('author', ''),
                        'book_description': json_content.get('description', '')
                    }
                    
                    metadatas.append(metadata)
                    ids.append(doc_id)
        
        if not documents:
            logger.warning(f"未找到任何文本内容: {book_id}")
            return {
                'status': 'warning',
                'message': '未找到任何文本内容',
                'document_count': 0
            }
            
        # 批量生成向量并添加到数据库
        try:
            embeddings = await loop.run_in_executor(
                None,
                lambda: embedding_function(documents)
            )
            
            # 分批添加数据
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                await loop.run_in_executor(
                    None,
                    lambda: collection.add(
                        documents=documents[i:end],
                        metadatas=metadatas[i:end],
                        embeddings=embeddings[i:end],
                        ids=ids[i:end]
                    )
                )
                await asyncio.sleep(0.1)  # 添加短暂延迟
                
            return {
                'status': 'success',
                'message': '文档更新成功',
                'document_count': len(documents),
                'updated': True
            }
            
        except Exception as e:
            logger.error(f"批量生成向量失败: {str(e)}")
            raise
            
    except Exception as e:
        error_msg = f"处理 JSON 失败: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def init_collections():
    """初始化 ChromaDB 集合"""
    try:
        logger.info("开始初始化 ChromaDB 集合...")
        
        # 获取或创建开发环境集合
        try:
            dev_collection = client.get_collection(name=DEV_COLLECTION)
            logger.info("已获取开发环境集合")
        except Exception as e:
            logger.info(f"创建新的开发环境集合: {str(e)}")
            dev_collection = create_collection_with_params(
                name=DEV_COLLECTION,
                metadata={"environment": "development"}
            )
        
        # 获取或创建生产环境集合
        try:
            prod_collection = client.get_collection(name=PROD_COLLECTION)
            logger.info("已获取生产环境集合")
        except Exception as e:
            logger.info(f"创建新的生产环境集合: {str(e)}")
            prod_collection = create_collection_with_params(
                name=PROD_COLLECTION,
                metadata={"environment": "production"}
            )
        
        return dev_collection, prod_collection
    except Exception as e:
        logger.error(f"初始化集合失败: {str(e)}")
        raise

def add_texts_to_collection(texts, metadata_list, ids, is_test=False):
    """添加文本到 ChromaDB 集合"""
    try:
        collection = client.get_collection(
            name=DEV_COLLECTION if is_test else PROD_COLLECTION
        )
        
        # 添加文档到集合
        collection.add(
            documents=texts,          # 原始文本
            metadatas=metadata_list,  # 元数据列表
            ids=ids                   # 文档ID列表
        )
        
        return True
    except Exception as e:
        logger.error(f"添加文本到集合失败: {str(e)}")
        return False

def query_similar_texts(query_text, n_results=3, is_test=False):
    """查询相似本"""
    try:
        collection = client.get_collection(
            name=DEV_COLLECTION if is_test else PROD_COLLECTION
        )
        
        # 查询相似文档
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']  # 包原始文档和元数据
        )
        
        return {
            'documents': results['documents'][0],    # 原始文本列表
            'metadatas': results['metadatas'][0],   # 元数据列表
            'distances': results['distances'][0]     # 相似度距离列表
        }
    except Exception as e:
        logger.error(f"查询相似文本失败: {str(e)}")
        raise

def check_collection_vectors(collection_name):
    """检查集合中的向量维度"""
    try:
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        if results['embeddings']:
            sample_vector = results['embeddings'][0]
            logger.info(f"集合 {collection_name} 中的向量维度: {len(sample_vector)}")
            return len(sample_vector)
        return None
    except Exception as e:
        logger.error(f"检查集合向量维度失败: {str(e)}")
        return None

def recreate_collection(collection_name):
    """重新创建集合"""
    try:
        # 删除现有集合
        try:
            client.delete_collection(collection_name)
            logger.info(f"已删除现有集合: {collection_name}")
        except:
            pass
        
        # 创建新集合
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"recreated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        )
        logger.info(f"已创建新集合: {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"重新创建集合失败: {str(e)}")
        raise

async def check_and_fix_collection(collection_name):
    """检查并修复集合"""
    try:
        logger.info(f"开始检查并修复集合: {collection_name}")
        
        loop = asyncio.get_event_loop()
        
        # 1. 备份现有数据
        try:
            collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=collection_name)
            )
            all_data = await loop.run_in_executor(
                None,
                collection.get
            )
            doc_count = len(all_data['ids']) if all_data['ids'] else 0
            logger.info(f"当前集合包含 {doc_count} 条数据")
            
            if doc_count == 0:
                logger.warning(f"集合 {collection_name} 为空，无需修复")
                return True
                
            # 保存要的数据字段
            backup_data = {
                'ids': all_data['ids'],
                'embeddings': all_data['embeddings'],
                'documents': all_data['documents'],
                'metadatas': all_data['metadatas']
            }
            logger.info("数据备份完成")
            
        except Exception as e:
            logger.error(f"备份数据失败: {str(e)}")
            return False
            
        # 2. 删除并重新创建集合
        try:
            await loop.run_in_executor(
                None,
                lambda: client.delete_collection(collection_name)
            )
            logger.info(f"已删除集合: {collection_name}")
            
            # 等待一小段时间确保删除操作完成
            await asyncio.sleep(0.1)
            
            new_collection = await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"recreated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                )
            )
            logger.info(f"已创建新集合: {collection_name}")
            
        except Exception as e:
            logger.error(f"重新创建集合失败: {str(e)}")
            return False
            
        # 3. 分批重新添加数据
        try:
            batch_size = 50  # 减小批次大小
            total_batches = (doc_count + batch_size - 1) // batch_size
            
            for i in range(0, doc_count, batch_size):
                batch_end = min(i + batch_size, doc_count)
                logger.info(f"正在处理批次 {i//batch_size + 1}/{total_batches}")
                
                # 准备批次数据
                batch_ids = backup_data['ids'][i:batch_end]
                batch_embeddings = backup_data['embeddings'][i:batch_end]
                batch_documents = backup_data['documents'][i:batch_end]
                batch_metadatas = backup_data['metadatas'][i:batch_end]
                
                # 添加数据
                await loop.run_in_executor(
                    None,
                    lambda: new_collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                )
                
                # 添加短暂延迟
                await asyncio.sleep(0.1)
                
            logger.info(f"集合 {collection_name} 修复完成")
            return True
            
        except Exception as e:
            logger.error(f"新加数据失败: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"修复集合失败: {str(e)}")
        return False

async def rebuild_collection(collection_name):
    """重建集合索引"""
    try:
        loop = asyncio.get_event_loop()
        
        # 获取现有数据
        collection = await loop.run_in_executor(
            None,
            lambda: client.get_collection(name=collection_name)
        )
        existing_data = await loop.run_in_executor(
            None,
            collection.get
        )
        
        # 删除集合
        await loop.run_in_executor(
            None,
            lambda: client.delete_collection(collection_name)
        )
        
        # 重新创建集合
        new_collection = await loop.run_in_executor(
            None,
            lambda: create_collection_with_params(
                name=collection_name,
                metadata={"rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            )
        )
        
        # 如果有现有数据，重新添加
        if existing_data['ids']:
            batch_size = 50
            for i in range(0, len(existing_data['ids']), batch_size):
                end = min(i + batch_size, len(existing_data['ids']))
                await loop.run_in_executor(
                    None,
                    lambda: new_collection.add(
                        ids=existing_data['ids'][i:end],
                        embeddings=existing_data['embeddings'][i:end],
                        documents=existing_data['documents'][i:end],
                        metadatas=existing_data['metadatas'][i:end]
                    )
                )
                await asyncio.sleep(0.1)  # 添加短暂延迟
                
        return True
        
    except Exception as e:
        logger.error(f"重建集合失败: {str(e)}")
        return False

# 在初始化部分添加集合检查和清理功能
async def clean_and_init_collections():
    """清理和初始化集合"""
    try:
        logger.info("开始清理和初始化集合...")
        loop = asyncio.get_event_loop()
        
        # 确保开发环境集合存在
        try:
            dev_collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=DEV_COLLECTION)
            )
            logger.info(f"开发环境集合 {DEV_COLLECTION} 已存在")
        except Exception:
            logger.info(f"创建开发环境集合 {DEV_COLLECTION}")
            dev_collection = await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    name=DEV_COLLECTION,
                    embedding_function=embedding_function,
                    metadata={"environment": "development"}
                )
            )
        
        # 确保生产环境集合存在
        try:
            prod_collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=PROD_COLLECTION)
            )
            logger.info(f"生产环境集合 {PROD_COLLECTION} 已存在")
        except Exception:
            logger.info(f"创建生产环境集合 {PROD_COLLECTION}")
            prod_collection = await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    name=PROD_COLLECTION,
                    embedding_function=embedding_function,
                    metadata={"environment": "production"}
                )
            )
        
        return dev_collection, prod_collection
        
    except Exception as e:
        logger.error(f"清理和初始化集合失败: {str(e)}")
        raise

# ============= 路由定义 =============



@app.route('/test', methods=['GET', 'OPTIONS'])
@async_error_handler
async def test_route():
    if request.method == 'OPTIONS':
        return await handle_options_request(['GET', 'OPTIONS'])
    return jsonify({'message': 'Test route works!', 'status': 'success'})



@app.route('/upload/json', methods=['POST', 'OPTIONS'])
async def upload_json():
    if request.method == 'OPTIONS':
        return await handle_options_request(['POST', 'OPTIONS'])

    if not check_auth(request.headers):
        return jsonify({'error': '未授权'}), 401
    
    try:
        data = await request.get_json()
        if not data:
            return jsonify({'error': '无效的JSON数据'}), 400
            
        json_content = data.get('content')
        is_test = data.get('is_test', False)
        notify_url = data.get('notify_url')

        # 验证 notify_url
        if not notify_url or not validate_url(notify_url):
            return jsonify({'error': 'notify_url地址不正确'}), 400
        
        book_id = json_content.get('book_id')
        if not book_id:
            return jsonify({'error': '缺少book_id字段'}), 400
            
        logger.info(f"已接收上传请求: book_id={book_id}, notify_url={notify_url},is_test={is_test}")
        
        @copy_current_app_context
        async def process_upload():
            try:
                chroma_result = await process_json_for_chroma(json_content, book_id, is_test)
                logger.info(f"ChromaDB处理完成: {book_id}")
                
                # 使用动态回调地址
                async with aiohttp.ClientSession() as session:
                    try:
                        callback_url = f"{notify_url.rstrip('/')}/{book_id}"
                        async with session.post(
                            callback_url,
                            ssl=False,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status != 200:
                                response_text = await response.text()
                                logger.warning(f"回调接口调用失败: {response.status} - {response_text}")
                            else:
                                logger.info(f"回调接口调用成功: {book_id}")
                                logger.info(f"回调URL调用成功: {callback_url}")
                    except Exception as e:
                        logger.error(f"回调接口调用异常: {str(e)}")
                        
            except Exception as e:
                logger.error(f"后台处理失败: {str(e)}")
        
        # 启动后台任务
        asyncio.create_task(process_upload())
        
        # 立即返回成功响应
        return jsonify({
            'status': 'accepted',
            'message': f'已接收上传请求，book_id: {book_id}，正在后台处理',
            'notify_url': notify_url
        })
        
    except Exception as e:
        error_msg = f"请求处理失败: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/chat/ai/stream', methods=['POST', 'OPTIONS'])
async def stream_chat_with_ai():
    """流式AI聊天接口"""
    if request.method == 'OPTIONS':
        return await handle_options_request(['POST', 'OPTIONS'])
        
    if not check_auth(request.headers):
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        data = await request.get_json()
        prompt = data.get('prompt')
        conversation_id = data.get('conversation_id')

        if not prompt:
            return jsonify({'error': '缺少必要的 prompt 参数'}), 400

        async def generate():
            async for chunk in stream_deepseek_api(prompt, conversation_id):
                yield chunk

        return Response(
            generate(),
            mimetype='text/event-stream'
        )

    except Exception as e:
        logger.error(f"流式聊天接口错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 添加查询接口
@app.route('/search', methods=['POST', 'OPTIONS'])
@async_error_handler
async def search_content():
    if request.method == 'OPTIONS':
        return await handle_options_request(['POST', 'OPTIONS'])

    if not check_auth(request.headers):
        return jsonify({'error': '未授权'}), 401
        
    try:
        data = await request.get_json()
        book_id = data.get('book_id')
        is_test = data.get('is_test', False)
        
        if not book_id:
            return jsonify({'error': '缺少book_id参数'}), 400
            
        # 使用线程池执行同步操作
        loop = asyncio.get_event_loop()
        
        # 异步获取 ChromaDB 数据
        async def get_book_data():
            collection_name = DEV_COLLECTION if is_test else PROD_COLLECTION
            collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=collection_name)
            )
            
            results = await loop.run_in_executor(
                None,
                lambda: collection.get(where={"book_id": book_id})
            )
            return results
            
        results = await get_book_data()
        
        if not results['ids']:
            return jsonify({'error': f'未找到书籍ID: {book_id}'}), 404
            
        # 重建书籍结构
        book_data = {
            'book_id': book_id,
            'title': results['metadatas'][0].get('book_title', ''),
            'author': results['metadatas'][0].get('book_author', ''),
            'description': results['metadatas'][0].get('book_description', ''),
            'chapters': []
        }
        
        # 按章节组织数据
        chapter_dict = {}
        for doc, metadata in zip(results['documents'], results['metadatas']):
            chapter_number = metadata.get('chapter_number')
            if chapter_number not in chapter_dict:
                chapter_dict[chapter_number] = {
                    'chapter_number': chapter_number,
                    'chapter_title': metadata.get('chapter_title', ''),
                    'chapter_index': metadata.get('chapter_index', -1),
                    'paragraphs': []
                }
            
            chapter_dict[chapter_number]['paragraphs'].append({
                'text': doc,
                'paragraph_number': metadata.get('paragraph_number'),
                'section_index': metadata.get('section_index', -1),
                'section_title': metadata.get('section_title', '')
            })
        
        # 按章节序号排序
        book_data['chapters'] = [
            chapter_dict[num] for num in sorted(chapter_dict.keys())
        ]
        
        return jsonify(book_data)
            
    except Exception as e:
        error_msg = f"搜索失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@app.route('/rag_search', methods=['POST', 'OPTIONS'])
@async_error_handler
@rate_limit(max_requests=100, window=60)
async def rag_search():
    if request.method == 'OPTIONS':
        return await handle_options_request(['POST', 'OPTIONS'])

    if not check_auth(request.headers):
        return jsonify({'error': '未授权'}), 401
        
    try:
        data = await request.get_json()
        query_text = data.get('query')
        book_id = data.get('book_id')
        is_test = data.get('is_test', False)
        top_k = data.get('top_k', 3)
        
        collection_name = DEV_COLLECTION if is_test else PROD_COLLECTION
        logger.info(f"使用集合: {collection_name}")
        
        # 使用线程池执行同步操作
        loop = asyncio.get_event_loop()
        
        # 获取集合和执行查询
        async def perform_search():
            collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=collection_name)
            )
            
            # 检查文档数量
            check_results = await loop.run_in_executor(
                None,
                lambda: collection.get(where={"book_id": book_id})
            )
            
            doc_count = len(check_results['ids']) if check_results['ids'] else 0
            logger.info(f"找到符合book_id={book_id}的文档数: {doc_count}")
            
            if doc_count == 0:
                return None
                
            # 生成查询向量
            query_embedding = await loop.run_in_executor(
                None,
                lambda: embedding_function([query_text])
            )
            
            logger.info(f"查询向量类型: {type(query_embedding)}")
            logger.info(f"查询向量形状: {np.array(query_embedding).shape}")
            
            # 执行向量查询
            vector_results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=query_embedding,
                    where={"book_id": book_id},
                    n_results=min(top_k * 2, doc_count),
                    include=['documents', 'metadatas', 'distances']
                )
            )
            
            return vector_results, doc_count
            
        search_result = await perform_search()
        if search_result is None:
            return jsonify({
                'query': query_text,
                'book_id': book_id,
                'results': [],
                'message': f'未找到与书籍ID {book_id} 相关的文档'
            })
            
        vector_results, doc_count = search_result
        
        # 关键索引
        logger.info("开始分词...")
        keywords = list(jieba.cut_for_search(query_text))
        logger.info(f"分词结果: {keywords}")
        
        # 混合排序结果
        logger.info("开始处理检索结果...")
        formatted_results = []
        for idx, (doc, metadata, distance) in enumerate(zip(
            vector_results['documents'][0],
            vector_results['metadatas'][0],
            vector_results['distances'][0]
        )):
            keyword_score = sum(1 for keyword in keywords if keyword in doc) / len(keywords) if keywords else 0
            logger.info(f"文档 {idx+1} 关键词匹配分数: {keyword_score}")
            
            result = {
                'content': doc,
                'chapter_info': {
                    'chapter_number': metadata.get('chapter_number', ''),
                    'index': metadata.get('chapter_index', ''),
                    'title': metadata.get('chapter_title', '')
                },
                'section_info': {
                    'paragraph_number': metadata.get('paragraph_number', ''),
                    'index': metadata.get('section_index', ''),
                    'title': metadata.get('section_title', '')
                },
                'scores': {
                    'keyword': keyword_score,
                    'vector_distance': float(distance)
                }
            }
            formatted_results.append(result)
        
        # 排序
        formatted_results.sort(key=lambda x: x['scores']['keyword'], reverse=True)
        formatted_results = formatted_results[:top_k]
        logger.info(f"最终返回结果数: {len(formatted_results)}")
        
        response = {
            'query': query_text,
            'book_id': book_id,
            'environment': 'test' if is_test else 'production',
            'results': formatted_results
        }
    
        return jsonify(response)
            
    except Exception as e:
        error_msg = f"RAG检索失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

async def get_book_list(is_test: bool = False):
    """异步获取书籍列表"""
    try:
        collection_name = DEV_COLLECTION if is_test else PROD_COLLECTION
        loop = asyncio.get_event_loop()
        
        collection = await loop.run_in_executor(
            None,
            lambda: client.get_collection(name=collection_name)
        )
        
        results = await loop.run_in_executor(
            None,
            collection.get
        )
        
        books = []
        seen_books = set()
        
        for metadata in results['metadatas']:
            book_id = metadata.get('book_id')
            if book_id and book_id not in seen_books:
                books.append({
                    'id': book_id,
                    'title': metadata.get('book_title', ''),
                    'author': metadata.get('book_author', ''),
                    'is_test': is_test
                })
                seen_books.add(book_id)
                
        return books
        
    except Exception as e:
        logger.error(f"获取书籍列表失败: {str(e)}")
        return []

# 在主程序之前添加
async def shutdown():
    """增强的优雅关闭"""
    logger.info("开始关闭服务...")
    
    try:
        # 关闭线程池
        executor.shutdown(wait=True)
        logger.info("线程池已关闭")
        
        # 关闭 ChromaDB 客户端
        client.persist()
        logger.info("ChromaDB 已持久化")
        
        # 清理其他资源
        # ...
        
    except Exception as e:
        logger.error(f"关闭服务时发生错误: {str(e)}")
    finally:
        logger.info("服务已完全关闭")

@app.before_serving
async def startup():
    """服务启动前的准备工作"""
    logger.info("服务启动中...")
    # 清理和初始化集合
    global dev_collection, prod_collection
    dev_collection, prod_collection = await clean_and_init_collections()
    logger.info("ChromaDB 集合初始化成功")

@app.after_serving
async def cleanup():
    """服务关闭时的清理工作"""
    await shutdown()

@app.route('/health', methods=['GET'])
async def health_check():
    """增强的健康检查接口"""
    try:
        # 检查基本组件
        checks = {
            "app": True,
            "chromadb": False,
            "model": False
        }
        
        # 检查 ChromaDB
        loop = asyncio.get_event_loop()
        collections = await loop.run_in_executor(None, client.list_collections)
        checks["chromadb"] = True
        
        # 检查模型
        test_text = ["测试文本"]
        test_embedding = await loop.run_in_executor(None, lambda: embedding_function(test_text))
        checks["model"] = True if test_embedding else False
        
        # 检查磁盘空间
        disk = os.statvfs(PERSIST_DIRECTORY)
        free_space = disk.f_bavail * disk.f_frsize / (1024 * 1024 * 1024)  # GB
        
        return jsonify({
            'status': 'healthy' if all(checks.values()) else 'degraded',
            'checks': checks,
            'collections': len(collections),
            'free_space_gb': round(free_space, 2),
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# 添加全局错误处理
@app.errorhandler(Exception)
async def handle_exception(e):
    logger.error(f"未捕获的异常: {str(e)}", exc_info=True)
    return jsonify({
        "error": "服务器内部错误",
        "message": str(e) if app.debug else "请联系管理员"
    }), 500

# 添加请求限制装饰器
def rate_limit(max_requests=100, window=60):  # 每分钟100个请求
    def decorator(f):
        requests = []
        
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            now = time.time()
            
            # 清理过期的请求记录
            requests[:] = [req_time for req_time in requests if now - req_time < window]
            
            if len(requests) >= max_requests:
                raise Exception("请求过于频繁，请稍后再试")
                
            requests.append(now)
            return await f(*args, **kwargs)
        return decorated_function
    return decorator

# ============= 主程序 =============
if __name__ == '__main__':
    try:
        logger.info("启动 Quart 服务...")
        app.run(host='0.0.0.0', port=5100)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        exit(1)