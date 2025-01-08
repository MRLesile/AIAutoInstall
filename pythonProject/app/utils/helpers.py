import re
import time
import asyncio
from urllib.parse import urlparse
from ..extensions import logger
from typing import Union, List, Dict, Any
import json
import aiohttp
from functools import wraps

def validate_url(url: str) -> bool:
    """验证URL是否有效"""
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False

def format_error_response(error: Union[str, Exception], status_code: int = 500) -> tuple:
    """格式化错误响应"""
    error_message = str(error)
    logger.error(f"错误: {error_message}")
    return {
        'error': error_message,
        'status': 'error',
        'timestamp': time.time()
    }, status_code

async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """异步重试装饰器的实现函数"""
    retries = 0
    current_delay = delay
    
    while retries < max_retries:
        try:
            return await func()
        except exceptions as e:
            retries += 1
            if retries == max_retries:
                logger.error(f"重试{max_retries}次后仍然失败: {str(e)}")
                raise
                
            logger.warning(f"第{retries}次重试失败，等待{current_delay}秒后重试: {str(e)}")
            await asyncio.sleep(current_delay)
            current_delay *= backoff

def chunk_list(lst: list, chunk_size: int) -> List[list]:
    """将列表分割成固定大小的块"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

async def safe_request(
    url: str,
    method: str = 'GET',
    **kwargs
) -> Dict[str, Any]:
    """安全的HTTP请求封装"""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, **kwargs) as response:
                if response.content_type == 'application/json':
                    return await response.json()
                return {'text': await response.text()}
    except Exception as e:
        logger.error(f"HTTP请求失败: {str(e)}")
        raise

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全字符"""
    # 移除不安全字符，只保留字母、数字、下划线、横线和点
    safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
    return safe_filename

def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """安全地解析JSON字符串"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        return {}

class AsyncTimer:
    """异步计时器类"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"计时器 {self.name} 耗时: {duration:.2f}秒")

def memoize(timeout: int = 300):
    """带超时的记忆化装饰器"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str((args, sorted(kwargs.items())))
            now = time.time()
            
            # 检查缓存是否存在且未过期
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < timeout:
                    return result
                    
            # 执行函数并缓存结果
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            cache[key] = (result, now)
            return result
            
        return wrapper
    return decorator

def validate_book_id(book_id: str) -> bool:
    """验证书籍ID格式"""
    # 假设书籍ID的格式为：字母和数字的组合，长度在4-32之间
    pattern = r'^[A-Za-z0-9]{4,32}$'
    return bool(re.match(pattern, book_id))

async def cleanup_old_files(directory: str, max_age_days: int = 7):
    """清理指定目录中的旧文件"""
    import os
    from datetime import datetime, timedelta
    
    try:
        now = datetime.now()
        cutoff = now - timedelta(days=max_age_days)
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if mtime < cutoff:
                    try:
                        os.remove(filepath)
                        logger.info(f"已删除旧文件: {filepath}")
                    except Exception as e:
                        logger.error(f"删除文件失败 {filepath}: {str(e)}")
                        
    except Exception as e:
        logger.error(f"清理旧文件失败: {str(e)}")

def get_file_size(file_path: str) -> str:
    """获取文件大小的人类可读格式"""
    import os
    
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f}{unit}"
            size /= 1024
        return f"{size:.2f}TB"
    except Exception as e:
        logger.error(f"获取文件大小失败 {file_path}: {str(e)}")
        return "未知" 