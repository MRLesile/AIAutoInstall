from functools import wraps
from quart import jsonify
from ..extensions import logger

def async_error_handler(f):
    """异步错误处理装饰器"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            logger.error(f"{f.__name__} 失败: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    return decorated_function

def validate_json_data(*required_fields):
    """验证JSON数据的装饰器"""
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            from quart import request
            try:
                data = await request.get_json()
                if not data:
                    return jsonify({'error': '无效的JSON数据'}), 400
                
                # 检查必需字段
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        'error': f'缺少必需字段: {", ".join(missing_fields)}'
                    }), 400
                    
                return await f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"JSON数据验证失败: {str(e)}")
                return jsonify({'error': '无效的请求数据'}), 400
        return decorated_function
    return decorator

def rate_limit(max_requests: int, time_window: int):
    """简单的速率限制装饰器"""
    from collections import defaultdict
    import time
    
    # 使用默认字典存储请求记录
    request_records = defaultdict(list)
    
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            from quart import request
            
            # 获取客户端IP
            client_ip = request.remote_addr
            current_time = time.time()
            
            # 清理过期的请求记录
            request_records[client_ip] = [
                timestamp for timestamp in request_records[client_ip]
                if current_time - timestamp < time_window
            ]
            
            # 检查是否超过速率限制
            if len(request_records[client_ip]) >= max_requests:
                return jsonify({
                    'error': '请求过于频繁，请稍后再试',
                    'retry_after': time_window
                }), 429
                
            # 记录新的请求
            request_records[client_ip].append(current_time)
            
            return await f(*args, **kwargs)
        return decorated_function
    return decorator

def require_fields(**field_types):
    """验证请求字段类型的装饰器"""
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            from quart import request
            try:
                data = await request.get_json()
                if not data:
                    return jsonify({'error': '无效的JSON数据'}), 400
                
                # 验证字段类型
                for field, expected_type in field_types.items():
                    if field in data:
                        value = data[field]
                        if not isinstance(value, expected_type):
                            return jsonify({
                                'error': f'字段 {field} 类型错误，应为 {expected_type.__name__}'
                            }), 400
                            
                return await f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"字段验证失败: {str(e)}")
                return jsonify({'error': '请求数据验证失败'}), 400
        return decorated_function
    return decorator

def cache_response(timeout: int = 300):
    """简单的缓存装饰器"""
    from collections import defaultdict
    import time
    
    # 使用默认字典存储缓存
    cache_store = defaultdict(dict)
    
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{f.__name__}:{str(args)}:{str(kwargs)}"
            current_time = time.time()
            
            # 检查缓存是否存在且未过期
            if cache_key in cache_store:
                cached_time, cached_response = cache_store[cache_key]
                if current_time - cached_time < timeout:
                    return cached_response
                
            # 执行原始函数
            response = await f(*args, **kwargs)
            
            # 存储响应到缓存
            cache_store[cache_key] = (current_time, response)
            
            return response
        return decorated_function
    return decorator 