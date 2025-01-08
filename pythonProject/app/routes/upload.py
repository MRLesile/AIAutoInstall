from quart import Blueprint, request, jsonify
from ..utils.auth import check_auth, handle_options_request
from ..services.upload_service import process_json_for_chroma, check_and_fix_collection
from ..extensions import logger
import aiohttp
from ..utils.helpers import validate_url

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload/json', methods=['POST', 'OPTIONS'])
async def upload_json():
    """上传JSON数据接口"""
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
            
        logger.info(f"已接收上传请求: book_id={book_id}, notify_url={notify_url}, is_test={is_test}")
        
        async def process_upload():
            try:
                # 处理JSON数据
                chroma_result = await process_json_for_chroma(json_content, book_id, is_test)
                logger.info(f"ChromaDB处理完成: {book_id}")
                
                # 调用回调接口
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
        from quart import copy_current_app_context
        @copy_current_app_context
        async def wrapped_process():
            await process_upload()
            
        from asyncio import create_task
        create_task(wrapped_process())
        
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

@upload_bp.route('/upload/repair', methods=['POST'])
async def repair_collection():
    """修复集合"""
    if not check_auth(request.headers):
        return jsonify({'error': '未授权'}), 401
        
    try:
        data = await request.get_json()
        collection_name = data.get('collection_name')
        
        if not collection_name:
            return jsonify({'error': '缺少collection_name参数'}), 400
            
        success = await check_and_fix_collection(collection_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'集合 {collection_name} 修复完成'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'集合 {collection_name} 修复失败'
            }), 500
            
    except Exception as e:
        error_msg = f"修复请求处理失败: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500 