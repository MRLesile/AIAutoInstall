from quart import Blueprint, request, jsonify, Response
from ..utils.auth import check_auth, handle_options_request
from ..services.ai_service import stream_deepseek_api
from ..extensions import logger

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat/ai/stream', methods=['POST', 'OPTIONS'])
async def stream_chat_with_ai():
    """流式AI聊天接口"""
    if request.method == 'OPTIONS':
        return await handle_options_request(['POST', 'OPTIONS'])
        
    if not check_auth(request.headers):
        return jsonify({'error': '未授权'}), 401
        
    try:
        data = await request.get_json()
        prompt = data.get('prompt')
        conversation_id = data.get('conversation_id')

        if not prompt:
            return jsonify({'error': '缺少必要的 prompt 参数'}), 400

        async def generate():
            try:
                response = await stream_deepseek_api(prompt, conversation_id)
                if response:
                    yield response
                else:
                    yield "抱歉，AI服务暂时无法响应，请稍后重试。"
            except Exception as e:
                logger.error(f"生成响应时出错: {str(e)}")
                yield f"生成响应时发生错误: {str(e)}"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # 禁用Nginx缓冲
            }
        )

    except Exception as e:
        logger.error(f"流式聊天接口错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/chat/health', methods=['GET'])
async def chat_health_check():
    """聊天服务健康检查"""
    try:
        # 简单的测试prompt
        test_prompt = "你好"
        response = await stream_deepseek_api(test_prompt)
        
        if response:
            return jsonify({
                'status': 'healthy',
                'message': '聊天服务正常'
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'message': 'AI服务无响应'
            }), 503
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'健康检查失败: {str(e)}'
        }), 500

#TODO: 获取聊天上下文
@chat_bp.route('/chat/context', methods=['POST'])
async def get_chat_context():
    """获取聊天上下文"""
    if not check_auth(request.headers):
        return jsonify({'error': '未授权'}), 401
        
    try:
        data = await request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({'error': '缺少conversation_id参数'}), 400
            
        # TODO: 实现从数据库获取聊天历史的功能
        # 这里需要根据实际需求实现聊天历史的存储和获取
        
        return jsonify({
            'conversation_id': conversation_id,
            'history': []  # 返回空列表，等待实现
        })
        
    except Exception as e:
        logger.error(f"获取聊天上下文失败: {str(e)}")
        return jsonify({'error': str(e)}), 500 