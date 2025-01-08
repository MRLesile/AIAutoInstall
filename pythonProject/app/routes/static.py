from quart import Blueprint, send_from_directory
import os

static_bp = Blueprint('static', __name__)

@static_bp.route('/')
async def index():
    """提供网站的主页"""
    return await send_from_directory('templates', 'index.html')

@static_bp.route('/<path:filename>')
async def serve_static(filename):
    """处理所有静态文件的请求"""
    if filename.endswith('.html'):
        # HTML文件从templates目录提供
        return await send_from_directory('templates', filename)
    # 其他静态文件（CSS、JS等）从static目录提供
    return await send_from_directory('static', filename) 