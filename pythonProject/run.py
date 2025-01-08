import asyncio
from app import create_app
from app.extensions import logger

app = create_app()

if __name__ == '__main__':
    try:
        logger.info("启动 Quart 服务...")
        app.run(host='0.0.0.0', port=5100)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        exit(1) 