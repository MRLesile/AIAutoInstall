from quart import Quart
from quart_cors import cors
from .routes.search import search_bp
from .routes.rag import rag_bp
from .routes.chat import chat_bp
from .routes.upload import upload_bp

def create_app():
    app = Quart(__name__)
    app = cors(app)
    
    # 注册蓝图
    app.register_blueprint(search_bp)
    app.register_blueprint(rag_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(upload_bp)
    
    return app 