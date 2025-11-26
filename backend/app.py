"""
Main Flask Application for AMR Prediction System
"""
import logging
from flask import Flask
from flask_cors import CORS

from config import get_config
from api.routes import api_bp
from api.middleware import setup_middleware
from utils.logging_config import setup_logging


def create_app(config=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    if config is None:
        config = get_config()
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(app)
    logger = logging.getLogger(__name__)
    logger.info("Initializing AMR Prediction Application")
    
    # Enable CORS
    CORS(app, origins=config.CORS_ORIGINS)
    
    # Setup middleware (monitoring, audit logging)
    setup_middleware(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    
    # Health check endpoint
    @app.route("/health")
    def health_check():
        return {"status": "healthy", "service": "amr-prediction"}
    
    logger.info("AMR Prediction Application initialized successfully")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
