"""
Flask application for CV parsing and RAG-based querying.
Entry point for the application.
"""

import os
from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS

# Load environment variables
load_dotenv()


def create_app():
    """
    Application factory for creating and configuring the Flask app.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['DATABASE'] = os.getenv('DATABASE_PATH', 'data/cv.db')
    app.config['FAISS_INDEX_PATH'] = os.getenv('FAISS_INDEX_PATH', 'data/faiss.index')
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from routes.upload import upload_bp
    from routes.query import query_bp
    
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(query_bp, url_prefix='/api')
    
    # Frontend routes
    @app.route('/')
    def index():
        """Serve frontend index.html"""
        return send_from_directory('frontend', 'index.html')
    
    @app.route('/<path:filename>')
    def serve_frontend(filename):
        """Serve frontend static files"""
        if filename.startswith('api/'):
            return {'error': 'Resource not found'}, 404
        return send_from_directory('frontend', filename)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return {'status': 'healthy'}, 200
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors"""
        return {'error': 'Bad request', 'message': str(error)}, 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors"""
        return {'error': 'Resource not found', 'message': str(error)}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors"""
        return {'error': 'Internal server error', 'message': str(error)}, 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    
    # Development settings
    debug = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('FLASK_PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
