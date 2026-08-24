"""
AI Assistant Portal — Main Application
=======================================
Enterprise-grade AI-powered chatbot web application built with Flask.
Provides authenticated chat, file analysis, analytics dashboard,
and exportable reports.

Author: EY Project Team
Version: 1.0.0
"""

import os
import logging
from flask import Flask, render_template
from config import config_map
from extensions import db, login_manager, csrf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """
    Application factory function.
    Creates and configures the Flask application with all extensions,
    blueprints, and database initialization.

    Args:
        config_name: Configuration profile to use ('development', 'production', 'testing').

    Returns:
        Flask: Configured Flask application instance.
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(config_map.get(config_name, config_map['default']))

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    # User loader for Flask-Login
    from models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from routes.auth import auth_bp
    from routes.chatbot import chatbot_bp
    from routes.uploads import uploads_bp
    from routes.analytics import analytics_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(uploads_bp)
    app.register_blueprint(analytics_bp)

    # Main/landing page route
    @app.route('/')
    def index():
        """Render the landing page."""
        return render_template('index.html')

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500

    # Ensure required directories exist
    with app.app_context():
        os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'database'), exist_ok=True)

        # Create database tables
        db.create_all()
        logger.info("Database tables created/verified successfully.")

    return app


# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(
        debug=app.config.get('DEBUG', False),
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )
