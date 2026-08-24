"""
Application Configuration Module
================================
Manages all application-level configurations including database settings,
security parameters, file upload limits, and API integration keys.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration class with sensible defaults."""

    # Flask Core Settings
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

    # SQLAlchemy Database Configuration
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(BASE_DIR, "database", "app.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'}

    # Ollama Cloud API Configuration
    OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', '')
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.1')
    # Comma-separated list of fallback models tried in order when OLLAMA_MODEL fails
    OLLAMA_FALLBACK_MODELS = os.environ.get(
        'OLLAMA_FALLBACK_MODELS', 'qwen3:32b-cloud,llama3.1'
    )

    # Session Configuration
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour session timeout
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # CSRF Protection
    WTF_CSRF_ENABLED = True


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production-specific configuration with stricter security."""
    DEBUG = False
    FLASK_ENV = 'production'
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    """Testing-specific configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
