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


def _parse_list(env_value, default_csv):
    raw = os.environ.get(env_value, default_csv)
    return [item.strip().lower() for item in raw.split(',') if item.strip()]


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

    # --- Chart / Financial Insight Configuration ---------------------------
    # Seaborn style: darkgrid, whitegrid, dark, white, ticks
    CHART_STYLE = os.environ.get('CHART_STYLE', 'darkgrid')
    # 'ey' (black/yellow/grey/white brand theme, default) or 'blue'/'red'/
    # 'green' (additional single-hue options), or any regular seaborn/
    # matplotlib palette name: viridis, mako, rocket, coolwarm, Set2,
    # husl, deep, muted, pastel, etc. See services/chart_service.py's
    # BRAND_PALETTES / COLOR_ALIASES for the full resolution logic.
    CHART_PALETTE = os.environ.get('CHART_PALETTE', 'ey')
    # Seaborn context: paper, notebook, talk, poster (controls font/line scale)
    CHART_CONTEXT = os.environ.get('CHART_CONTEXT', 'notebook')
    CHART_DPI = int(os.environ.get('CHART_DPI', 150))
    CHART_FIGSIZE = (
        float(os.environ.get('CHART_WIDTH_IN', 10)),
        float(os.environ.get('CHART_HEIGHT_IN', 6)),
    )
    # Keywords that trigger chart generation instead of (or alongside) a
    # plain text answer. Comma-separated env var, e.g.
    # CHART_TRIGGER_KEYWORDS="graph,chart,plot,visualize,trend"
    CHART_TRIGGER_KEYWORDS = _parse_list(
        'CHART_TRIGGER_KEYWORDS',
        'graph,chart,plot,visualize,visualise,trend,bar chart,line chart,pie chart,draw,diagram,picture'
    )
    # Month a fiscal year starts on: 1 = calendar year (Q1 = Jan-Mar),
    # 4 = Indian-style fiscal year (Q1 = Apr-Jun). Defaults to 4 for this
    # deployment; set FISCAL_YEAR_START_MONTH=1 for calendar quarters.
    FISCAL_YEAR_START_MONTH = int(os.environ.get('FISCAL_YEAR_START_MONTH', 4))
    CHART_AI_ENABLED = os.environ.get('CHART_AI_ENABLED', 'true').lower() == 'true'
    CHART_MAX_POINTS = int(os.environ.get('CHART_MAX_POINTS', 60))
    CHART_ALTERNATIVES = int(os.environ.get('CHART_ALTERNATIVES', 2))


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