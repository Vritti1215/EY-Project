"""
Flask Extensions
================
Centralized initialization of Flask extensions to avoid circular imports.
Extensions are initialized here and bound to the app in the factory function.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

# Database ORM
db = SQLAlchemy()

# Authentication manager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# CSRF Protection
csrf = CSRFProtect()
