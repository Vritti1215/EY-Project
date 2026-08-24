"""
User Model
==========
Defines the User database model with authentication support via Flask-Login.
Handles password hashing and verification using Werkzeug's security utilities.
"""

from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db


class User(UserMixin, db.Model):
    """
    User model representing registered application users.

    Attributes:
        id: Primary key identifier.
        username: Unique username for the user.
        email: Unique email address for the user.
        password_hash: Hashed password (never stores plaintext).
        created_at: Timestamp of account creation.
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    chat_history = db.relationship('ChatHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    uploaded_files = db.relationship('UploadedFile', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    analytics_logs = db.relationship('AnalyticsLog', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and store the user's password securely."""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        """Verify a password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
