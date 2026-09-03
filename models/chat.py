"""
Chat Models
===========
Defines the Conversation and ChatHistory models for managing
user conversations and individual messages with the AI assistant.
"""

from datetime import datetime, timezone
from extensions import db


class Conversation(db.Model):
    """
    Represents a conversation session between a user and the AI.

    Attributes:
        id: Primary key identifier.
        user_id: Foreign key linking to the owning user.
        title: Display title for the conversation (auto-generated or user-set).
        created_at: Timestamp of conversation creation.
        updated_at: Timestamp of last activity in the conversation.
    """
    __tablename__ = 'conversations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(200), default='New Conversation')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    messages = db.relationship('ChatHistory', backref='conversation', lazy='dynamic',
                               cascade='all, delete-orphan', order_by='ChatHistory.timestamp')

    def __repr__(self):
        return f'<Conversation {self.id}: {self.title}>'


class ChatHistory(db.Model):
    """
    Represents a single message exchange (user prompt + AI response).

    Attributes:
        id: Primary key identifier.
        user_id: Foreign key linking to the user who sent the message.
        conversation_id: Foreign key linking to the parent conversation.
        prompt: The user's input message.
        response: The AI-generated response.
        timestamp: When the message exchange occurred.
        response_time: Time taken for the AI to respond (in seconds).
        chart_image: Optional base64 PNG data URI of a generated chart,
            so charts persist across page reloads / conversation history
            fetches without needing separate file storage.
    """
    __tablename__ = 'chat_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=True, index=True)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    response_time = db.Column(db.Float, default=0.0)  # seconds
    chart_image = db.Column(db.Text, nullable=True, default=None)

    def to_dict(self):
        """Serialize the message to a dictionary for JSON responses."""
        return {
            'id': self.id,
            'prompt': self.prompt,
            'response': self.response,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else None,
            'response_time': round(self.response_time, 2) if self.response_time else 0,
            'chart_image': self.chart_image
        }

    def __repr__(self):
        return f'<ChatHistory {self.id}>'