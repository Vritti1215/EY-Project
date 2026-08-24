"""
Analytics Model
===============
Defines the AnalyticsLog model for tracking user activity
across the application for dashboard metrics and reporting.
"""

from datetime import datetime, timezone
from extensions import db


class AnalyticsLog(db.Model):
    """
    Tracks user activities for analytics and reporting.

    Attributes:
        id: Primary key identifier.
        user_id: Foreign key linking to the user performing the action.
        activity_type: Category of activity (e.g., 'chat', 'upload', 'login', 'export').
        details: Additional context or metadata about the activity.
        timestamp: When the activity occurred.
    """
    __tablename__ = 'analytics_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    activity_type = db.Column(db.String(50), nullable=False, index=True)
    details = db.Column(db.String(500), default='')
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self):
        return f'<AnalyticsLog {self.activity_type} by User {self.user_id}>'
