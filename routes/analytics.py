"""
Analytics Routes
================
Provides dashboard analytics with metrics, charts data,
and activity tracking for users and administrators.
"""

from datetime import datetime, timedelta, timezone
from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func
from extensions import db
from models.user import User
from models.chat import ChatHistory, Conversation
from models.uploads import UploadedFile
from models.analytics import AnalyticsLog

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/dashboard')
@login_required
def dashboard():
    """Render the analytics dashboard with key metrics."""
    # Core metrics
    total_users = User.query.count()
    total_conversations = Conversation.query.count()
    total_messages = ChatHistory.query.count()
    total_uploads = UploadedFile.query.count()

    # User-specific metrics
    my_conversations = Conversation.query.filter_by(user_id=current_user.id).count()
    my_messages = ChatHistory.query.filter_by(user_id=current_user.id).count()
    my_uploads = UploadedFile.query.filter_by(user_id=current_user.id).count()

    # Average response time
    avg_response = db.session.query(
        func.avg(ChatHistory.response_time)
    ).filter(ChatHistory.user_id == current_user.id).scalar()
    avg_response_time = round(avg_response, 2) if avg_response else 0

    return render_template('dashboard.html',
                         total_users=total_users,
                         total_conversations=total_conversations,
                         total_messages=total_messages,
                         total_uploads=total_uploads,
                         my_conversations=my_conversations,
                         my_messages=my_messages,
                         my_uploads=my_uploads,
                         avg_response_time=avg_response_time)


@analytics_bp.route('/dashboard/chart-data')
@login_required
def chart_data():
    """
    Provide JSON chart data for the analytics dashboard.
    Returns daily activity, user growth, and upload statistics.
    """
    now = datetime.now(timezone.utc)
    days_range = 30

    # Daily Chat Activity (last 30 days)
    daily_chats = []
    daily_labels = []
    for i in range(days_range - 1, -1, -1):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)

        count = ChatHistory.query.filter(
            ChatHistory.timestamp >= day_start,
            ChatHistory.timestamp <= day_end
        ).count()

        daily_chats.append(count)
        daily_labels.append(day.strftime('%b %d'))

    # User Growth (last 30 days cumulative)
    user_growth = []
    for i in range(days_range - 1, -1, -1):
        day = now - timedelta(days=i)
        day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)

        count = User.query.filter(User.created_at <= day_end).count()
        user_growth.append(count)

    # File Upload Statistics (by type)
    upload_stats = db.session.query(
        UploadedFile.file_type,
        func.count(UploadedFile.id)
    ).group_by(UploadedFile.file_type).all()

    upload_types = [stat[0].upper() for stat in upload_stats] if upload_stats else ['PDF', 'DOCX', 'TXT', 'CSV']
    upload_counts = [stat[1] for stat in upload_stats] if upload_stats else [0, 0, 0, 0]

    # Most Active Users (top 5)
    active_users = db.session.query(
        User.username,
        func.count(ChatHistory.id).label('msg_count')
    ).join(ChatHistory, User.id == ChatHistory.user_id
    ).group_by(User.username
    ).order_by(func.count(ChatHistory.id).desc()
    ).limit(5).all()

    active_user_names = [u[0] for u in active_users] if active_users else ['No data']
    active_user_counts = [u[1] for u in active_users] if active_users else [0]

    return jsonify({
        'daily_activity': {
            'labels': daily_labels,
            'data': daily_chats
        },
        'user_growth': {
            'labels': daily_labels,
            'data': user_growth
        },
        'upload_stats': {
            'labels': upload_types,
            'data': upload_counts
        },
        'active_users': {
            'labels': active_user_names,
            'data': active_user_counts
        }
    })
