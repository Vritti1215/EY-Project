"""
Authentication Routes
=====================
Handles user registration, login, logout, and profile management.
Implements Flask-Login for session-based authentication with
password hashing via Werkzeug.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from models.user import User
from models.analytics import AnalyticsLog

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handle user registration.
    GET: Display the registration form.
    POST: Validate input, create user, and redirect to login.
    """
    if current_user.is_authenticated:
        return redirect(url_for('chatbot.chat'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Input validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address.')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')

        # Check for existing users
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html', username=username, email=email)

        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.commit()

            # Log registration activity
            log = AnalyticsLog(user_id=user.id, activity_type='register',
                              details=f'User {username} registered')
            db.session.add(log)
            db.session.commit()

            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle user login.
    GET: Display the login form.
    POST: Validate credentials and create a session.
    """
    if current_user.is_authenticated:
        return redirect(url_for('chatbot.chat'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user, remember=bool(remember))

            # Log login activity
            log = AnalyticsLog(user_id=user.id, activity_type='login',
                              details=f'User {user.username} logged in')
            db.session.add(log)
            db.session.commit()

            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('chatbot.chat'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Log the user out and redirect to the landing page."""
    # Log logout activity
    log = AnalyticsLog(user_id=current_user.id, activity_type='logout',
                      details=f'User {current_user.username} logged out')
    db.session.add(log)
    db.session.commit()

    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))


@auth_bp.route('/profile')
@login_required
def profile():
    """Display the user's profile page with account information."""
    from models.chat import ChatHistory, Conversation
    from models.uploads import UploadedFile

    total_conversations = Conversation.query.filter_by(user_id=current_user.id).count()
    total_messages = ChatHistory.query.filter_by(user_id=current_user.id).count()
    total_uploads = UploadedFile.query.filter_by(user_id=current_user.id).count()

    return render_template('profile.html',
                         total_conversations=total_conversations,
                         total_messages=total_messages,
                         total_uploads=total_uploads)
