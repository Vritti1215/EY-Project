"""
Authentication Routes
=====================
Handles registration, login, logout, profile management,
account details updates, and password changes.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user

from extensions import db
from models.user import User
from models.analytics import AnalyticsLog


auth_bp = Blueprint("auth", __name__)


# ============================================================
# REGISTER
# ============================================================

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Register a new user."""

    if current_user.is_authenticated:
        return redirect(url_for("main_home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []

        if len(username) < 3:
            errors.append("Username must be at least 3 characters.")

        if not email or "@" not in email:
            errors.append("Please enter a valid email address.")

        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")

        if password != confirm_password:
            errors.append("Passwords do not match.")

        if User.query.filter_by(username=username).first():
            errors.append("Username already taken.")

        if User.query.filter_by(email=email).first():
            errors.append("Email already registered.")

        if errors:
            for error in errors:
                flash(error, "error")

            return render_template(
                "register.html",
                username=username,
                email=email
            )

        try:
            user = User(
                username=username,
                email=email
            )
            user.set_password(password)

            db.session.add(user)
            db.session.flush()

            db.session.add(
                AnalyticsLog(
                    user_id=user.id,
                    activity_type="register",
                    details=f"User {username} registered"
                )
            )

            db.session.commit()

            flash(
                "Registration successful! Please log in.",
                "success"
            )

            return redirect(url_for("auth.login"))

        except Exception:
            db.session.rollback()

            flash(
                "An error occurred during registration. Please try again.",
                "error"
            )

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate an existing user."""

    if current_user.is_authenticated:
        return redirect(url_for("main_home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        try:
            login_user(user, remember=remember)

            db.session.add(
                AnalyticsLog(
                    user_id=user.id,
                    activity_type="login",
                    details=f"User {user.username} logged in"
                )
            )

            db.session.commit()

            flash(
                f"Welcome back, {user.username}!",
                "success"
            )

            next_page = request.args.get("next")

            return redirect(
                next_page or url_for("main_home")
            )

        except Exception:
            db.session.rollback()

            flash(
                "Unable to complete login. Please try again.",
                "error"
            )

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@auth_bp.route("/logout")
@login_required
def logout():
    """Log out the current user."""

    try:
        db.session.add(
            AnalyticsLog(
                user_id=current_user.id,
                activity_type="logout",
                details=f"User {current_user.username} logged out"
            )
        )

        db.session.commit()

    except Exception:
        db.session.rollback()

    logout_user()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(url_for("index"))


# ============================================================
# PROFILE
# ============================================================

@auth_bp.route("/profile")
@login_required
def profile():
    """Display user profile and activity statistics."""

    from models.chat import ChatHistory, Conversation
    from models.uploads import UploadedFile

    user_id = current_user.id

    total_conversations = Conversation.query.filter_by(
        user_id=user_id
    ).count()

    total_messages = ChatHistory.query.filter_by(
        user_id=user_id
    ).count()

    total_uploads = UploadedFile.query.filter_by(
        user_id=user_id
    ).count()

    return render_template(
        "profile.html",
        total_conversations=total_conversations,
        total_messages=total_messages,
        total_uploads=total_uploads
    )


# ============================================================
# SETTINGS
# ============================================================

@auth_bp.route("/settings")
@login_required
def settings():
    """Display account settings."""

    return render_template("settings.html")


# ============================================================
# EDIT ACCOUNT DETAILS
# ============================================================

@auth_bp.route("/settings/edit", methods=["POST"])
@login_required
def edit_details():
    """Update username and email."""

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()

    # Basic validation
    if len(username) < 3:
        flash(
            "Name must be at least 3 characters.",
            "error"
        )
        return redirect(url_for("auth.settings"))

    if not email or "@" not in email:
        flash(
            "Please enter a valid email address.",
            "error"
        )
        return redirect(url_for("auth.settings"))

    # Check username only if it changed
    if username != current_user.username:
        if User.query.filter(
            User.username == username,
            User.id != current_user.id
        ).first():

            flash(
                "This username is already taken.",
                "error"
            )
            return redirect(url_for("auth.settings"))

    # Check email only if it changed
    if email != current_user.email:
        if User.query.filter(
            User.email == email,
            User.id != current_user.id
        ).first():

            flash(
                "This email address is already registered.",
                "error"
            )
            return redirect(url_for("auth.settings"))

    try:
        current_user.username = username
        current_user.email = email

        db.session.add(
            AnalyticsLog(
                user_id=current_user.id,
                activity_type="profile_update",
                details="User updated account details"
            )
        )

        db.session.commit()

        flash(
            "Your account details have been updated successfully.",
            "success"
        )

    except Exception:
        db.session.rollback()

        flash(
            "Unable to update your account details. Please try again.",
            "error"
        )

    return redirect(url_for("auth.settings"))


# ============================================================
# CHANGE PASSWORD
# ============================================================

@auth_bp.route("/settings/password", methods=["POST"])
@login_required
def change_password():
    """Change the current user's password."""

    current_password = request.form.get(
        "current_password",
        ""
    )

    new_password = request.form.get(
        "new_password",
        ""
    )

    confirm_password = request.form.get(
        "confirm_password",
        ""
    )

    # Verify current password
    if not current_user.check_password(current_password):
        flash(
            "Current password is incorrect.",
            "error"
        )
        return redirect(url_for("auth.settings"))

    # Validate new password
    if len(new_password) < 6:
        flash(
            "New password must be at least 6 characters.",
            "error"
        )
        return redirect(url_for("auth.settings"))

    if new_password != confirm_password:
        flash(
            "New passwords do not match.",
            "error"
        )
        return redirect(url_for("auth.settings"))

    if current_user.check_password(new_password):
        flash(
            "New password must be different from your current password.",
            "error"
        )
        return redirect(url_for("auth.settings"))

    try:
        current_user.set_password(new_password)

        db.session.add(
            AnalyticsLog(
                user_id=current_user.id,
                activity_type="password_change",
                details="User changed their password"
            )
        )

        db.session.commit()

        flash(
            "Password changed successfully.",
            "success"
        )

    except Exception:
        db.session.rollback()

        flash(
            "Unable to change your password. Please try again.",
            "error"
        )

    return redirect(url_for("auth.settings"))