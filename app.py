"""
AI Assistant Portal — Main Application
=======================================
Enterprise-grade AI-powered chatbot web application built with Flask.
Provides authenticated home, chat, file analysis, analytics dashboard,
profile, settings, and exportable reports.
"""

import os
import logging

from flask import Flask, render_template, redirect, url_for
from flask_login import current_user, login_required
from sqlalchemy import inspect, text

from config import config_map
from extensions import db, login_manager, csrf


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database schema migration helper
# ---------------------------------------------------------------------------

def _ensure_schema_up_to_date(app):
    """Safely add missing model columns without dropping existing data."""

    with app.app_context():
        try:
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()

            for mapper in db.Model.registry.mappers:
                model = mapper.class_
                table_name = model.__tablename__

                if table_name not in existing_tables:
                    continue

                existing_columns = {
                    col["name"]
                    for col in inspector.get_columns(table_name)
                }

                model_columns = {
                    col.name: col
                    for col in model.__table__.columns
                }

                missing = set(model_columns) - existing_columns

                if not missing:
                    continue

                with db.engine.connect() as conn:
                    for col_name in missing:
                        col = model_columns[col_name]

                        col_type = col.type.compile(
                            dialect=db.engine.dialect
                        )

                        try:
                            conn.execute(
                                text(
                                    f"ALTER TABLE {table_name} "
                                    f"ADD COLUMN {col_name} {col_type}"
                                )
                            )
                            conn.commit()

                            logger.info(
                                "Schema migration: added column '%s' to '%s'.",
                                col_name,
                                table_name
                            )

                        except Exception as col_error:
                            logger.error(
                                "Schema migration failed for %s.%s: %s",
                                table_name,
                                col_name,
                                col_error
                            )

        except Exception as e:
            logger.error(
                "Schema check failed, continuing startup: %s",
                e
            )


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app(config_name=None):
    """Application factory."""

    if config_name is None:
        config_name = os.environ.get(
            "FLASK_ENV",
            "development"
        )

    app = Flask(__name__)

    app.config.from_object(
        config_map.get(
            config_name,
            config_map["default"]
        )
    )

    # -----------------------------------------------------------------------
    # Initialize extensions
    # -----------------------------------------------------------------------

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    # -----------------------------------------------------------------------
    # User loader
    # -----------------------------------------------------------------------

    from models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        """Load authenticated user from database."""
        return db.session.get(User, int(user_id))

    # -----------------------------------------------------------------------
    # Register blueprints
    # -----------------------------------------------------------------------

    from routes.auth import auth_bp
    from routes.chatbot import chatbot_bp
    from routes.uploads import uploads_bp
    from routes.analytics import analytics_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(uploads_bp)
    app.register_blueprint(analytics_bp)

    # -----------------------------------------------------------------------
    # Public Landing Page
    # -----------------------------------------------------------------------

    @app.route("/")
    def index():
        """
        Public landing page.

        If the user is already authenticated, send them directly
        to the application Home page.
        """

        if current_user.is_authenticated:
            return redirect(url_for("main_home"))

        return render_template("index.html")

    # -----------------------------------------------------------------------
    # Authenticated Home Page
    # -----------------------------------------------------------------------

    @app.route("/home")
    @login_required
    def main_home():
        """
        Main authenticated Home page.

        This is the page users should see immediately after login.
        """

        return render_template(
            "home.html",
            user=current_user
        )

    # -----------------------------------------------------------------------
    # Error Handlers
    # -----------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(error):
        return render_template(
            "errors/404.html"
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()

        return render_template(
            "errors/500.html"
        ), 500

    # -----------------------------------------------------------------------
    # Database + Required Directories
    # -----------------------------------------------------------------------

    with app.app_context():

        os.makedirs(
            app.config.get(
                "UPLOAD_FOLDER",
                "uploads"
            ),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(
                app.root_path,
                "database"
            ),
            exist_ok=True
        )

        db.create_all()

        logger.info(
            "Database tables created/verified successfully."
        )

        _ensure_schema_up_to_date(app)

    return app


# ---------------------------------------------------------------------------
# Application Instance
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Development Server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=app.config.get("DEBUG", False),
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )