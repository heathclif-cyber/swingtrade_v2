import os
from flask import Flask
from dotenv import load_dotenv
from app.extensions import db

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)

    _configure(app)
    _init_extensions(app)
    _register_blueprints(app)
    _init_scheduler(app)

    return app


def _configure(app: Flask) -> None:
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///instance/app.db")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": 5,
        "max_overflow": 2,
        "pool_pre_ping": True,   # handle Neon auto-suspend gracefully
    }
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    with app.app_context():
        # import semua models agar SQLAlchemy tahu tabel yang ada
        import app.models  # noqa: F401
        db.create_all()


def _register_blueprints(app: Flask) -> None:
    from app.api.dashboard import bp as dashboard_bp
    from app.api.coins import bp as coins_bp
    from app.api.signals import bp as signals_bp
    from app.api.trades import bp as trades_bp
    from app.api.models_bp import bp as models_bp
    from app.api.health import bp as health_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(coins_bp)
    app.register_blueprint(signals_bp)
    app.register_blueprint(trades_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(health_bp)


def _init_scheduler(app: Flask) -> None:
    from app.jobs import init_scheduler
    init_scheduler(app)
