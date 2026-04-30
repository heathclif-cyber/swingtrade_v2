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
    
    # Database URL with fallback to SQLite
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    else:
        # Use absolute path for SQLite to avoid "unable to open database file"
        db_path = os.path.join(os.getcwd(), "instance", "app.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    
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
        
        # Auto-seed if database is empty
        from app.models.coin import Coin
        if Coin.query.count() == 0:
            _auto_seed(app)


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


def _auto_seed(app: Flask) -> None:
    """Auto-seed database if empty (for Railway deployment)."""
    import json
    import logging
    from pathlib import Path
    from datetime import datetime, timezone
    from app.extensions import db, utcnow
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    
    logger = logging.getLogger(__name__)
    logger.info("[auto_seed] Database is empty, seeding...")
    
    # Load configs
    models_dir = Path(__file__).parent.parent / "models"
    registry_path = models_dir / "model_registry.json"
    
    if not registry_path.exists():
        logger.warning("[auto_seed] model_registry.json not found, skipping seed")
        return
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    version = registry["versions"][0]
    run_id = version["run_id"]
    
    # Load inference config
    config_path = models_dir / f"v{run_id}" / "inference_config.json"
    if not config_path.exists():
        config_path = models_dir / "inference_config.json"
    
    with open(config_path) as f:
        inference_config = json.load(f)
    
    # Get coins to seed (recommended + acceptable + caution)
    cv = inference_config.get("coins_validated", {})
    symbols = list(dict.fromkeys(
        cv.get("recommended", []) +
        cv.get("acceptable", []) +
        cv.get("caution", [])
    ))
    
    if not symbols:
        logger.warning("[auto_seed] No coins found in config, skipping seed")
        return
    
    # Insert coins
    for symbol in symbols:
        coin = Coin(symbol=symbol, status="active")
        db.session.add(coin)
    db.session.flush()
    
    # Insert ModelMeta for each coin (with per-coin backtest data)
    trained_at = datetime.fromisoformat(inference_config.get("created_at", "2026-04-25T17:02:50+00:00"))
    bs = inference_config.get("backtest_summary", {})
    per_coin_data = inference_config.get("backtest_per_coin", {})
    paths = version.get("paths", {})
    
    for coin in Coin.query.all():
        # Get per-coin backtest data (fallback to global summary)
        per_coin = per_coin_data.get(coin.symbol, {})
        
        meta = ModelMeta(
            coin_id=coin.id,
            model_type="ensemble",
            run_id=run_id,
            win_rate=per_coin.get("winrate", bs.get("mean_winrate")),
            total_trades=per_coin.get("total_trades"),
            max_drawdown=per_coin.get("dd_lev3x", bs.get("mean_drawdown_lev3x")),
            n_features=version.get("n_features", 85),
            model_path=paths.get("lstm"),
            scaler_path=paths.get("scaler"),
            meta_learner_path=paths.get("meta"),
            calibrator_path=paths.get("calibrator"),
            inference_config_path=paths.get("inference_config"),
            status="available",
            trained_at=trained_at,
            evaluated_at=trained_at,
        )
        db.session.add(meta)
        db.session.flush()
        
        # Insert ModelSelection
        sel = ModelSelection(coin_id=coin.id, model_meta_id=meta.id)
        db.session.add(sel)
    
    db.session.commit()
    logger.info(f"[auto_seed] Seeded {len(symbols)} coins with model {run_id}")
