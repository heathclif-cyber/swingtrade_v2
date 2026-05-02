import logging
import os

from flask import Flask
from dotenv import load_dotenv
from sqlalchemy import text
from app.extensions import db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        "pool_size": 3,           # kecil karena Neon pooler manage pool sendiri
        "max_overflow": 2,
        "pool_pre_ping": True,    # handle Neon auto-suspend gracefully
        "pool_reset_on_return": "rollback",  # kompatibel dengan PgBouncer transaction mode
    }
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    with app.app_context():
        # import semua models agar SQLAlchemy tahu tabel yang ada
        # NOTE: Gunakan `from` import atau alias untuk mencegah `app` (Flask
        # instance parameter) tertimpa oleh binding `import app.models`.
        import app.models as _app_models  # noqa: F401
        db.create_all()

        # Jalankan migration untuk menambah kolom baru ke tabel existing
        _run_migrations(app)
        
        # Auto-seed if database is empty, else ensure model variants exist
        from app.models.coin import Coin
        if Coin.query.count() == 0:
            _auto_seed(app)
        else:
            _ensure_model_variants(app)
            _update_model_meta(app)


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

    # Register WITA timezone filter for Jinja templates
    from app.extensions import wita_format
    app.jinja_env.filters["wita_fmt"] = wita_format


def _run_migrations(flask_app: Flask) -> None:
    """Tambahkan kolom baru ke tabel existing jika belum ada.
    
    db.create_all() hanya membuat tabel baru, bukan menambah kolom
    ke tabel yang sudah ada. Fungsi ini menjalankan ALTER TABLE
    untuk kolom-kolom yang ditambahkan setelah deploy awal.
    
    Parameter bernama flask_app (bukan app) untuk menghindari
    konflik nama dengan modul paket 'app' di __init__.py.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    migrations = [
        ("signal", "h4_swing_high", "DOUBLE PRECISION"),
        ("signal", "h4_swing_low",  "DOUBLE PRECISION"),
        ("trade",  "h4_swing_high", "DOUBLE PRECISION"),
        ("trade",  "h4_swing_low",  "DOUBLE PRECISION"),
    ]
    
    # Deteksi tipe database — gunakan flask_app bukan app agar tidak
    # bentrok dengan nama modul paket 'app'
    db_uri = flask_app.config.get("SQLALCHEMY_DATABASE_URI", "")
    is_sqlite = "sqlite" in db_uri
    
    for table, column, col_type in migrations:
        try:
            if is_sqlite:
                # SQLite tidak support IF NOT EXISTS untuk ALTER TABLE
                # Cek keberadaan kolom lewat PRAGMA
                pragma = db.session.execute(
                    text(f"PRAGMA table_info({table})")
                ).fetchall()
                existing_cols = [row[1] for row in pragma]
                if column in existing_cols:
                    continue
                db.session.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                )
            else:
                # PostgreSQL support IF NOT EXISTS
                db.session.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}")
                )
            db.session.commit()
            logger.info(f"[migration] Kolom {table}.{column} berhasil ditambahkan")
        except Exception as e:
            db.session.rollback()
            logger.warning(f"[migration] Gagal tambah {table}.{column}: {e}")


def _init_scheduler(app: Flask) -> None:
    from app.jobs import init_scheduler
    init_scheduler(app)


def _update_model_meta(app: Flask) -> None:
    """Update existing model_meta records with per-coin data if missing."""
    import json
    import logging
    from pathlib import Path
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    
    logger = logging.getLogger(__name__)
    
    # Check if we need to update (if any total_trades is None)
    needs_update = ModelMeta.query.filter(ModelMeta.total_trades.is_(None)).first()
    if not needs_update:
        return
        
    logger.info("[auto_seed] Updating missing per-coin model performance data...")
    models_dir = Path(__file__).parent.parent / "models"
    registry_path = models_dir / "model_registry.json"
    
    if not registry_path.exists():
        return
        
    with open(registry_path) as f:
        registry = json.load(f)
        
    version = registry["versions"][0]

    from app.services.model_registry import resolve_path
    config_path = resolve_path(version["paths"]["inference_config"])
    if not config_path.exists():
        return

    with open(config_path) as f:
        inference_config = json.load(f)

    bs = inference_config.get("backtest_summary", {})
    per_coin_data = inference_config.get("backtest_per_coin", {})

    updated = False
    for meta in ModelMeta.query.filter(ModelMeta.total_trades.is_(None)).all():
        coin = Coin.query.get(meta.coin_id)
        if not coin:
            continue
            
        per_coin = per_coin_data.get(coin.symbol, {})
        if per_coin:
            meta.win_rate = per_coin.get("winrate", meta.win_rate)
            meta.total_trades = per_coin.get("total_trades", meta.total_trades)
            meta.max_drawdown = per_coin.get("dd_lev5x", per_coin.get("dd_lev3x", meta.max_drawdown))
            updated = True
            
    if updated:
        db.session.commit()
        logger.info("[auto_seed] Model performance data updated successfully.")


def _ensure_model_variants(app: Flask) -> None:
    """Pastikan setiap koin memiliki ModelMeta untuk lgbm dan lstm (migrasi DB lama)."""
    import json
    import logging
    from pathlib import Path
    from datetime import datetime
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta

    logger = logging.getLogger(__name__)

    models_dir = Path(__file__).parent.parent / "models"
    registry_path = models_dir / "model_registry.json"
    if not registry_path.exists():
        return

    with open(registry_path) as f:
        registry = json.load(f)

    versions_by_type = {v["model_type"]: v for v in registry["versions"]}
    missing_types = [t for t in ("lgbm", "lstm") if t not in versions_by_type]
    if missing_types:
        logger.warning(f"[ensure_model_variants] Tipe tidak ada di registry, dilewati: {missing_types}")
    types_to_process = [t for t in ("lgbm", "lstm") if t in versions_by_type]
    if not types_to_process:
        return

    # Gunakan inference_config dari versi ensemble (atau versi pertama yang ada)
    from app.services.model_registry import resolve_path
    ref_version = versions_by_type.get("ensemble") or next(iter(versions_by_type.values()))
    config_path = resolve_path(ref_version["paths"]["inference_config"])
    with open(config_path) as f:
        inference_config = json.load(f)

    bs = inference_config.get("backtest_summary", {})
    per_coin_data = inference_config.get("backtest_per_coin", {})
    trained_at = datetime.fromisoformat(
        inference_config.get("created_at", "2026-04-25T17:02:50+00:00")
    )

    added = 0
    for coin in Coin.query.all():
        per_coin = per_coin_data.get(coin.symbol, {})
        for vtype in types_to_process:
            exists = ModelMeta.query.filter_by(
                coin_id=coin.id, model_type=vtype
            ).first()
            if exists:
                continue
            v = versions_by_type[vtype]
            p = v.get("paths", {})
            meta = ModelMeta(
                coin_id               = coin.id,
                model_type            = vtype,
                win_rate              = per_coin.get("winrate",   bs.get("mean_winrate")),
                total_trades          = per_coin.get("total_trades"),
                max_drawdown          = per_coin.get("dd_lev5x", per_coin.get("dd_lev3x", bs.get("mean_drawdown_lev5x", bs.get("mean_drawdown_lev3x")))),
                n_features            = v.get("n_features", 85),
                model_path            = p.get("lstm") or p.get("lgbm"),
                scaler_path           = p.get("scaler"),
                meta_learner_path     = None,
                calibrator_path       = None,
                inference_config_path = p.get("inference_config"),
                status                = "available",
                trained_at            = trained_at,
                evaluated_at          = trained_at,
            )
            db.session.add(meta)
            added += 1

    if added:
        db.session.commit()
        logger.info(f"[ensure_model_variants] Ditambahkan {added} ModelMeta baru (lgbm/lstm)")


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

    # Load inference config via registry path (bukan hardcoded)
    from app.services.model_registry import resolve_path
    config_path = resolve_path(version["paths"]["inference_config"])

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
    
    # Kumpulkan semua versi dari registry (ensemble, lgbm, lstm)
    trained_at = datetime.fromisoformat(inference_config.get("created_at", "2026-04-25T17:02:50+00:00"))
    bs = inference_config.get("backtest_summary", {})
    per_coin_data = inference_config.get("backtest_per_coin", {})

    versions_by_type = {v["model_type"]: v for v in registry["versions"]}

    def _make_meta(coin, vtype):
        v = versions_by_type.get(vtype, {})
        p = v.get("paths", {})
        per_coin = per_coin_data.get(coin.symbol, {})
        return ModelMeta(
            coin_id               = coin.id,
            model_type            = vtype,
            win_rate              = per_coin.get("winrate",     bs.get("mean_winrate")),
            total_trades          = per_coin.get("total_trades"),
            max_drawdown          = per_coin.get("dd_lev5x",   per_coin.get("dd_lev3x", bs.get("mean_drawdown_lev5x", bs.get("mean_drawdown_lev3x")))),
            n_features            = v.get("n_features", 85),
            model_path            = p.get("lstm") or p.get("lgbm"),
            scaler_path           = p.get("scaler"),
            meta_learner_path     = p.get("meta"),
            calibrator_path       = p.get("calibrator"),
            inference_config_path = p.get("inference_config"),
            status                = "available",
            trained_at            = trained_at,
            evaluated_at          = trained_at,
        )

    for coin in Coin.query.all():
        ensemble_meta = _make_meta(coin, "ensemble")
        lgbm_meta     = _make_meta(coin, "lgbm")
        lstm_meta     = _make_meta(coin, "lstm")
        db.session.add(ensemble_meta)
        db.session.add(lgbm_meta)
        db.session.add(lstm_meta)
        db.session.flush()

        # ModelSelection default → lstm (bukan ensemble)
        sel = ModelSelection(coin_id=coin.id, model_meta_id=lstm_meta.id)
        db.session.add(sel)

    db.session.commit()
    logger.info(f"[auto_seed] Seeded {len(symbols)} coins × 3 model types (default=lstm)")
