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
            _sync_ensemble_model_type(app)
            _fix_stale_model_paths(app)


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
        # PerformanceSummary snapshot history
        ("performance_summary", "snapshot_at", "TIMESTAMP WITH TIME ZONE"),
    ]

    # Drop unique constraint agar bisa menyimpan history snapshot
    # (SQLite tidak support DROP CONSTRAINT, harus recreate table)
    if not is_sqlite:
        try:
            db.session.execute(
                text("ALTER TABLE performance_summary DROP CONSTRAINT IF EXISTS uq_perf_coin_period")
            )
            db.session.commit()
            logger.info("[migration] Unique constraint uq_perf_coin_period dihapus")
        except Exception as e:
            db.session.rollback()
            logger.warning(f"[migration] Gagal hapus constraint uq_perf_coin_period: {e}")
    
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
    """Pastikan setiap koin memiliki ModelMeta untuk lgbm dan lstm.

    Jika lgbm/lstm tidak ada di registry (misal hanya ada ensemble_v2),
    buat record dari paths versi ensemble utama — karena semua file model
    (lstm_best.pt, lstm_scaler.pkl, lgbm_baseline.pkl) ada di direktori
    models/ yang sama.
    """
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

    # Cari tipe ensemble utama (bukan lgbm/lstm) sebagai referensi paths
    main_types = [t for t in versions_by_type if t not in ("lgbm", "lstm")]
    if not main_types:
        logger.warning("[ensure_model_variants] Tidak ada tipe model utama di registry")
        return

    main_type = main_types[0]
    main_version = versions_by_type[main_type]

    # Tentukan legacy types yang perlu dibuat
    # Jika lgbm/lstm sudah ada di registry → buat dari entry masing-masing
    # Jika tidak ada → buat dari main_version (paths sama, file satu direktori)
    legacy_types = ("lgbm", "lstm")
    types_to_process = [t for t in legacy_types if t not in versions_by_type]
    if not types_to_process:
        # Semua legacy type sudah ada di registry, buat dari entry masing-masing
        types_to_process = list(legacy_types)

    from app.services.model_registry import resolve_path
    config_path = resolve_path(main_version["paths"]["inference_config"])
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

            # Paths: dari registry jika ada entry khusus, atau fallback ke main_version
            if vtype in versions_by_type:
                v = versions_by_type[vtype]
                p = v.get("paths", {})
            else:
                v = main_version
                p = main_version.get("paths", {})

            meta = ModelMeta(
                coin_id               = coin.id,
                model_type            = vtype,
                win_rate              = per_coin.get("winrate",   bs.get("mean_winrate")),
                total_trades          = per_coin.get("total_trades"),
                max_drawdown          = per_coin.get("dd_lev5x", per_coin.get("dd_lev3x", bs.get("mean_drawdown_lev5x", bs.get("mean_drawdown_lev3x")))),
                n_features            = v.get("n_features", 85),
                model_path            = p.get("lstm") or p.get("lgbm"),
                scaler_path           = p.get("scaler"),
                # lgbm/lstm standalone tidak pakai meta_learner / calibrator
                meta_learner_path     = p.get("meta") if vtype not in ("lgbm", "lstm") else None,
                calibrator_path       = p.get("calibrator") if vtype not in ("lgbm", "lstm") else None,
                inference_config_path = p.get("inference_config"),
                status                = "available",
                trained_at            = trained_at,
                evaluated_at          = trained_at,
            )
            db.session.add(meta)
            added += 1

    if added:
        db.session.commit()
        logger.info(f"[ensure_model_variants] Ditambahkan {added} ModelMeta baru ({', '.join(types_to_process)})")


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
    
    # Kumpulkan semua versi dari registry (ensemble_v2, lgbm, lstm)
    trained_at = datetime.fromisoformat(inference_config.get("created_at", "2026-04-25T17:02:50+00:00"))
    bs = inference_config.get("backtest_summary", {})
    per_coin_data = inference_config.get("backtest_per_coin", {})

    versions_by_type = {v["model_type"]: v for v in registry["versions"]}

    # Cari tipe ensemble utama sebagai referensi paths untuk legacy lgbm/lstm
    main_types = [t for t in versions_by_type if t not in ("lgbm", "lstm")]
    main_version = versions_by_type[main_types[0]] if main_types else next(iter(versions_by_type.values()))

    # Selalu buat lgbm/lstm: dari registry jika ada, fallback ke main_version
    all_types_to_seed = list(dict.fromkeys(
        list(versions_by_type.keys()) + ["lgbm", "lstm"]
    ))

    def _make_meta(coin, vtype):
        # Paths: dari registry jika ada entry khusus, atau fallback ke main_version
        if vtype in versions_by_type:
            v = versions_by_type[vtype]
            p = v.get("paths", {})
        else:
            v = main_version
            p = main_version.get("paths", {})
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
            # lgbm/lstm standalone tidak pakai meta_learner / calibrator
            meta_learner_path     = p.get("meta") if vtype not in ("lgbm", "lstm") else None,
            calibrator_path       = p.get("calibrator") if vtype not in ("lgbm", "lstm") else None,
            inference_config_path = p.get("inference_config"),
            status                = "available",
            trained_at            = trained_at,
            evaluated_at          = trained_at,
        )

    for coin in Coin.query.all():
        created = []
        for vtype in all_types_to_seed:
            meta = _make_meta(coin, vtype)
            db.session.add(meta)
            created.append(meta)
        db.session.flush()

        # ModelSelection default → main_type (ensemble_v2) jika ada, fallback ke first available
        main_type = main_types[0] if main_types else created[0].model_type
        default = next(
            (m for m in created if m.model_type == main_type), created[0]
        )
        sel = ModelSelection(coin_id=coin.id, model_meta_id=default.id)
        db.session.add(sel)

    db.session.commit()
    logger.info(f"[auto_seed] Seeded {len(symbols)} coins × {len(all_types_to_seed)} model types (default={default.model_type})")


def _fix_stale_model_paths(app: Flask) -> None:
    """Sync ModelMeta path fields from current registry.json.
    
    Existing ModelMeta records may have stale paths if model_registry.json
    was updated after they were created (e.g., subfolder removed from paths).
    Only updates records whose current resolved path file does not exist.
    """
    import json
    import logging
    from pathlib import Path
    from app.extensions import db
    from app.models.model_meta import ModelMeta
    from app.services.model_registry import resolve_path

    logger = logging.getLogger(__name__)

    models_dir = Path(__file__).parent.parent / "models"
    registry_path = models_dir / "model_registry.json"
    if not registry_path.exists():
        return

    with open(registry_path) as f:
        registry = json.load(f)

    versions_by_type = {v["model_type"]: v for v in registry["versions"]}

    # Build path map per model_type
    reg_paths = {}
    for mt, v in versions_by_type.items():
        reg_paths[mt] = v.get("paths", {})

    # Fallback: main type (ensemble) for any model_type not in registry
    main_types = [t for t in versions_by_type if t not in ("lgbm", "lstm")]
    fallback_paths = reg_paths.get(main_types[0], {}) if main_types else {}

    updated = 0
    for meta in ModelMeta.query.all():
        p = reg_paths.get(meta.model_type, fallback_paths)
        if not p:
            continue

        dirty = False

        # model_path
        new_val = p.get("lstm") or p.get("lgbm")
        if new_val and meta.model_path and meta.model_path != new_val:
            if not resolve_path(meta.model_path).exists():
                meta.model_path = new_val
                dirty = True

        # scaler_path
        new_val = p.get("scaler")
        if new_val and meta.scaler_path and meta.scaler_path != new_val:
            if not resolve_path(meta.scaler_path).exists():
                meta.scaler_path = new_val
                dirty = True

        # meta_learner_path
        new_val = p.get("meta")
        if new_val and meta.meta_learner_path and meta.meta_learner_path != new_val:
            if not resolve_path(meta.meta_learner_path).exists():
                meta.meta_learner_path = new_val
                dirty = True

        # calibrator_path
        new_val = p.get("calibrator")
        if new_val and meta.calibrator_path and meta.calibrator_path != new_val:
            if not resolve_path(meta.calibrator_path).exists():
                meta.calibrator_path = new_val
                dirty = True

        # inference_config_path
        new_val = p.get("inference_config")
        if new_val and meta.inference_config_path and meta.inference_config_path != new_val:
            if not resolve_path(meta.inference_config_path).exists():
                meta.inference_config_path = new_val
                dirty = True

        if dirty:
            updated += 1

    if updated:
        db.session.commit()
        logger.info(f"[fix_stale_paths] {updated} ModelMeta records diperbaiki path-nya")


def _sync_ensemble_model_type(app: Flask) -> None:
    """Update existing ModelMeta records with old model_type='ensemble' to registry value.

    Migrasi satu-kali untuk memperbaiki record yang dibuat oleh _auto_seed()
    versi lama yang masih hardcode model_type='ensemble'.
    """
    import json
    import logging
    from pathlib import Path
    from app.extensions import db
    from app.models.model_meta import ModelMeta

    logger = logging.getLogger(__name__)

    models_dir = Path(__file__).parent.parent / "models"
    registry_path = models_dir / "model_registry.json"
    if not registry_path.exists():
        return

    with open(registry_path) as f:
        registry = json.load(f)

    versions_by_type = {v["model_type"]: v for v in registry["versions"]}

    # Cari record existing dengan model_type="ensemble" (versi lama)
    old_records = ModelMeta.query.filter_by(model_type="ensemble").all()
    if not old_records:
        return

    # Tentukan model_type ensemble baru dari registry (misal "ensemble_v2")
    new_types = [t for t in versions_by_type if t not in ("lgbm", "lstm")]
    if not new_types:
        return

    new_type = new_types[0]

    logger.info(
        f"[sync_ensemble] Migrasi {len(old_records)} record: "
        f"'ensemble' → '{new_type}'"
    )
    for meta in old_records:
        meta.model_type = new_type

    db.session.commit()
    logger.info(f"[sync_ensemble] Selesai. {len(old_records)} record di-update.")
