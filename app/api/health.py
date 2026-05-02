from flask import Blueprint, jsonify

bp = Blueprint("health", __name__)


@bp.get("/api/health")
def health():
    from app.jobs import get_scheduler
    from app.services.memory import get_memory_status
    from app.services.cache import model_cache
    from app.models.trade import Trade
    from app.models.signal import Signal
    from app.models.model_meta import ModelMeta
    from app.extensions import db
    from sqlalchemy import func

    sched   = get_scheduler()
    mem     = get_memory_status()
    open_tr = Trade.query.filter_by(status="open").count()
    signals = Signal.query.count()

    meta_counts = dict(
        db.session.query(ModelMeta.model_type, func.count(ModelMeta.id))
        .group_by(ModelMeta.model_type)
        .all()
    )

    signal_by_direction = dict(
        db.session.query(Signal.direction, func.count(Signal.id))
        .group_by(Signal.direction)
        .all()
    )

    return jsonify({
        "status":               "ok",
        "scheduler_running":    sched is not None and sched.running,
        "jobs":                 [j.id for j in sched.get_jobs()] if sched else [],
        "models_loaded":        list(model_cache._store.keys()),
        "active_trades":        open_tr,
        "total_signals":        signals,
        "signals_by_direction": signal_by_direction,
        "memory_mb":            mem["rss_mb"],
        "memory_pct":           mem["pct"],
        "memory_limit_mb":      mem["limit_mb"],
        "meta_counts":          meta_counts,
    })
