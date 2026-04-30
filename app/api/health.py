from flask import Blueprint, jsonify

bp = Blueprint("health", __name__)


@bp.get("/api/health")
def health():
    from app.jobs import get_scheduler
    from app.services.memory import get_memory_status
    from app.services.cache import model_cache
    from app.models.trade import Trade
    from app.models.signal import Signal

    sched   = get_scheduler()
    mem     = get_memory_status()
    open_tr = Trade.query.filter_by(status="open").count()
    signals = Signal.query.count()

    return jsonify({
        "status":           "ok",
        "scheduler_running": sched is not None and sched.running,
        "jobs":             [j.id for j in sched.get_jobs()] if sched else [],
        "models_loaded":    list(model_cache._store.keys()),
        "active_trades":    open_tr,
        "total_signals":    signals,
        "memory_mb":        mem["rss_mb"],
        "memory_pct":       mem["pct"],
        "memory_limit_mb":  mem["limit_mb"],
    })


@bp.get("/api/stats")
def stats():
    from app.models.trade import Trade
    from app.models.signal import Signal
    from app.models.coin import Coin

    return jsonify({
        "active_coins":  Coin.query.filter_by(status="active").count(),
        "open_trades":   Trade.query.filter_by(status="open").count(),
        "closed_trades": Trade.query.filter_by(status="closed").count(),
        "total_signals": Signal.query.count(),
    })
