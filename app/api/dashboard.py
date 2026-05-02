from flask import Blueprint, render_template
from sqlalchemy import func

bp = Blueprint("dashboard", __name__)


@bp.get("/")
def index():
    return dashboard()


@bp.get("/dashboard")
def dashboard():
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.trade import Trade
    from app.models.signal import Signal
    from app.models.performance_summary import PerformanceSummary
    from app.models.model_selection import ModelSelection
    from app.models.model_meta import ModelMeta
    from app.services.memory import get_memory_status
    from app.jobs import get_scheduler

    active_coins  = Coin.query.filter_by(status="active").count()
    open_trades   = Trade.query.filter_by(status="open").all()
    latest_signals = (
        Signal.query
        .join(Coin, Signal.coin_id == Coin.id)
        .order_by(Signal.signal_time.desc())
        .limit(10)
        .all()
    )

    # 30d PnL total + Sharpe — hitung real-time dari Trade
    from datetime import timedelta
    import numpy as np
    from app.extensions import utcnow
    cutoff_30d   = utcnow() - timedelta(days=30)
    trades_30d   = Trade.query.filter_by(status="closed").filter(Trade.closed_at >= cutoff_30d).all()
    total_pnl_30d = sum(t.pnl_net or 0 for t in trades_30d)
    pnl_list_30d  = [t.pnl_net or 0 for t in trades_30d]
    if len(pnl_list_30d) > 1:
        std_30d        = float(np.std(pnl_list_30d))
        avg_sharpe_30d = float(np.mean(pnl_list_30d)) / std_30d if std_30d > 0 else 0.0
    else:
        avg_sharpe_30d = 0.0

    # Model performance per koin — sumber dari PerformanceSummary (paper trading)
    from sqlalchemy import and_
    model_rows = (
        db.session.query(Coin, ModelMeta, PerformanceSummary)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id)
        .join(
            PerformanceSummary,
            and_(PerformanceSummary.coin_id == Coin.id, PerformanceSummary.period == "all"),
            isouter=True,
        )
        .filter(Coin.status == "active")
        .order_by(Coin.symbol)
        .all()
    )

    mem  = get_memory_status()
    sched = get_scheduler()

    return render_template(
        "dashboard.html",
        active_coins    = active_coins,
        open_trades     = open_trades,
        latest_signals  = latest_signals,
        total_pnl_30d   = total_pnl_30d,
        avg_sharpe_30d  = avg_sharpe_30d,
        model_rows      = model_rows,
        memory          = mem,
        scheduler_ok    = sched is not None and sched.running,
    )
