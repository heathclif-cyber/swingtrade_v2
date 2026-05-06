from flask import Blueprint, render_template, jsonify
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

    # Model performance per koin — ambil snapshot terbaru dari PerformanceSummary
    from sqlalchemy import and_, func
    latest_snap = (
        db.session.query(
            PerformanceSummary.coin_id,
            PerformanceSummary.period,
            func.max(PerformanceSummary.snapshot_at).label("max_snap"),
        )
        .group_by(PerformanceSummary.coin_id, PerformanceSummary.period)
        .subquery()
    )
    model_rows = (
        db.session.query(Coin, ModelMeta, PerformanceSummary)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id)
        .join(
            latest_snap,
            and_(latest_snap.c.coin_id == Coin.id, latest_snap.c.period == "all"),
            isouter=True,
        )
        .join(
            PerformanceSummary,
            and_(
                PerformanceSummary.coin_id == latest_snap.c.coin_id,
                PerformanceSummary.period == latest_snap.c.period,
                PerformanceSummary.snapshot_at == latest_snap.c.max_snap,
            ),
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


@bp.get("/api/equity-curve")
def api_equity_curve():
    """Return daily equity curve data untuk Chart.js — SEMUA koin."""
    from datetime import timedelta
    from collections import defaultdict
    import numpy as np
    from app.extensions import db, utcnow
    from app.models.trade import Trade

    days = 60
    cutoff = utcnow() - timedelta(days=days)
    trades = (
        Trade.query.filter_by(status="closed")
        .filter(Trade.closed_at >= cutoff)
        .order_by(Trade.closed_at)
        .all()
    )

    # Group PnL by date
    daily: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        day = t.closed_at.strftime("%m-%d")
        daily[day].append(t.pnl_net or 0)

    # Fill all days in range (even days without trades = PnL 0)
    labels = []
    for i in range(days - 1, -1, -1):
        d = (utcnow() - timedelta(days=i)).strftime("%m-%d")
        labels.append(d)

    daily_pnl = [sum(daily.get(d, [0])) for d in labels]
    equity = np.cumsum(daily_pnl).tolist()

    # Daily win rate & rolling 10-day
    daily_wr = []
    for d in labels:
        pnls = daily.get(d, [])
        wins = sum(1 for p in pnls if p > 0)
        daily_wr.append(round(wins / len(pnls) * 100, 1) if pnls else 0)

    rolling_wr = []
    window = 10
    for i in range(len(labels)):
        wr_vals = daily_wr[max(0, i - window + 1):i + 1]
        rolling_wr.append(round(sum(wr_vals) / len(wr_vals), 1) if wr_vals else 0)

    return jsonify({
        "labels": labels,
        "equity": equity,
        "daily_pnl": daily_pnl,
        "rolling_wr": rolling_wr,
    })
