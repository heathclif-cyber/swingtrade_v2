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

    # 30d PnL total + Sharpe dari performance_summary
    perf_30d = PerformanceSummary.query.filter_by(period="30d").all()
    total_pnl_30d  = sum(p.total_pnl  or 0 for p in perf_30d)
    avg_sharpe_30d = (
        sum(p.sharpe_ratio or 0 for p in perf_30d) / len(perf_30d)
        if perf_30d else 0.0
    )

    # Model performance per koin
    model_rows = (
        db.session.query(Coin, ModelMeta)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id)
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
