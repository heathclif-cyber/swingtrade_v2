from flask import Blueprint, render_template, jsonify, abort, request

bp = Blueprint("coins", __name__)


@bp.get("/coins")
def coins():
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.models.performance_summary import PerformanceSummary
    from sqlalchemy import and_

    rows = (
        db.session.query(Coin, ModelMeta, ModelSelection, PerformanceSummary)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id, isouter=True)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id, isouter=True)
        .join(
            PerformanceSummary,
            and_(PerformanceSummary.coin_id == Coin.id, PerformanceSummary.period == "all"),
            isouter=True,
        )
        .filter(Coin.status == "active")
        .order_by(Coin.symbol)
        .all()
    )

    return render_template("coins.html", rows=rows)


@bp.get("/coins/<symbol>")
def coin_detail(symbol: str):
    from app.models.coin import Coin
    from app.models.signal import Signal
    from app.models.trade import Trade
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.models.performance_summary import PerformanceSummary

    coin = Coin.query.filter_by(symbol=symbol).first_or_404()

    signals = (
        Signal.query.filter_by(coin_id=coin.id)
        .order_by(Signal.signal_time.desc())
        .limit(20).all()
    )
    open_trade = Trade.query.filter_by(coin_id=coin.id, status="open").first()
    recent_trades = (
        Trade.query.filter_by(coin_id=coin.id, status="closed")
        .order_by(Trade.closed_at.desc())
        .limit(10).all()
    )
    perf = {
        p.period: p for p in PerformanceSummary.query.filter_by(coin_id=coin.id).all()
    }
    sel = ModelSelection.query.filter_by(coin_id=coin.id).first()
    meta = ModelMeta.query.get(sel.model_meta_id) if sel else None

    return render_template(
        "coin_detail.html",
        coin=coin, signals=signals, open_trade=open_trade,
        recent_trades=recent_trades, perf=perf, meta=meta,
        sel=sel,
    )
