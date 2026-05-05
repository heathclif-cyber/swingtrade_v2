import csv, io
from flask import Blueprint, render_template, jsonify, abort, request, Response

bp = Blueprint("coins", __name__)


@bp.get("/coins")
def coins():
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.models.trade import Trade
    from collections import defaultdict
    from types import SimpleNamespace
    import numpy as np

    rows = (
        db.session.query(Coin, ModelMeta, ModelSelection)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id, isouter=True)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id, isouter=True)
        .filter(Coin.status == "active")
        .order_by(Coin.symbol)
        .all()
    )

    # Per-coin stats real-time dari Trade — satu query untuk semua koin
    trades_by_coin = defaultdict(list)
    for t in Trade.query.filter_by(status="closed").order_by(Trade.coin_id, Trade.closed_at).all():
        trades_by_coin[t.coin_id].append(t.pnl_net or 0)

    perf_map = {}
    for coin_id, pnls in trades_by_coin.items():
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        gross_loss = abs(sum(losses))
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float((peak - equity).max()) if n > 0 else 0.0
        perf_map[coin_id] = SimpleNamespace(
            total_trades=n,
            win_count=len(wins),
            win_rate=round(len(wins) / n, 4),
            total_pnl=round(total_pnl, 4),
            max_drawdown=round(max_dd, 4),
        )

    return render_template("coins.html", rows=rows, perf_map=perf_map)


@bp.get("/coins/export.csv")
def coins_export_csv():
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.models.trade import Trade
    from collections import defaultdict
    from types import SimpleNamespace
    import numpy as np

    rows = (
        db.session.query(Coin, ModelMeta, ModelSelection)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id, isouter=True)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id, isouter=True)
        .filter(Coin.status == "active")
        .order_by(Coin.symbol)
        .all()
    )

    trades_by_coin = defaultdict(list)
    for t in Trade.query.filter_by(status="closed").order_by(Trade.coin_id, Trade.closed_at).all():
        trades_by_coin[t.coin_id].append(t.pnl_net or 0)

    perf_map = {}
    for coin_id, pnls in trades_by_coin.items():
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        gross_loss = abs(sum(losses))
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float((peak - equity).max()) if n > 0 else 0.0
        perf_map[coin_id] = SimpleNamespace(
            total_trades=n, win_count=len(wins),
            win_rate=round(len(wins) / n, 4),
            total_pnl=round(total_pnl, 4),
            max_drawdown=round(max_dd, 4),
        )

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Symbol", "Model", "Win Rate (%)", "Max DD ($)", "Trades", "Total PnL ($)", "Last Signal", "Status"])
    for coin, meta, sel in rows:
        perf = perf_map.get(coin.id)
        w.writerow([
            coin.symbol,
            meta.model_type if meta else "",
            f"{perf.win_rate * 100:.1f}" if perf and perf.win_rate else "",
            f"{perf.max_drawdown:.2f}" if perf and perf.max_drawdown else "",
            perf.total_trades if perf and perf.total_trades else "",
            f"{perf.total_pnl:+.2f}" if perf and perf.total_pnl is not None else "",
            coin.last_signal_at.strftime("%Y-%m-%d %H:%M") if coin.last_signal_at else "",
            coin.status,
        ])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=coins.csv"})


@bp.get("/coins/<symbol>")
def coin_detail(symbol: str):
    from app.models.coin import Coin
    from app.models.signal import Signal
    from app.models.trade import Trade
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    import numpy as np

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

    # Live Performance — real-time per period dari Trade
    def _calc(trades_list):
        pnls = [t.pnl_net or 0 for t in trades_list]
        n = len(pnls)
        if n == 0:
            return None
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_loss = abs(sum(losses))
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float((peak - equity).max()) if n > 0 else 0.0
        gross_profit = sum(wins)
        pf = gross_profit / gross_loss if gross_loss > 0 else (0.0 if gross_profit == 0 else 99.0)
        return SimpleNamespace(
            total_trades=n,
            win_count=len(wins),
            win_rate=round(len(wins) / n, 4),
            total_pnl=round(sum(pnls), 4),
            profit_factor=round(pf, 4),
            max_drawdown=round(max_dd, 4),
        )

    now = datetime.now(timezone.utc)
    perf = {}
    for period_name, days in [("7d", 7), ("30d", 30), ("all", None)]:
        q = Trade.query.filter_by(coin_id=coin.id, status="closed")
        if days:
            q = q.filter(Trade.closed_at >= now - timedelta(days=days))
        stats = _calc(q.all())
        if stats:
            perf[period_name] = stats

    sel = ModelSelection.query.filter_by(coin_id=coin.id).first()
    meta = ModelMeta.query.get(sel.model_meta_id) if sel else None

    # Performance snapshot history — 14 baris terakhir untuk trend
    from app.models.performance_summary import PerformanceSummary
    snap_history = (
        PerformanceSummary.query
        .filter_by(coin_id=coin.id, period="30d")
        .order_by(PerformanceSummary.snapshot_at.desc())
        .limit(14).all()
    )

    return render_template(
        "coin_detail.html",
        coin=coin, signals=signals, open_trade=open_trade,
        recent_trades=recent_trades, perf=perf, meta=meta,
        sel=sel, snap_history=snap_history,
    )


@bp.get("/api/equity-curve/<symbol>")
def api_equity_curve_coin(symbol: str):
    """Return daily equity curve data untuk Chart.js — per koin."""
    from datetime import timedelta
    from collections import defaultdict
    import numpy as np
    from app.extensions import db, utcnow
    from app.models.trade import Trade
    from app.models.coin import Coin

    coin = Coin.query.filter_by(symbol=symbol).first_or_404()

    days = 60
    cutoff = utcnow() - timedelta(days=days)
    trades = (
        Trade.query.filter_by(coin_id=coin.id, status="closed")
        .filter(Trade.closed_at >= cutoff)
        .order_by(Trade.closed_at)
        .all()
    )

    daily = defaultdict(list)
    for t in trades:
        day = t.closed_at.strftime("%m-%d")
        daily[day].append(t.pnl_net or 0)

    labels = list(daily.keys())
    daily_pnl = [sum(daily[d]) for d in labels]
    equity = np.cumsum(daily_pnl).tolist()

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = [round((p - e) * 100 / (p + 0.01), 1) for p, e in zip(peak, equity)]

    return jsonify({
        "labels": labels,
        "equity": equity,
        "daily_pnl": daily_pnl,
        "drawdown_pct": drawdown,
    })
