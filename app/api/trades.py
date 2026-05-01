from flask import Blueprint, render_template, jsonify, request, abort

bp = Blueprint("trades", __name__)

PAGE_SIZE = 50


@bp.get("/paper/trades")
def trades():
    from app.extensions import db
    from app.models.trade import Trade
    from app.models.coin import Coin
    from app.models.performance_summary import PerformanceSummary

    page      = request.args.get("page", 1, type=int)
    symbol    = request.args.get("symbol", "")
    status    = request.args.get("status", "")
    direction = request.args.get("direction", "")

    q = Trade.query.join(Coin, Trade.coin_id == Coin.id)
    if symbol:
        q = q.filter(Coin.symbol == symbol)
    if status:
        q = q.filter(Trade.status == status)
    if direction:
        q = q.filter(Trade.direction == direction)

    pagination = (
        q.order_by(Trade.opened_at.desc())
        .paginate(page=page, per_page=PAGE_SIZE, error_out=False)
    )

    # Summary cards — semua koin, period "all"
    perf_all = PerformanceSummary.query.filter_by(period="all").all()
    if perf_all:
        summary = {
            "total_trades":  sum(p.total_trades  or 0 for p in perf_all),
            "win_count":     sum(p.win_count     or 0 for p in perf_all),
            "total_pnl":     sum(p.total_pnl     or 0 for p in perf_all),
            "win_rate":      (
                sum(p.win_rate or 0 for p in perf_all) / len(perf_all)
                if perf_all else 0.0
            ),
            "profit_factor": (
                sum(p.profit_factor or 0 for p in perf_all) / len(perf_all)
                if perf_all else 0.0
            ),
            "sharpe_ratio":  (
                sum(p.sharpe_ratio or 0 for p in perf_all) / len(perf_all)
                if perf_all else 0.0
            ),
        }
    else:
        # Fallback if PerformanceSummary is empty (job hasn't run yet)
        trades_closed = Trade.query.filter_by(status="closed").all()
        total_trades = len(trades_closed)
        win_count = sum(1 for t in trades_closed if (t.pnl_net or 0) > 0)
        total_pnl = sum(t.pnl_net or 0 for t in trades_closed)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        gross_profit = sum(t.pnl_net or 0 for t in trades_closed if (t.pnl_net or 0) > 0)
        gross_loss = abs(sum(t.pnl_net or 0 for t in trades_closed if (t.pnl_net or 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        pnl_pcts = [t.pnl_pct or 0 for t in trades_closed]
        if len(pnl_pcts) > 1:
            import numpy as np
            mean_pct = np.mean(pnl_pcts)
            std_pct = np.std(pnl_pcts)
            sharpe_ratio = mean_pct / std_pct if std_pct > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        summary = {
            "total_trades": total_trades,
            "win_count": win_count,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
        }

    coins = Coin.query.filter_by(status="active").order_by(Coin.symbol).all()

    return render_template(
        "trades.html",
        pagination = pagination,
        summary    = summary,
        coins      = coins,
        symbol     = symbol,
        status     = status,
        direction  = direction,
    )


@bp.get("/paper/trades/<int:trade_id>")
def trade_detail(trade_id: int):
    from app.models.trade import Trade
    t = Trade.query.get_or_404(trade_id)
    return jsonify({
        "id":            t.id,
        "direction":     t.direction,
        "entry_price":   t.entry_price,
        "exit_price":    t.exit_price,
        "tp_price":      t.tp_price,
        "sl_price":      t.sl_price,
        "h4_swing_high": t.h4_swing_high,
        "h4_swing_low":  t.h4_swing_low,
        "pnl_net":       t.pnl_net,
        "pnl_pct":       t.pnl_pct,
        "status":        t.status,
        "exit_reason":   t.exit_reason,
        "hold_bars":     t.hold_bars,
        "opened_at":     t.opened_at.isoformat() if t.opened_at else None,
        "closed_at":     t.closed_at.isoformat() if t.closed_at else None,
    })


@bp.post("/paper/trades/<int:trade_id>/close")
def close_trade(trade_id: int):
    from app.extensions import db, utcnow
    from app.models.trade import Trade
    from app.models.coin import Coin
    from app.services.paper_trading import PaperTradingEngine
    from core.binance_client import BinanceClient
    import os, time

    trade = Trade.query.get_or_404(trade_id)
    if trade.status != "open":
        return jsonify({"error": "Trade sudah ditutup"}), 400

    # Fetch harga terkini
    coin   = Coin.query.get(trade.coin_id)
    client = BinanceClient(
        base_url=os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
    )
    now_ms = int(time.time() * 1000)
    raw = client.get_klines(
        symbol=coin.symbol, interval="1h",
        start_time_ms=now_ms - 3_600_000, end_time_ms=now_ms, limit=1
    )
    close_price = float(raw[-1][4]) if raw else trade.entry_price

    engine  = PaperTradingEngine()
    engine._close_trade(trade, close_price, "manual_close")
    db.session.commit()

    return jsonify({"status": "closed", "exit_price": close_price, "pnl_net": trade.pnl_net})
