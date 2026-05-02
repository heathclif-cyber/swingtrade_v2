import csv, io
from flask import Blueprint, render_template, jsonify, request, abort, Response

bp = Blueprint("trades", __name__)

PAGE_SIZE = 50


@bp.get("/paper/trades")
def trades():
    from app.extensions import db
    from app.models.trade import Trade
    from app.models.coin import Coin

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

    # Summary cards — hitung real-time dari Trade agar selalu sinkron
    import numpy as np
    trades_closed = Trade.query.filter_by(status="closed").all()
    total_trades  = len(trades_closed)
    win_count     = sum(1 for t in trades_closed if (t.pnl_net or 0) > 0)
    total_pnl     = sum(t.pnl_net or 0 for t in trades_closed)
    win_rate      = win_count / total_trades if total_trades > 0 else 0.0

    gross_profit  = sum(t.pnl_net or 0 for t in trades_closed if (t.pnl_net or 0) > 0)
    gross_loss    = abs(sum(t.pnl_net or 0 for t in trades_closed if (t.pnl_net or 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (0.0 if gross_profit == 0 else 99.0)

    pnl_pcts      = [t.pnl_pct or 0 for t in trades_closed]
    if len(pnl_pcts) > 1:
        std_pct    = float(np.std(pnl_pcts))
        sharpe_ratio = float(np.mean(pnl_pcts)) / std_pct if std_pct > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    summary = {
        "total_trades":  total_trades,
        "win_count":     win_count,
        "total_pnl":     total_pnl,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio":  sharpe_ratio,
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


@bp.get("/paper/trades/export.csv")
def trades_export_csv():
    from app.extensions import db
    from app.models.trade import Trade
    from app.models.coin import Coin

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

    trades = q.order_by(Trade.opened_at.desc()).limit(5000).all()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Opened", "Closed", "Coin", "Direction", "Entry", "Exit",
                 "TP", "SL", "H4 High", "H4 Low", "Qty", "Leverage",
                 "PnL ($)", "PnL (%)", "Exit Reason", "Hold Bars", "Status"])
    for t in trades:
        w.writerow([
            t.opened_at.strftime("%Y-%m-%d %H:%M") if t.opened_at else "",
            t.closed_at.strftime("%Y-%m-%d %H:%M") if t.closed_at else "",
            t.coin.symbol,
            t.direction,
            f"{t.entry_price:.4f}",
            f"{t.exit_price:.4f}" if t.exit_price else "",
            f"{t.tp_price:.4f}" if t.tp_price else "",
            f"{t.sl_price:.4f}" if t.sl_price else "",
            f"{t.h4_swing_high:.4f}" if t.h4_swing_high else "",
            f"{t.h4_swing_low:.4f}" if t.h4_swing_low else "",
            f"{t.quantity:.4f}" if t.quantity else "",
            t.leverage if t.leverage else "",
            f"{t.pnl_net:+.2f}" if t.pnl_net is not None else "",
            f"{t.pnl_pct:+.1f}" if t.pnl_pct is not None else "",
            t.exit_reason or "",
            t.hold_bars if t.hold_bars is not None else "",
            t.status,
        ])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=trades.csv"})


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


@bp.post("/paper/trades/<int:trade_id>/delete")
def delete_trade(trade_id: int):
    from app.extensions import db
    from app.models.trade import Trade

    trade = Trade.query.get_or_404(trade_id)
    db.session.delete(trade)
    db.session.commit()
    return jsonify({"status": "deleted", "id": trade_id})


@bp.post("/paper/trades/delete-all")
def delete_all_trades():
    from app.extensions import db
    from app.models.trade import Trade

    deleted = Trade.query.delete()
    db.session.commit()
    return jsonify({"status": "deleted", "count": deleted})
