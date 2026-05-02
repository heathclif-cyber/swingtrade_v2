import csv, io
from flask import Blueprint, render_template, jsonify, request, Response

bp = Blueprint("signals", __name__)

PAGE_SIZE = 50


@bp.get("/paper/signals")
def signals():
    from app.models.signal import Signal
    from app.models.coin import Coin

    page      = request.args.get("page", 1, type=int)
    symbol    = request.args.get("symbol", "")
    direction = request.args.get("direction", "")

    q = Signal.query.join(Coin, Signal.coin_id == Coin.id)
    if symbol:
        q = q.filter(Coin.symbol == symbol)
    if direction:
        q = q.filter(Signal.direction == direction)
    else:
        q = q.filter(Signal.direction.in_(["LONG", "SHORT"]))

    pagination = (
        q.order_by(Signal.signal_time.desc())
        .paginate(page=page, per_page=PAGE_SIZE, error_out=False)
    )

    coins = Coin.query.filter_by(status="active").order_by(Coin.symbol).all()

    return render_template(
        "signals.html",
        pagination  = pagination,
        coins       = coins,
        symbol      = symbol,
        direction   = direction,
    )


@bp.get("/paper/signals/export.csv")
def signals_export_csv():
    from app.models.signal import Signal
    from app.models.coin import Coin

    symbol    = request.args.get("symbol", "")
    direction = request.args.get("direction", "")

    q = Signal.query.join(Coin, Signal.coin_id == Coin.id)
    if symbol:
        q = q.filter(Coin.symbol == symbol)
    if direction:
        q = q.filter(Signal.direction == direction)
    else:
        q = q.filter(Signal.direction.in_(["LONG", "SHORT"]))

    signals = q.order_by(Signal.signal_time.desc()).limit(5000).all()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Time", "Coin", "Direction", "Confidence", "Entry", "TP", "SL", "ATR", "H4 High", "H4 Low"])
    for s in signals:
        w.writerow([
            s.signal_time.strftime("%Y-%m-%d %H:%M") if s.signal_time else "",
            s.coin.symbol,
            s.direction,
            f"{s.confidence:.4f}" if s.confidence else "",
            f"{s.entry_price:.4f}" if s.entry_price else "",
            f"{s.tp_price:.4f}" if s.tp_price else "",
            f"{s.sl_price:.4f}" if s.sl_price else "",
            f"{s.atr_at_signal:.4f}" if s.atr_at_signal else "",
            f"{s.h4_swing_high:.4f}" if s.h4_swing_high else "",
            f"{s.h4_swing_low:.4f}" if s.h4_swing_low else "",
        ])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=signals.csv"})


@bp.get("/paper/signals/<int:signal_id>")
def signal_detail(signal_id: int):
    from app.models.signal import Signal
    import json

    s = Signal.query.get_or_404(signal_id)
    snapshot = {}
    if s.feature_snapshot:
        try:
            snapshot = json.loads(s.feature_snapshot)
        except Exception:
            pass

    return jsonify({
        "id":            s.id,
        "direction":     s.direction,
        "confidence":    s.confidence,
        "entry":         s.entry_price,
        "tp":            s.tp_price,
        "sl":            s.sl_price,
        "atr":           s.atr_at_signal,
        "h4_swing_high": s.h4_swing_high,
        "h4_swing_low":  s.h4_swing_low,
        "time":          s.signal_time.isoformat() if s.signal_time else None,
        "snapshot":      snapshot,
    })
