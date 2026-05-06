import csv, io
from flask import Blueprint, render_template, jsonify, request, Response
from sqlalchemy.orm import joinedload

bp = Blueprint("signals", __name__)

PAGE_SIZE = 50


@bp.get("/paper/signals")
def signals():
    from app.models.signal import Signal
    from app.models.coin import Coin

    page       = request.args.get("page", 1, type=int)
    symbol_raw = request.args.get("symbol", "")
    dir_raw    = request.args.get("direction", "")

    symbols   = [s.strip() for s in symbol_raw.split(",") if s.strip()]
    directions = [d.strip().upper() for d in dir_raw.split(",") if d.strip()]

    q = Signal.query.join(Coin, Signal.coin_id == Coin.id) \
                    .options(joinedload(Signal.model_meta))
    if symbols:
        q = q.filter(Coin.symbol.in_(symbols))
    if directions:
        q = q.filter(Signal.direction.in_(directions))

    pagination = (
        q.order_by(Signal.signal_time.desc())
        .paginate(page=page, per_page=PAGE_SIZE, error_out=False)
    )

    coins = Coin.query.filter_by(status="active").order_by(Coin.symbol).all()
    from app.services.model_registry import load_inference_config
    cfg = load_inference_config()
    leverage = float(cfg.get("risk", {}).get("leverage_recommended", 5.0))

    return render_template(
        "signals.html",
        pagination  = pagination,
        coins       = coins,
        symbols     = symbols,
        directions  = directions,
        leverage    = leverage,
    )


@bp.get("/paper/signals/export.csv")
def signals_export_csv():
    from app.models.signal import Signal
    from app.models.coin import Coin

    symbol_raw = request.args.get("symbol", "")
    dir_raw    = request.args.get("direction", "")

    symbols   = [s.strip() for s in symbol_raw.split(",") if s.strip()]
    directions = [d.strip().upper() for d in dir_raw.split(",") if d.strip()]

    q = Signal.query.join(Coin, Signal.coin_id == Coin.id) \
                    .options(joinedload(Signal.model_meta))
    if symbols:
        q = q.filter(Coin.symbol.in_(symbols))
    if directions:
        q = q.filter(Signal.direction.in_(directions))

    limit_raw = request.args.get("limit", 5000, type=int)
    limit = limit_raw if limit_raw > 0 else 100000

    signals = q.order_by(Signal.signal_time.desc()).limit(limit).all()

    from app.services.model_registry import load_inference_config
    from app.extensions import price_fmt
    cfg = load_inference_config()
    lev = float(cfg.get("risk", {}).get("leverage_recommended", 5.0))

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Time", "Coin", "Direction", "Confidence", "Model", "Entry", "TP", "TP%", "SL", "SL%", "ATR", "H4 High", "H4 Low"])
    for s in signals:
        e = s.entry_price
        tp_pct = f"{abs(s.tp_price - e) / e * 100 * lev:.1f}" if s.tp_price and e else ""
        sl_pct = f"{abs(s.sl_price - e) / e * 100 * lev:.1f}" if s.sl_price and e else ""
        w.writerow([
            s.signal_time.strftime("%Y-%m-%d %H:%M") if s.signal_time else "",
            s.coin.symbol,
            s.direction,
            f"{s.confidence:.4f}" if s.confidence else "",
            s.model_meta.model_type if s.model_meta else "",
            price_fmt(e) if e else "",
            price_fmt(s.tp_price) if s.tp_price else "",
            tp_pct,
            price_fmt(s.sl_price) if s.sl_price else "",
            sl_pct,
            price_fmt(s.atr_at_signal) if s.atr_at_signal else "",
            price_fmt(s.h4_swing_high) if s.h4_swing_high else "",
            price_fmt(s.h4_swing_low) if s.h4_swing_low else "",
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

    from app.services.model_registry import load_inference_config
    cfg = load_inference_config()
    lev = float(cfg.get("risk", {}).get("leverage_recommended", 5.0))

    entry = s.entry_price
    tp_pct = round(abs(s.tp_price - entry) / entry * 100 * lev, 2) if s.tp_price and entry else None
    sl_pct = round(abs(s.sl_price - entry) / entry * 100 * lev, 2) if s.sl_price and entry else None

    return jsonify({
        "id":            s.id,
        "direction":     s.direction,
        "confidence":    s.confidence,
        "model_type":    s.model_meta.model_type if s.model_meta else None,
        "entry":         entry,
        "tp":            s.tp_price,
        "sl":            s.sl_price,
        "tp_pct":        tp_pct,
        "sl_pct":        sl_pct,
        "atr":           s.atr_at_signal,
        "h4_swing_high": s.h4_swing_high,
        "h4_swing_low":  s.h4_swing_low,
        "time":          s.signal_time.isoformat() if s.signal_time else None,
        "snapshot":      snapshot,
    })
