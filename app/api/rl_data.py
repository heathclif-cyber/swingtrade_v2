"""Blueprint untuk halaman RL Training Data — ringkasan + preview dari DB."""

import logging
import os

from flask import Blueprint, render_template, jsonify
from sqlalchemy import func

from app.extensions import db
from app.models.signal import Signal
from app.models.coin import Coin

logger = logging.getLogger(__name__)
bp = Blueprint("rl_data", __name__)

IS_SQLITE = "sqlite" in os.environ.get("DATABASE_URL", "")


@bp.get("/rl-data")
def rl_data_page():
    return render_template("rl_data.html")


@bp.get("/api/rl-data/scan")
def rl_data_scan():
    total_signals = db.session.query(func.count(Signal.id)).scalar() or 0
    total_months = 0
    total_size_mb = 0.0
    months = []
    preview_rows = []

    if total_signals > 0:
        # Format bulan: TO_CHAR (PG) vs STRFTIME (SQLite)
        if IS_SQLITE:
            month_expr = func.strftime("%Y-%m", Signal.signal_time)
        else:
            month_expr = func.to_char(Signal.signal_time, "YYYY-MM")

        # Ringkasan per bulan
        month_rows = (
            db.session.query(
                month_expr.label("month"),
                func.count(Signal.id).label("cnt"),
                func.min(Signal.signal_time).label("first"),
                func.max(Signal.signal_time).label("last"),
            )
            .group_by("month")
            .order_by(month_expr.desc())
            .all()
        )
        total_months = len(month_rows)
        # Estimasi: rata-rata 3 KB per baris (85 fitur + metadata JSON)
        total_size_mb = round(total_signals * 3 / 1024, 2)

        for row in month_rows:
            months.append({
                "month": row.month,
                "signals": row.cnt,
                "size_mb": round(row.cnt * 3 / 1024, 2),
                "first": str(row.first)[:16] if row.first else "—",
                "last": str(row.last)[:16] if row.last else "—",
            })

        # Preview 10 sinyal terakhir
        latest = (
            Signal.query
            .join(Coin, Signal.coin_id == Coin.id)
            .options(db.joinedload(Signal.coin))
            .order_by(Signal.signal_time.desc())
            .limit(10)
            .all()
        )
        for s in latest:
            preview_rows.append({
                "signal_id": str(s.id),
                "signal_time": str(s.signal_time)[:19] if s.signal_time else "—",
                "symbol": s.coin.symbol if s.coin else "—",
                "direction": s.direction,
                "confidence": round((s.confidence or 0) * 100, 1),
                "entry_price": s.entry_price,
            })

    return jsonify({
        "months": months,
        "total_signals": total_signals,
        "total_size_mb": total_size_mb,
        "total_months": total_months,
        "preview": preview_rows,
    })
