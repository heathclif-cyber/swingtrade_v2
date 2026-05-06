"""Blueprint untuk halaman RL Training Data — ringkasan + preview + backup DB."""

import json
import logging
import os
from datetime import datetime, timezone

from flask import Blueprint, Response, render_template, jsonify
from sqlalchemy import func

from app.extensions import db
from app.models.signal import Signal
from app.models.trade import Trade
from app.models.coin import Coin
from app.models.performance_summary import PerformanceSummary

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


@bp.get("/api/backup/download")
def backup_download():
    # Signals
    signals = []
    for s in Signal.query.order_by(Signal.id.asc()).all():
        signals.append({
            "id": s.id,
            "symbol": s.coin.symbol if s.coin else "—",
            "direction": s.direction,
            "confidence": s.confidence,
            "entry_price": s.entry_price,
            "tp_price": s.tp_price,
            "sl_price": s.sl_price,
            "atr_at_signal": s.atr_at_signal,
            "h4_swing_high": s.h4_swing_high,
            "h4_swing_low": s.h4_swing_low,
            "signal_time": str(s.signal_time) if s.signal_time else None,
            "feature_snapshot": json.loads(s.feature_snapshot) if s.feature_snapshot else None,
        })

    # Trades
    trades = []
    for t in Trade.query.order_by(Trade.id.asc()).all():
        trades.append({
            "id": t.id,
            "symbol": t.coin.symbol if t.coin else "—",
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "tp_price": t.tp_price,
            "sl_price": t.sl_price,
            "h4_swing_high": t.h4_swing_high,
            "h4_swing_low": t.h4_swing_low,
            "pnl_net": t.pnl_net,
            "pnl_pct": t.pnl_pct,
            "quantity": t.quantity,
            "exit_reason": t.exit_reason,
            "status": t.status,
            "opened_at": str(t.opened_at) if t.opened_at else None,
            "closed_at": str(t.closed_at) if t.closed_at else None,
        })

    # Performance Summary
    performance = []
    for p in PerformanceSummary.query.order_by(PerformanceSummary.id.asc()).all():
        performance.append({
            "id": p.id,
            "symbol": p.coin.symbol if p.coin else "—",
            "period": p.period,
            "total_trades": p.total_trades,
            "win_rate": p.win_rate,
            "total_pnl": p.total_pnl,
            "max_drawdown": p.max_drawdown,
            "sharpe_ratio": p.sharpe_ratio,
            "snapshot_at": str(p.snapshot_at) if p.snapshot_at else None,
        })

    backup = {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "totals": {
            "signals": len(signals),
            "trades": len(trades),
            "performance_snapshots": len(performance),
        },
        "signals": signals,
        "trades": trades,
        "performance": performance,
    }

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Response(
        json.dumps(backup, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=backup_swingtrade_{date_str}.json"
        },
    )
