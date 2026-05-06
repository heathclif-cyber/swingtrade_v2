"""Blueprint untuk halaman RL Training Data — ringkasan + preview parquet bulanan."""

import logging
from pathlib import Path

import pandas as pd
from flask import Blueprint, render_template, jsonify

from app.services.rl_data import RL_DATA_DIR

logger = logging.getLogger(__name__)
bp = Blueprint("rl_data", __name__)


@bp.get("/rl-data")
def rl_data_page():
    return render_template("rl_data.html")


@bp.get("/api/rl-data/scan")
def rl_data_scan():
    months = []
    total_signals = 0
    total_size_mb = 0.0
    preview_rows = []

    if RL_DATA_DIR.exists():
        for month_dir in sorted(RL_DATA_DIR.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue
            parquet_path = month_dir / "signals.parquet"
            if not parquet_path.exists():
                continue

            try:
                df = pd.read_parquet(parquet_path)
                row_count = len(df)
                size_mb = parquet_path.stat().st_size / (1024 * 1024)
                first_ts = df["signal_time"].min() if "signal_time" in df.columns else None
                last_ts = df["signal_time"].max() if "signal_time" in df.columns else None

                months.append({
                    "month": month_dir.name,
                    "signals": row_count,
                    "size_mb": round(size_mb, 2),
                    "first": str(first_ts)[:16] if first_ts is not None else "—",
                    "last": str(last_ts)[:16] if last_ts is not None else "—",
                })

                total_signals += row_count
                total_size_mb += size_mb

                # Preview dari bulan terkini saja
                if len(months) == 1:
                    preview_cols = [
                        "signal_id", "symbol", "direction", "confidence",
                        "entry_price", "atr_at_signal", "signal_time",
                    ]
                    available_cols = [c for c in preview_cols if c in df.columns]
                    latest = df.sort_values("signal_time", ascending=False).head(10)
                    for _, row in latest.iterrows():
                        preview_rows.append({
                            "signal_id": str(row.get("signal_id", "—")),
                            "signal_time": str(row.get("signal_time", "—"))[:19],
                            "symbol": str(row.get("symbol", "—")),
                            "direction": str(row.get("direction", "—")),
                            "confidence": round(float(row["confidence"]) * 100, 1) if row.get("confidence") is not None else None,
                            "entry_price": row.get("entry_price"),
                        })
            except Exception:
                logger.warning(f"[RL] Gagal baca {parquet_path}", exc_info=True)

    return jsonify({
        "months": months,
        "total_signals": total_signals,
        "total_size_mb": round(total_size_mb, 2),
        "total_months": len(months),
        "preview": preview_rows,
    })
