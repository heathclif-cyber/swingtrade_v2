"""
app/jobs/update_metrics.py — Refresh performance_summary + scan model baru dari registry.

Dipanggil setiap 6 jam.
Juga membersihkan signal lama (> 90 hari atau > 10K rows).
"""

import logging
from datetime import datetime, timedelta, timezone

from flask import Flask

logger = logging.getLogger(__name__)

PERIODS = {
    "7d":  7,
    "30d": 30,
    "all": None,
}
MAX_SIGNAL_ROWS = 10_000
SIGNAL_RETENTION_DAYS = 90


def run(app: Flask) -> None:
    with app.app_context():
        _refresh_performance_summary()
        _scan_new_models()
        _rotate_old_signals()


def _refresh_performance_summary() -> None:
    from app.extensions import db, utcnow
    from app.models.coin import Coin
    from app.models.trade import Trade
    from app.models.performance_summary import PerformanceSummary
    import numpy as np

    coins = Coin.query.filter_by(status="active").all()

    for coin in coins:
        for period, days in PERIODS.items():
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=days)
                if days else None
            )
            q = Trade.query.filter_by(coin_id=coin.id, status="closed")
            if cutoff:
                q = q.filter(Trade.closed_at >= cutoff)
            trades = q.all()

            if not trades:
                continue

            pnls    = [t.pnl_net for t in trades if t.pnl_net is not None]
            wins    = [p for p in pnls if p > 0]
            losses  = [p for p in pnls if p <= 0]
            n       = len(pnls)
            win_cnt = len(wins)
            loss_cnt= len(losses)
            wr      = win_cnt / n if n else 0.0

            total_pnl = sum(pnls)
            avg_pnl   = total_pnl / n if n else 0.0

            gross_profit = sum(wins)  if wins   else 0.0
            gross_loss   = abs(sum(losses)) if losses else 0.0
            pf = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)

            # Sharpe (simplified, anggap risk-free = 0)
            if n > 1:
                std = float(np.std(pnls))
                sharpe = (avg_pnl / std) if std > 0 else 0.0
            else:
                sharpe = 0.0

            # Max drawdown dari equity curve
            equity = np.cumsum(pnls)
            peak   = np.maximum.accumulate(equity)
            dd     = peak - equity
            max_dd = float(dd.max()) if len(dd) > 0 else 0.0

            summary = PerformanceSummary.query.filter_by(
                coin_id=coin.id, period=period
            ).first()

            if not summary:
                summary = PerformanceSummary(coin_id=coin.id, period=period)
                db.session.add(summary)

            summary.total_trades  = n
            summary.win_count     = win_cnt
            summary.loss_count    = loss_cnt
            summary.win_rate      = round(wr, 4)
            summary.total_pnl     = round(total_pnl, 4)
            summary.avg_pnl       = round(avg_pnl, 4)
            summary.profit_factor = round(pf, 4)
            summary.sharpe_ratio  = round(sharpe, 4)
            summary.max_drawdown  = round(max_dd, 4)
            summary.updated_at    = utcnow()

    db.session.commit()
    logger.info("[update_metrics] performance_summary diperbarui")


def _scan_new_models() -> None:
    """
    Cek model_registry.json untuk versi baru yang belum ada di MODEL_META.
    Versi terdeteksi baru jika inference_config_path berbeda dari yang ada di DB.
    Validasi n_features dan file exists sebelum upsert.
    """
    from app.extensions import db, utcnow
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.services.model_registry import load_registry, resolve_path
    from app.services.config_loader import get_n_features

    N_FEATURES = get_n_features()
    versions = load_registry()

    for version in versions:
        model_type = version.get("model_type", "ensemble")
        run_id_label = version.get("run_id", "?")  # hanya untuk logging

        if version.get("n_features") != N_FEATURES:
            logger.warning(
                f"[scan_models] {model_type} (run={run_id_label}) n_features={version.get('n_features')} "
                f"!= {N_FEATURES} — skip"
            )
            continue

        paths = version.get("paths", {})
        lstm_path = resolve_path(paths.get("lstm", ""))
        if not lstm_path.exists():
            logger.debug(f"[scan_models] {model_type} (run={run_id_label}) file tidak ada — skip")
            continue

        new_config_path = paths.get("inference_config", "")

        # Cek apakah sudah up-to-date: bandingkan inference_config_path di DB
        sample_coin = Coin.query.filter_by(status="active").first()
        if sample_coin:
            up_to_date = ModelMeta.query.filter_by(
                coin_id=sample_coin.id,
                model_type=model_type,
                inference_config_path=new_config_path,
            ).first()
            if up_to_date:
                continue

        # Versi baru terdeteksi — upsert untuk semua koin aktif
        coins = Coin.query.filter_by(status="active").all()
        bs = version.get("backtest_summary", {})
        upserted = 0
        for coin in coins:
            meta = ModelMeta.query.filter_by(coin_id=coin.id, model_type=model_type).first()
            if meta:
                meta.n_features            = N_FEATURES
                meta.win_rate              = bs.get("mean_winrate", meta.win_rate)
                meta.max_drawdown          = bs.get("mean_drawdown_lev5x", bs.get("mean_drawdown_lev3x", meta.max_drawdown))
                meta.model_path            = paths.get("lstm") or paths.get("lgbm")
                meta.scaler_path           = paths.get("scaler")
                meta.meta_learner_path     = paths.get("meta")
                meta.calibrator_path       = paths.get("calibrator")
                meta.inference_config_path = new_config_path
                meta.status                = "available"
                meta.trained_at            = utcnow()
                meta.evaluated_at          = utcnow()
            else:
                meta = ModelMeta(
                    coin_id               = coin.id,
                    model_type            = model_type,
                    n_features            = N_FEATURES,
                    win_rate              = bs.get("mean_winrate"),
                    max_drawdown          = bs.get("mean_drawdown_lev5x", bs.get("mean_drawdown_lev3x")),
                    model_path            = paths.get("lstm") or paths.get("lgbm"),
                    scaler_path           = paths.get("scaler"),
                    meta_learner_path     = paths.get("meta"),
                    calibrator_path       = paths.get("calibrator"),
                    inference_config_path = new_config_path,
                    status                = "available",
                    trained_at            = utcnow(),
                    evaluated_at          = utcnow(),
                )
                db.session.add(meta)
            upserted += 1

        db.session.commit()
        logger.info(f"[scan_models] {model_type} (run={run_id_label}) — {upserted} ModelMeta diperbarui")


def _rotate_old_signals() -> None:
    """Hapus signal > 90 hari atau jika total > 10K rows."""
    from app.extensions import db
    from app.models.signal import Signal
    from sqlalchemy import func

    cutoff = datetime.now(timezone.utc) - timedelta(days=SIGNAL_RETENTION_DAYS)
    old_count = Signal.query.filter(Signal.created_at < cutoff).count()
    if old_count > 0:
        Signal.query.filter(Signal.created_at < cutoff).delete()
        db.session.commit()
        logger.info(f"[rotate_signals] Hapus {old_count} signal > {SIGNAL_RETENTION_DAYS} hari")

    total = Signal.query.count()
    if total > MAX_SIGNAL_ROWS:
        excess = total - MAX_SIGNAL_ROWS
        oldest = Signal.query.order_by(Signal.created_at.asc()).limit(excess).all()
        for s in oldest:
            db.session.delete(s)
        db.session.commit()
        logger.info(f"[rotate_signals] Hapus {excess} signal oldest (total melebihi {MAX_SIGNAL_ROWS})")
