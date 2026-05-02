"""
app/jobs/__init__.py — Inisialisasi APScheduler.

KRITIS: Hanya boleh diinisialisasi SEKALI di dalam satu worker.
        Procfile wajib --workers 1 untuk mencegah duplikasi job.
"""

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from flask import Flask

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler | None:
    return _scheduler


def init_scheduler(app: Flask) -> None:
    global _scheduler

    # Guard: jangan inisialisasi ulang (e.g. Flask reloader spawn 2 proses)
    if _scheduler is not None and _scheduler.running:
        logger.warning("[scheduler] Sudah berjalan — skip init")
        return

    fetch_interval    = int(os.getenv("FETCH_INTERVAL_MINUTES", 15))
    signal_interval   = int(os.getenv("SIGNAL_INTERVAL_HOURS", 4))
    position_interval = int(os.getenv("POSITION_CHECK_INTERVAL_MINUTES", 5))
    metrics_interval  = 6  # jam, tidak dikonfigurasi via env karena jarang diubah

    _scheduler = BackgroundScheduler(timezone="Asia/Makassar", daemon=True)

    from app.jobs import fetch_latest, check_positions, generate_signals, update_metrics

    # fetch_latest — setiap 15 menit, mulai segera
    _scheduler.add_job(
        func     = lambda: fetch_latest.run(app),
        trigger  = IntervalTrigger(minutes=fetch_interval),
        id       = "fetch_latest",
        name     = "Fetch Latest Klines",
        replace_existing = True,
    )

    # check_positions — setiap 5 menit, mulai segera
    _scheduler.add_job(
        func     = lambda: check_positions.run(app),
        trigger  = IntervalTrigger(minutes=position_interval),
        id       = "check_positions",
        name     = "Check Open Positions",
        replace_existing = True,
    )

    # generate_signals — setiap jam lewat 5 menit (HH:05)
    # Beri waktu 5 menit setelah jam penuh agar candle 1h terbaru sudah settle dan fetch_latest sempat meng-cache data
    _scheduler.add_job(
        func          = lambda: generate_signals.run(app),
        trigger       = CronTrigger(minute=5),
        id            = "generate_signals",
        name          = "Generate Trading Signals",
        replace_existing = True,
    )
    logger.info(f"[scheduler] generate_signals dijadwalkan setiap jam lewat 5 menit (HH:05)")

    # update_metrics — setiap 6 jam, mulai segera
    _scheduler.add_job(
        func     = lambda: update_metrics.run(app),
        trigger  = IntervalTrigger(hours=metrics_interval),
        id       = "update_metrics",
        name     = "Update Performance Metrics",
        replace_existing = True,
    )

    _scheduler.start()
    logger.info(
        f"[scheduler] Berjalan — "
        f"fetch={fetch_interval}m, signals={signal_interval}h, "
        f"positions={position_interval}m, metrics={metrics_interval}h"
    )
