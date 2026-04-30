"""
app/jobs/fetch_latest.py — Fetch klines terbaru dan simpan ke parquet cache.

Dipanggil scheduler setiap 15 menit.
Tidak melakukan inference — hanya refresh raw data di disk.
"""

import logging
import time
from datetime import datetime, timezone

from flask import Flask

logger = logging.getLogger(__name__)

SYMBOLS_DEFAULT = []


def run(app: Flask) -> None:
    with app.app_context():
        from app.models.coin import Coin
        from app.services.data_service import InferenceDataService
        from app.services.memory import check_and_free

        check_and_free()

        coins = Coin.query.filter_by(status="active").all()
        symbols = [c.symbol for c in coins] or SYMBOLS_DEFAULT

        if not symbols:
            logger.warning("[fetch_latest] Tidak ada koin aktif di DB")
            return

        svc = InferenceDataService()
        ok = fail = 0

        for symbol in symbols:
            try:
                df = svc.prepare_latest_features(symbol)
                if df is not None:
                    ok += 1
                else:
                    fail += 1
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"[fetch_latest] {symbol} error: {e}")
                fail += 1

        logger.info(f"[fetch_latest] Selesai: {ok} OK, {fail} gagal / {len(symbols)} koin")
