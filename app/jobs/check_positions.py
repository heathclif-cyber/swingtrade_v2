"""
app/jobs/check_positions.py — Cek TP/SL semua open trades setiap 5 menit.

Fetch hanya current candle (1 bar) per koin yang punya open trade.
Tidak perlu engineer features — hanya butuh high/low/close terkini.
"""

import logging
import time

from flask import Flask

logger = logging.getLogger(__name__)


def _fetch_current_candle(client, symbol: str) -> dict | None:
    """Fetch 1 candle 1h terbaru — hanya butuh high/low/close."""
    import time as t
    now_ms = int(t.time() * 1000)
    start_ms = now_ms - 3_600_000  # 1 jam lalu
    raw = client.get_klines(
        symbol=symbol, interval="1h",
        start_time_ms=start_ms, end_time_ms=now_ms, limit=2
    )
    if not raw:
        return None
    last = raw[-1]
    return {
        "high":  float(last[2]),
        "low":   float(last[3]),
        "close": float(last[4]),
    }


def run(app: Flask) -> None:
    with app.app_context():
        from app.models.trade import Trade
        from app.models.coin import Coin
        from app.services.paper_trading import PaperTradingEngine
        from core.binance_client import BinanceClient
        import os

        open_trades = Trade.query.filter_by(status="open").all()
        if not open_trades:
            return

        coin_ids = {t.coin_id for t in open_trades}
        coins = {c.id: c.symbol for c in Coin.query.filter(Coin.id.in_(coin_ids)).all()}

        client = BinanceClient(
            base_url=os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com"),
            sleep_between=0.1,
        )

        current_candles = {}
        for coin_id, symbol in coins.items():
            candle = _fetch_current_candle(client, symbol)
            if candle:
                current_candles[coin_id] = candle
            time.sleep(0.1)

        engine  = PaperTradingEngine()
        closed  = engine.check_open_positions(current_candles)

        if closed:
            logger.info(f"[check_positions] Ditutup {len(closed)} trade(s)")
