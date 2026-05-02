"""
app/services/paper_trading.py — Engine paper trading incremental (bar per bar).

Alur:
  process_signal(signal_row, features_df) → buka posisi atau skip
  check_open_positions()                  → cek TP/SL semua open trades

TP/SL dihitung via calculate_tp_sl_swing() menggunakan H4 swing high/low.
Fallback ke fixed ATR multiplier dari inference_config["fallback_tp_sl"].
"""

import logging
import math
import os
from typing import Optional

import numpy as np

from app.extensions import db, utcnow
from app.models.trade import Trade
from app.models.coin import Coin
from app.services.model_registry import load_inference_config

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    def __init__(self):
        self._config  = load_inference_config()
        self._fallback = self._config.get("fallback_tp_sl", {})
        inf  = self._config.get("inference", {})
        risk = self._config.get("risk", {})
        vcb  = self._config.get("vcb", {})

        self._confidence_threshold  = inf.get("confidence_threshold_entry", 0.50)
        self._max_holding_bars      = inf.get("max_hold_bars", 48)
        self._modal_per_trade       = float(os.getenv("PAPER_MODAL")    or risk.get("modal_per_trade", 100.0))
        self._leverage              = float(os.getenv("PAPER_LEVERAGE") or risk.get("leverage_recommended", 5.0))
        self._fee_per_side          = risk.get("fee_per_side", 0.0004)
        self._same_dir_cooldown_hrs = self._config.get("same_dir_cooldown_hours", 4)
        self._vcb_enabled           = vcb.get("enabled", True)
        self._vcb_lookback_bars     = vcb.get("lookback_bars", 24)
        self._vcb_atr_multiplier    = vcb.get("atr_multiplier", 2.5)

        position_interval = int(os.getenv("POSITION_CHECK_INTERVAL_MINUTES", 5))
        logger.info(
            f"[PT] Init: max_hold_bars={self._max_holding_bars}h "
            f"(check_positions interval={position_interval}m, hold maks efektif={self._max_holding_bars}h)"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_signal(self, signal_row, features_df) -> Optional[Trade]:
        """
        Buka posisi baru jika semua kondisi terpenuhi.
        signal_row : ORM Signal instance
        features_df: DataFrame dari data_service (ada h4_swing_high/low)
        Return Trade atau None.
        """
        direction  = signal_row.direction
        confidence = signal_row.confidence or 0.0
        coin_id    = signal_row.coin_id
        entry_price = signal_row.entry_price
        atr         = signal_row.atr_at_signal or 0.0

        # ── 1. Confidence check ───────────────────────────────────────────────
        if confidence < self._confidence_threshold:
            logger.debug(f"[PT] Skip: confidence {confidence:.2f} < {self._confidence_threshold}")
            return None

        # ── 2. FLAT → tidak buka posisi ───────────────────────────────────────
        if direction not in ("LONG", "SHORT"):
            return None

        # ── 3. Cooldown: cek trade terakhir arah sama ─────────────────────────
        if self._is_cooldown_active(coin_id, direction):
            logger.info(f"[PT] Skip: cooldown aktif untuk {direction} coin_id={coin_id}")
            return None

        # ── 4. VCB (Volatility Circuit Breaker) ───────────────────────────────
        if self._vcb_enabled and features_df is not None:
            if self._circuit_breaker_active(features_df):
                logger.info(f"[PT] Skip: VCB aktif coin_id={coin_id}")
                return None

        # ── 5. Sudah ada open position untuk coin ini ─────────────────────────
        existing = Trade.query.filter_by(coin_id=coin_id, status="open").first()
        if existing:
            logger.debug(f"[PT] Skip: sudah ada open trade coin_id={coin_id}")
            return None

        # ── 6. Hitung TP/SL ───────────────────────────────────────────────────
        last_row = features_df.iloc[-1] if features_df is not None else None
        tp, sl = self._calculate_tp_sl(direction, entry_price, atr, last_row)
        if tp is None or sl is None:
            logger.warning(f"[PT] Skip: TP/SL tidak valid coin_id={coin_id}")
            return None

        # ── 7. Tiered position sizing by confidence ────────────────────────────
        confidence = signal_row.confidence or 0.0
        sizing_factor = 0.0  # Default: skip
        if confidence > (self._config.get("inference", {}).get("confidence_full_size", 0.75)):
            sizing_factor = 1.0       # Full size: 100 USD
        elif confidence > (self._config.get("inference", {}).get("confidence_half_size", 0.60)):
            sizing_factor = 0.5       # Half size: 50 USD

        if sizing_factor == 0.0:
            logger.info(f"[PT] Skip {direction} coin_id={coin_id} conf={confidence:.2f} < 0.60 — tier tidak terpenuhi")
            return None

        modal = round(self._modal_per_trade * sizing_factor, 2)
        lev   = self._leverage
        fee   = 2 * self._fee_per_side * self._modal_per_trade  # fee tetap dari modal penuh

        # ── 8. Buka trade ──────────────────────────────────────────────────────

        sh_val = last_row.get("h4_swing_high") if last_row is not None else None
        sl_val = last_row.get("h4_swing_low") if last_row is not None else None
        
        sh = float(sh_val) if sh_val is not None and not math.isnan(sh_val) else None
        sl_lvl = float(sl_val) if sl_val is not None and not math.isnan(sl_val) else None

        trade = Trade(
            signal_id     = signal_row.id,
            coin_id       = coin_id,
            direction     = direction,
            entry_price   = entry_price,
            tp_price      = tp,
            sl_price      = sl,
            h4_swing_high = sh    if sh     and sh     > 0 else None,
            h4_swing_low  = sl_lvl if sl_lvl and sl_lvl > 0 else None,
            quantity      = modal,
            leverage      = lev,
            fee_total     = fee,
            status        = "open",
            opened_at     = utcnow(),
            hold_bars     = 0,
        )
        db.session.add(trade)
        db.session.commit()
        logger.info(
            f"[PT] OPEN {direction} coin_id={coin_id} conf={confidence:.2f} "
            f"size={sizing_factor:.0%} modal={modal:.1f} entry={entry_price:.4f} "
            f"TP={tp:.4f} SL={sl:.4f}"
        )

        # Kirim notifikasi Telegram
        try:
            from app.services.telegram import get_telegram_service
            coin = Coin.query.get(coin_id)
            symbol = coin.symbol if coin else f"coin_{coin_id}"
            tg = get_telegram_service()
            tg.send_trade_opened(trade, symbol)
        except Exception as e:
            logger.warning(f"[PT] Gagal kirim notifikasi Telegram: {e}")

        return trade

    def check_open_positions(self, current_candles: dict) -> list[Trade]:
        """
        Cek semua open trades terhadap current candle.
        current_candles: {coin_id: {"high": float, "low": float, "close": float}}
        Return list trade yang ditutup.
        """
        open_trades = Trade.query.filter_by(status="open").all()
        closed = []

        for trade in open_trades:
            candle = current_candles.get(trade.coin_id)
            if candle is None:
                continue

            high  = candle["high"]
            low   = candle["low"]
            close = candle["close"]

            exit_price  = None
            exit_reason = None

            if trade.direction == "LONG":
                if trade.tp_price and high >= trade.tp_price:
                    exit_price, exit_reason = trade.tp_price, "tp_hit"
                elif trade.sl_price and low <= trade.sl_price:
                    exit_price, exit_reason = trade.sl_price, "sl_hit"
            else:  # SHORT
                if trade.tp_price and low <= trade.tp_price:
                    exit_price, exit_reason = trade.tp_price, "tp_hit"
                elif trade.sl_price and high >= trade.sl_price:
                    exit_price, exit_reason = trade.sl_price, "sl_hit"

            # Hitung hold_bars dari selisih waktu aktual (satuan candle 1h)
            # Bug sebelumnya: diincrement tiap job run (5 menit) → efektif 2 jam.
            # Fix: hitung jam yang berlalu sejak trade dibuka → efektif 24 jam.
            opened_at = trade.opened_at
            if opened_at.tzinfo is None:
                from datetime import timezone as _tz
                opened_at = opened_at.replace(tzinfo=_tz.utc)
            elapsed_hours = (utcnow() - opened_at).total_seconds() / 3600
            trade.hold_bars = max(0, int(elapsed_hours))
            if exit_price is None and trade.hold_bars >= self._max_holding_bars:
                exit_price, exit_reason = close, "time_exit"

            if exit_price is not None:
                self._close_trade(trade, exit_price, exit_reason)
                try:
                    db.session.commit()
                    closed.append(trade)
                except Exception as e:
                    db.session.rollback()
                    logger.error(f"[PT] Gagal commit penutupan trade id={trade.id}: {e}", exc_info=True)
                    continue

        # Commit sisa update hold_bars untuk trade yang tidak ditutup
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"[PT] Gagal commit hold_bars update: {e}", exc_info=True)

        return closed

    # ── TP/SL calculation ─────────────────────────────────────────────────────

    def _calculate_tp_sl(
        self,
        direction: str,
        entry: float,
        atr: float,
        last_row,
    ) -> tuple[Optional[float], Optional[float]]:
        """Swing-based TP/SL murni. Fallback ke fixed ATR jika swing tidak tersedia."""
        if last_row is not None and atr > 0:
            sh_val = last_row.get("h4_swing_high")
            sl_val = last_row.get("h4_swing_low")
            sh = float(sh_val) if sh_val is not None and not math.isnan(sh_val) else 0.0
            sl_lvl = float(sl_val) if sl_val is not None and not math.isnan(sl_val) else 0.0

            # Langsung terapkan Swing Level tanpa validasi ATR Mult
            if direction == "LONG" and sh > entry and sl_lvl < entry:
                return sh, sl_lvl

            if direction == "SHORT" and sl_lvl < entry and sh > entry:
                return sl_lvl, sh

        # Fallback ke fixed ATR (nilai fallback ini sekarang terhubung dinamis ke JSON config)
        if atr <= 0:
            return None, None
            
        tp_mult = self._fallback.get("tp_atr_mult", 3.0) 
        sl_mult = self._fallback.get("sl_atr_mult", 1.5)
        
        if direction == "LONG":
            return entry + tp_mult * atr, entry - sl_mult * atr
        else:
            return entry - tp_mult * atr, entry + sl_mult * atr

    # ── Circuit Breaker ───────────────────────────────────────────────────────

    def _circuit_breaker_active(self, features_df) -> bool:
        """True jika ATR current > VCB_ATR_MULTIPLIER × ATR mean (24 bars)."""
        try:
            atr_series = features_df["atr_14_h1"].dropna()
            if len(atr_series) < self._vcb_lookback_bars:
                return False
            recent   = atr_series.iloc[-self._vcb_lookback_bars:]
            atr_mean = recent.mean()
            atr_now  = atr_series.iloc[-1]
            return float(atr_now) > self._vcb_atr_multiplier * float(atr_mean)
        except Exception:
            return False

    # ── Cooldown check ────────────────────────────────────────────────────────

    def _is_cooldown_active(self, coin_id: int, direction: str) -> bool:
        from datetime import timedelta
        cutoff = utcnow() - timedelta(hours=self._same_dir_cooldown_hrs)
        recent = Trade.query.filter(
            Trade.coin_id   == coin_id,
            Trade.direction == direction,
            Trade.status    == "closed",
            Trade.closed_at >= cutoff,
        ).first()
        return recent is not None

    # ── PnL calculation ───────────────────────────────────────────────────────

    def _close_trade(self, trade: Trade, exit_price: float, reason: str) -> None:
        direction_sign = 1 if trade.direction == "LONG" else -1
        qty = trade.quantity or self._modal_per_trade
        lev = trade.leverage or self._leverage

        pnl_pct   = direction_sign * (exit_price - trade.entry_price) / trade.entry_price
        pnl_gross = pnl_pct * qty * lev
        pnl_net   = pnl_gross - (trade.fee_total or 0)

        trade.exit_price  = exit_price
        trade.exit_reason = reason
        trade.pnl_gross   = round(pnl_gross, 4)
        trade.pnl_net     = round(pnl_net, 4)
        trade.pnl_pct     = round(pnl_pct * 100, 2)
        trade.status      = "closed"
        trade.closed_at   = utcnow()

        logger.info(
            f"[PT] CLOSE {trade.direction} id={trade.id} "
            f"reason={reason} pnl_net={pnl_net:.2f}"
        )

        # Kirim notifikasi Telegram
        try:
            from app.services.telegram import get_telegram_service
            coin = Coin.query.get(trade.coin_id)
            symbol = coin.symbol if coin else f"coin_{trade.coin_id}"
            tg = get_telegram_service()
            tg.send_trade_closed(trade, symbol)
        except Exception as e:
            logger.warning(f"[PT] Gagal kirim notifikasi Telegram: {e}")
