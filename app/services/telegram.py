"""
app/services/telegram.py — Telegram notification service

Sends alerts for:
  - New trading signals (LONG/SHORT)
  - Trade opened
  - Trade closed (TP/SL hit)
  - Daily performance summary

Usage:
    from app.services.telegram import TelegramService
    tg = TelegramService()
    tg.send_signal_alert(signal, coin_symbol)
    tg.send_trade_opened(trade, coin_symbol)
    tg.send_trade_closed(trade, coin_symbol)
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

# WITA (Waktu Indonesia Tengah) = UTC+8
WITA_TZ = timezone(timedelta(hours=8))


def _format_wita(dt: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Convert UTC datetime to WITA and format as string."""
    if dt is None:
        return "N/A"
    if dt.tzinfo is not None:
        wita_dt = dt.astimezone(WITA_TZ)
    else:
        wita_dt = dt.replace(tzinfo=timezone.utc).astimezone(WITA_TZ)
    return wita_dt.strftime(fmt)

logger = logging.getLogger(__name__)


class TelegramService:
    """Telegram Bot API client for trading notifications."""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — notifications disabled")

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if result.get("ok"):
                logger.debug("[telegram] Message sent successfully")
                return True
            else:
                logger.error(f"[telegram] API error: {result.get('description')}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"[telegram] Request failed: {e}")
            return False

    def send_signal_alert(self, signal, coin_symbol: str) -> bool:
        """
        Send notification for a new trading signal.
        
        Args:
            signal: Signal ORM object
            coin_symbol: e.g. "ETHUSDT"
        
        Returns:
            True if sent successfully
        """
        if signal.direction == "FLAT":
            return False  # Don't notify for FLAT signals

        direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
        direction_text = "LONG 📈" if signal.direction == "LONG" else "SHORT 📉"
        
        model_label = signal.model_meta.model_type.upper() if signal.model_meta else "N/A"

        text = f"""
{direction_emoji} <b>New Signal: {direction_text}</b>

<b>Coin:</b> {coin_symbol}
<b>Model:</b> {model_label}
<b>Confidence:</b> {signal.confidence:.1%}
<b>Entry:</b> {signal.entry_price:.6f}
<b>ATR:</b> {signal.atr_at_signal:.6f}
"""
        
        if signal.tp_price:
            text += f"<b>TP:</b> {signal.tp_price:.6f}\n"
        if signal.sl_price:
            text += f"<b>SL:</b> {signal.sl_price:.6f}\n"
        
        text += f"\n<b>Time (WITA):</b> {_format_wita(signal.signal_time)}"
        text += f"\n<b>Timeframe:</b> {signal.timeframe or '1h'}"

        return self._send_message(text)

    def send_trade_opened(self, trade, coin_symbol: str) -> bool:
        """
        Send notification when a trade is opened.
        
        Args:
            trade: Trade ORM object
            coin_symbol: e.g. "ETHUSDT"
        
        Returns:
            True if sent successfully
        """
        direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
        
        text = f"""
{direction_emoji} <b>Trade Opened: {trade.direction}</b>

<b>Coin:</b> {coin_symbol}
<b>Entry:</b> {trade.entry_price:.6f}
<b>Size:</b> {trade.position_size:.4f}
<b>TP:</b> {trade.tp_price:.6f if trade.tp_price else 'N/A'}
<b>SL:</b> {trade.sl_price:.6f if trade.sl_price else 'N/A'}
<b>Leverage:</b> {trade.leverage}x

<b>Time (WITA):</b> {_format_wita(trade.opened_at)}
"""
        return self._send_message(text)

    def send_trade_closed(self, trade, coin_symbol: str) -> bool:
        """
        Send notification when a trade is closed.
        
        Args:
            trade: Trade ORM object
            coin_symbol: e.g. "ETHUSDT"
        
        Returns:
            True if sent successfully
        """
        pnl_emoji = "💰" if trade.pnl_net and trade.pnl_net > 0 else "📉"
        pnl_color = "profit" if trade.pnl_net and trade.pnl_net > 0 else "loss"
        
        text = f"""
{pnl_emoji} <b>Trade Closed: {trade.exit_reason or 'N/A'}</b>

<b>Coin:</b> {coin_symbol}
<b>Direction:</b> {trade.direction}
<b>Entry:</b> {trade.entry_price:.6f}
<b>Exit:</b> {trade.exit_price:.6f if trade.exit_price else 'N/A'}

<b>PnL:</b> {trade.pnl_net:.2f} ({trade.pnl_pct:.1f}%)
<b>Hold:</b> {trade.hold_bars} bars

<b>Closed (WITA):</b> {_format_wita(trade.closed_at)}
"""
        return self._send_message(text)

    def send_daily_summary(self, stats: dict) -> bool:
        """
        Send daily performance summary.
        
        Args:
            stats: dict with keys: total_trades, win_count, loss_count, 
                   total_pnl, win_rate, best_trade, worst_trade
        
        Returns:
            True if sent successfully
        """
        win_rate = stats.get("win_rate", 0)
        win_emoji = "🏆" if win_rate >= 0.6 else "📊"
        
        text = f"""
{win_emoji} <b>Daily Performance Summary</b>

<b>Trades:</b> {stats.get('total_trades', 0)} ({stats.get('win_count', 0)}W / {stats.get('loss_count', 0)}L)
<b>Win Rate:</b> {win_rate:.1%}
<b>Total PnL:</b> {stats.get('total_pnl', 0):.2f}

<b>Best Trade:</b> {stats.get('best_trade', 0):.2f}
<b>Worst Trade:</b> {stats.get('worst_trade', 0):.2f}

<b>Date (WITA):</b> {datetime.now(WITA_TZ).strftime('%Y-%m-%d')}
"""
        return self._send_message(text)

    def send_error_alert(self, error_msg: str, context: str = "") -> bool:
        """
        Send error notification for critical issues.
        
        Args:
            error_msg: Error message
            context: Additional context
        
        Returns:
            True if sent successfully
        """
        text = f"""
⚠️ <b>Error Alert</b>

<b>Context:</b> {context}
<b>Error:</b> <code>{error_msg[:500]}</code>

<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        return self._send_message(text)


# Singleton instance
_telegram_service: Optional[TelegramService] = None


def get_telegram_service() -> TelegramService:
    """Get or create TelegramService singleton."""
    global _telegram_service
    if _telegram_service is None:
        _telegram_service = TelegramService()
    return _telegram_service
