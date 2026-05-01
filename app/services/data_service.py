"""
app/services/data_service.py — Fetch data terbaru + engineer 85 fitur untuk inference.

InferenceDataService.prepare_latest_features(symbol, n_bars=250)
  → DataFrame (85 fitur + h4_swing_high + h4_swing_low), atau None jika data tidak cukup.

InferenceDataService.prepare_lstm_input(df, scaler)
  → ndarray (1, LSTM_SEQ_LEN, 85)

Fitur:
  - Multi-endpoint fallback via BinanceClient
  - Synthetic OI jika fapi tidak tersedia
  - Funding rate fallback ke 0.0 (netral)
"""

import logging
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.packages import urllib3  # type: ignore

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import BINANCE_BASE_URL
from app.services.config_loader import get_feature_cols, get_symbol_map, get_lstm_seq_len
from core.binance_client import BinanceClient
from core.features import engineer_features

logger = logging.getLogger(__name__)

INFERENCE_DIR = Path(__file__).parent.parent.parent / "data" / "inference"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

N_BARS        = 250          # minimum bars inference (LSTM_SEQ_LEN=32 + OI window=168 + buffer)
INTERVAL_MS   = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# Cache untuk macro data
_macro_cache: dict = {"data": None, "ts": 0.0}
MACRO_CACHE_TTL = 3600  # 1 jam


# ─── Binance raw klines parser (inline, tidak save ke disk) ──────────────────

def _parse_klines(raw: list, prefix: str) -> pd.DataFrame:
    """Parse Binance klines response ke DataFrame dengan prefixed columns."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "_ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df.index.name = "timestamp"
    float_cols = ["open", "high", "low", "close", "volume",
                  "taker_buy_volume", "taker_buy_quote_volume"]
    df[float_cols] = df[float_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.drop(columns=["close_time", "quote_volume", "_ignore"])
    df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns})
    return df


def _parse_funding(raw: list) -> pd.DataFrame:
    records = []
    for item in raw:
        ts = int(item["fundingTime"])
        records.append({
            "timestamp":    datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            "funding_rate": float(item.get("fundingRate", 0.0)),
        })
    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz=timezone.utc)
    return df.sort_index()


def _fetch_macro_latest() -> dict:
    """Fetch Fear & Greed + BTC dominance dengan cache. Return latest scalar values."""
    global _macro_cache
    
    # Check cache
    now_ts = time.time()
    if _macro_cache["data"] and (now_ts - _macro_cache["ts"] < MACRO_CACHE_TTL):
        return _macro_cache["data"]
    
    fg_val, btc_dom = 50.0, 55.0  # fallback defaults

    # Fear & Greed dari Alternative.me
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1, "format": "json"},
            timeout=10,
            verify=False,  # Bypass SSL issues
        )
        resp.raise_for_status()
        entries = resp.json().get("data", [])
        if entries:
            fg_val = float(entries[0]["value"])
            logger.info(f"[macro] Fear & Greed = {fg_val}")
    except Exception as e:
        logger.warning(f"[macro] Fear & Greed fetch gagal: {e}")

    # BTC dominance — coba CMC API jika ada key, fallback ke year-based
    try:
        cmc_key = os.environ.get("CMC_API_KEY", "")
        if cmc_key:
            resp = requests.get(
                "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                headers={"X-CMC_PRO_API_KEY": cmc_key, "Accept": "application/json"},
                timeout=10,
                verify=False,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                btc_dom = round(data.get("btc_dominance", 55.0), 1)
                logger.info(f"[macro] CMC BTC Dominance = {btc_dom}%")
    except Exception as e:
        logger.debug(f"[macro] CMC fetch gagal: {e}")
    
    # Fallback year-based jika CMC gagal
    if btc_dom == 55.0:
        year = datetime.now(timezone.utc).year
        btc_dom = {2024: 52.0, 2025: 55.0, 2026: 58.0}.get(year, 55.0)

    result = {"fear_greed": fg_val, "btc_dominance": btc_dom}
    _macro_cache = {"data": result, "ts": now_ts}
    return result


class InferenceDataService:
    def __init__(self):
        self._client = BinanceClient(
            base_url=os.getenv("BINANCE_BASE_URL", BINANCE_BASE_URL),
            sleep_between=0.1,
        )

    def prepare_latest_features(
        self, symbol: str, n_bars: int = N_BARS
    ) -> Optional[pd.DataFrame]:
        """
        Fetch klines 1h/4h/1d + funding + macro → engineer 85 fitur.
        Return None jika data tidak cukup untuk LSTM (< LSTM_SEQ_LEN bars).
        """
        try:
            return self._build_features(symbol, n_bars)
        except Exception as e:
            logger.error(f"[{symbol}] prepare_latest_features error: {e}", exc_info=True)
            return None

    @staticmethod
    def prepare_lstm_input(df: pd.DataFrame, scaler) -> np.ndarray:
        """Scale feature cols dan return sequence terakhir (1, seq_len, n_features)."""
        feature_cols = get_feature_cols()
        lstm_seq_len = get_lstm_seq_len()
        X = df[feature_cols].fillna(0).values
        X_scaled = scaler.transform(X)
        seq = X_scaled[-lstm_seq_len:]
        if len(seq) < lstm_seq_len:
            pad = np.zeros((lstm_seq_len - len(seq), X_scaled.shape[1]))
            seq = np.vstack([pad, seq])
        return seq[np.newaxis, :, :]

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_features(self, symbol: str, n_bars: int) -> Optional[pd.DataFrame]:
        now_ms = int(time.time() * 1000)

        # ── 1h base (n_bars candles) ─────────────────────────────────────────
        start_1h = now_ms - n_bars * INTERVAL_MS["1h"]
        raw_1h = self._client.get_klines(
            symbol=symbol, interval="1h",
            start_time_ms=start_1h, end_time_ms=now_ms, limit=n_bars
        )
        if not raw_1h or len(raw_1h) < get_lstm_seq_len() + 10:
            logger.warning(f"[{symbol}] 1h klines tidak cukup ({len(raw_1h or [])} bars)")
            return None

        df_1h = _parse_klines(raw_1h, "1h")
        df_1h = df_1h[~df_1h.index.duplicated(keep="first")].sort_index()

        # ── validasi kualitas 1h klines ──────────────────────────────────────
        n_invalid_price = (df_1h["1h_close"] <= 0).sum()
        if n_invalid_price > 0:
            logger.warning(f"[{symbol}] {n_invalid_price} bar dengan close price <= 0")
        if len(df_1h) > 1:
            gaps = df_1h.index.to_series().diff().dropna()
            large_gaps = gaps[gaps > pd.Timedelta(hours=2)]
            if not large_gaps.empty:
                logger.warning(
                    f"[{symbol}] {len(large_gaps)} gap timestamp di 1h klines "
                    f"(terbesar: {large_gaps.max()})"
                )

        # ── 4h context (aligned ke index H1 via ffill) ───────────────────────
        n_4h = n_bars // 4 + 20
        start_4h = now_ms - n_4h * INTERVAL_MS["4h"]
        raw_4h = self._client.get_klines(
            symbol=symbol, interval="4h",
            start_time_ms=start_4h, end_time_ms=now_ms, limit=n_4h
        )
        if raw_4h:
            df_4h = _parse_klines(raw_4h, "4h")
            df_4h = df_4h[~df_4h.index.duplicated(keep="first")].sort_index()
            df_4h_aligned = df_4h.reindex(
                df_4h.index.union(df_1h.index)
            ).ffill().reindex(df_1h.index)
        else:
            df_4h_aligned = pd.DataFrame(index=df_1h.index)

        # ── 1d context (aligned ke index H1 via ffill) ───────────────────────
        n_1d = n_bars // 24 + 5
        start_1d = now_ms - n_1d * INTERVAL_MS["1d"]
        raw_1d = self._client.get_klines(
            symbol=symbol, interval="1d",
            start_time_ms=start_1d, end_time_ms=now_ms, limit=n_1d
        )
        if raw_1d:
            df_1d = _parse_klines(raw_1d, "1d")
            df_1d = df_1d[~df_1d.index.duplicated(keep="first")].sort_index()
            df_1d_aligned = df_1d.reindex(
                df_1d.index.union(df_1h.index)
            ).ffill().reindex(df_1h.index)
        else:
            df_1d_aligned = pd.DataFrame(index=df_1h.index)

        # ── funding rate (forward-filled ke H1 index) ────────────────────────
        # Gunakan method get_funding_rate yang sudah handle FAPI block detection
        start_fr = now_ms - 90 * 24 * 3_600_000  # 90 hari
        raw_fr = self._client.get_funding_rate(
            symbol=symbol, start_time_ms=start_fr, end_time_ms=now_ms, limit=1000
        )
        if raw_fr:
            df_fr = _parse_funding(raw_fr)
            df_fr_aligned = df_fr.reindex(
                df_fr.index.union(df_1h.index)
            ).ffill().reindex(df_1h.index)
            df_fr_aligned = df_fr_aligned.rename(
                columns={"funding_rate": "funding_rate_fundingRate"}
            )
            logger.debug(f"[{symbol}] Funding rate OK: {len(df_fr)} records")
        else:
            # FAPI tidak tersedia → gunakan 0.0 (netral)
            # Nilai 0.0 aman untuk model — tidak mempengaruhi prediksi
            df_fr_aligned = pd.DataFrame(
                {"funding_rate_fundingRate": 0.0}, index=df_1h.index
            )
            logger.info(f"[{symbol}] Funding rate tidak tersedia, menggunakan 0.0 (netral)")

        # ── join semua ke base H1 ────────────────────────────────────────────
        df = pd.concat([df_1h, df_4h_aligned, df_1d_aligned, df_fr_aligned], axis=1)

        # ── macro (scalar fill) ──────────────────────────────────────────────
        macro = _fetch_macro_latest()
        df["btc_dominance"] = macro["btc_dominance"]
        df["fear_greed"]    = macro["fear_greed"]

        # ── engineer 85 fitur ────────────────────────────────────────────────
        symbol_id = get_symbol_map().get(symbol, 0)
        feat_df   = engineer_features(df, symbol=symbol, symbol_id=symbol_id, add_label=False)

        # ── long_short_ratio = 0 (by design — lihat arsitektur §5.2) ─────────
        feat_df["long_short_ratio"] = 0.0

        # ── distribusi fitur kunci (Fix 3 audit) ─────────────────────────────
        for feat in ("close", "atr_14_h1", "volume"):
            if feat in feat_df.columns:
                s = feat_df[feat].dropna()
                logger.debug(
                    f"[{symbol}] feat/{feat}: "
                    f"mean={s.mean():.4g} std={s.std():.4g} "
                    f"min={s.min():.4g} max={s.max():.4g} "
                    f"nan={feat_df[feat].isna().sum()}"
                )
        _feature_cols = get_feature_cols()
        nan_rate = feat_df[_feature_cols].isna().mean().mean()
        if nan_rate > 0.10:
            top_nan = (
                feat_df[_feature_cols].isna().mean()
                .pipe(lambda s: s[s > 0.10])
                .sort_values(ascending=False)
                .head(5)
            )
            logger.warning(
                f"[{symbol}] NaN rate tinggi ({nan_rate:.1%}) pada fitur — "
                f"top: {top_nan.to_dict()}"
            )

        # ── validasi kolom ────────────────────────────────────────────────────
        missing = [c for c in get_feature_cols() if c not in feat_df.columns]
        if missing:
            logger.error(f"[{symbol}] Missing features: {missing}")
            return None

        if len(feat_df) < get_lstm_seq_len():
            logger.warning(f"[{symbol}] Terlalu sedikit bars setelah engineering: {len(feat_df)}")
            return None

        # ── cache ke disk ─────────────────────────────────────────────────────
        cache_path = INFERENCE_DIR / f"{symbol}.parquet"
        feat_df.to_parquet(cache_path)

        logger.info(f"[{symbol}] Features OK: {len(feat_df)} bars × {len(feat_df.columns)} cols")
        return feat_df
