"""
core/utils.py — Helper functions yang dipakai di seluruh pipeline
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Parquet I/O ──────────────────────────────────────────────────────────────

def save_df(df: pd.DataFrame, path: Path, logger: logging.Logger = None) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, str(path), compression="snappy")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Gagal simpan {path}: {e}")
        return False


def load_df(path: Path, logger: logging.Logger = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        if logger:
            logger.warning(f"File tidak ditemukan: {path}")
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        df = ensure_utc_index(df)
        return df
    except Exception as e:
        if logger:
            logger.error(f"Gagal load {path}: {e}")
        return None


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.sort_index(inplace=True)
    return df


# ─── Path helpers ─────────────────────────────────────────────────────────────

def get_raw_path(data_type: str, symbol: str, interval: str = None) -> Path:
    """
    Mapping data_type → path:
      klines       → data/raw/klines/{symbol}/{interval}_all.parquet
      funding_rate → data/raw/funding_rate/{symbol}_8h.parquet
      open_interest→ data/raw/open_interest/{symbol}_1h.parquet
      macro_btc_dom→ data/raw/macro/btc_dominance.parquet
      macro_fear_greed → data/raw/macro/fear_greed_index.parquet
    """
    RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
    if data_type == "klines":
        return RAW_DIR / "klines" / symbol / f"{interval}_all.parquet"
    elif data_type == "funding_rate":
        return RAW_DIR / "funding_rate" / f"{symbol}_8h.parquet"
    elif data_type == "open_interest":
        return RAW_DIR / "open_interest" / f"{symbol}_1h.parquet"
    elif data_type == "macro_btc_dom":
        return RAW_DIR / "macro" / "btc_dominance.parquet"
    elif data_type == "macro_fear_greed":
        return RAW_DIR / "macro" / "fear_greed_index.parquet"
    else:
        return RAW_DIR / data_type / f"{symbol}.parquet"

# Alias untuk backward compat
def get_filepath(data_type: str, symbol: str, interval: str = None) -> Path:
    return get_raw_path(data_type, symbol, interval)


# ─── Progress tracking ────────────────────────────────────────────────────────

def make_key(*parts: str) -> str:
    return "_".join(str(p) for p in parts if p)


def is_done(progress: dict, key: str) -> bool:
    return progress.get(key, False)


def mark_done(progress: dict, key: str) -> None:
    progress[key] = True


def load_progress(path: Path) -> dict:
    if path.exists():
        import json
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_progress(progress: dict, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2))


# ─── Timestamp helpers ────────────────────────────────────────────────────────

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ─── Chunk time range ─────────────────────────────────────────────────────────

INTERVAL_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
    "1w":  604_800_000,
}


def interval_to_ms(interval: str) -> int:
    return INTERVAL_MS.get(interval, 900_000)


def chunk_time_range(
    start_ms: int,
    end_ms: int,
    interval: str,
    limit: int,
) -> list[tuple[int, int]]:
    step = interval_to_ms(interval) * limit
    chunks = []
    current = start_ms
    while current < end_ms:
        chunk_end = min(current + step, end_ms)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


# ─── OHLCV validation ─────────────────────────────────────────────────────────

def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    logger: logging.Logger = None,
) -> dict:
    gaps      = 0
    ohlc_err  = 0
    zero_vol  = 0

    if len(df) > 1:
        freq_ms  = interval_to_ms(interval)
        ts_diff  = df.index.to_series().diff().dropna()
        expected = pd.Timedelta(milliseconds=freq_ms)
        gaps     = int((ts_diff > expected * 1.5).sum())

    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        ohlc_err = int(
            ((df["high"] < df["low"]) |
             (df["close"] > df["high"]) |
             (df["close"] < df["low"]) |
             (df["open"]  > df["high"]) |
             (df["open"]  < df["low"])).sum()
        )

    if "volume" in df.columns:
        zero_vol = int((df["volume"] == 0).sum())

    result = {
        "symbol":   symbol,
        "interval": interval,
        "rows":     len(df),
        "gaps":     gaps,
        "ohlc_err": ohlc_err,
        "zero_vol": zero_vol,
    }

    if logger:
        status = "OK " if gaps == 0 and ohlc_err == 0 else "WARN"
        logger.info(
            f"[{symbol}] [{status}] {interval:>3}  "
            f"rows={len(df):>7,}  gaps={gaps:>4}  "
            f"ohlc_err={ohlc_err:>4}  zero_vol={zero_vol:>4}"
        )

    return result


def print_summary(symbol: str, val_results: list, logger: logging.Logger = None) -> None:
    sep = "─" * 55
    lines = [
        f"\n{sep}",
        f"  Summary: {symbol}",
        f"{sep}",
    ]
    total_rows   = 0
    total_issues = 0
    for r in val_results:
        status = "OK " if r["gaps"] == 0 and r["ohlc_err"] == 0 else "WARN"
        lines.append(
            f"  [{status}] {r['interval']:>3}   "
            f"rows={r['rows']:>7,}  gaps={r['gaps']:>4}  "
            f"ohlc_err={r['ohlc_err']:>3}  zero_vol={r['zero_vol']:>4}"
        )
        total_rows   += r["rows"]
        total_issues += r["gaps"] + r["ohlc_err"]

    lines += [
        f"{sep}",
        f"  Total rows: {total_rows:,}  |  Total issues: {total_issues}",
        f"{sep}\n",
    ]
    msg = "\n".join(lines)
    if logger:
        logger.info(msg)
    else:
        print(msg)

# ─── Model Registry Helpers ──────────────────────────────────────────────────

def load_model_registry(registry_path: Path = None) -> dict:
    """Load model registry dari JSON file."""
    if registry_path is None:
        registry_path = Path("models/model_registry.json")
    
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry tidak ditemukan di {registry_path}. "
            f"Pastikan file ini ada atau jalankan training pipeline terlebih dahulu."
        )
    
    with open(registry_path) as f:
        return json.load(f)


def save_model_registry(registry: dict, registry_path: Path = None) -> None:
    """Simpan model registry ke JSON file."""
    if registry_path is None:
        registry_path = Path("models/model_registry.json")
    
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def update_model_metrics(
    model_name: str,
    registry_path: Path = None,
    **metrics,
) -> None:
    """
    Update metrics untuk model di registry.
    
    Args:
        model_name: Nama model (key di registry["models"])
        registry_path: Path ke model_registry.json (optional)
        **metrics: f1_macro, winrate, pnl_lev3x, status, dst
    
    Example:
        update_model_metrics(
            "ensemble_v2",
            f1_macro=0.58,
            winrate=0.61,
            status="active"
        )
    """
    registry = load_model_registry(registry_path)
    
    if model_name not in registry["models"]:
        raise KeyError(f"Model '{model_name}' tidak ditemukan di registry.")
    
    for key, value in metrics.items():
        if value is not None:
            registry["models"][model_name][key] = value
    
    save_model_registry(registry, registry_path)


def get_active_model_config(registry_path: Path = None) -> dict:
    """Ambil config model yang sedang aktif."""
    registry = load_model_registry(registry_path)
    active_name = registry["active"]
    return registry["models"][active_name]