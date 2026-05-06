import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.utils import save_df
from app.services.config_loader import get_feature_cols

logger = logging.getLogger(__name__)

RL_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rl_training"


def _monthly_path() -> Path:
    month_str = datetime.now(timezone.utc).strftime("%Y-%m")
    return RL_DATA_DIR / month_str / "signals.parquet"


def save_signal_features(
    symbol: str,
    signal_id: int,
    direction: str,
    confidence: float,
    entry_price: float,
    atr_at_signal: float,
    tp_price: float | None,
    sl_price: float | None,
    signal_time: datetime,
    features_df: pd.DataFrame,
) -> bool:
    try:
        feature_cols = get_feature_cols()
        last_row = features_df.iloc[-1]

        row = {col: last_row[col] for col in feature_cols}
        row.update({
            "signal_id":    signal_id,
            "symbol":       symbol,
            "direction":    direction,
            "confidence":   confidence,
            "entry_price":  entry_price,
            "atr_at_signal": atr_at_signal,
            "tp_price":     tp_price,
            "sl_price":     sl_price,
            "signal_time":  signal_time,
        })

        new_df = pd.DataFrame([row])

        path = _monthly_path()
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        return save_df(combined, path, logger)

    except Exception:
        logger.error(
            f"[RL] Gagal simpan signal_id={signal_id} {symbol}",
            exc_info=True,
        )
        return False
