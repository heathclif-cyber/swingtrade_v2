"""
config.py — Sentralisasi semua parameter proyek
Edit file ini untuk mengubah parameter training, fetch, atau feature engineering.
"""

from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
LABEL_DIR  = DATA_DIR / "labeled"
MODEL_DIR  = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

# ─── Koin ────────────────────────────────────────────────────────────────────
TRAINING_COINS = [
    "SOLUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
]

NEW_COINS = [
    "TONUSDT", "ADAUSDT", "TRXUSDT", "1000SHIBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "SUIUSDT", "POLUSDT", "NEARUSDT",
    "1000PEPEUSDT", "TAOUSDT", "ARBUSDT",
]

ALL_COINS = TRAINING_COINS + NEW_COINS

SYMBOL_MAP = {coin: i for i, coin in enumerate(ALL_COINS)}

# ─── Periode Data ─────────────────────────────────────────────────────────────
TRAIN_START     = datetime(2022, 1, 1, tzinfo=timezone.utc)
TRAIN_END       = datetime(2025, 4, 1, tzinfo=timezone.utc)
NEW_COINS_START = datetime(2023, 4, 1, tzinfo=timezone.utc)
NEW_COINS_END   = datetime(2025, 4, 1, tzinfo=timezone.utc)

START_DATE = TRAIN_START
END_DATE   = TRAIN_END

# ─── Binance API ─────────────────────────────────────────────────────────────
BINANCE_BASE_URL       = "https://fapi.binance.com"
SLEEP_BETWEEN_REQUESTS = 0.12
SLEEP_ON_RATE_LIMIT    = 60.0
MAX_RETRIES            = 3
RETRY_BACKOFF_BASE     = 2.0

KLINE_LIMIT       = 1500
OI_LIMIT          = 500
FUNDING_LIMIT     = 1000
TAKER_RATIO_LIMIT = 500
LONG_SHORT_LIMIT  = 500

# ─── Timeframes ───────────────────────────────────────────────────────────────
KLINE_INTERVALS = ["1h", "4h", "1d"]

# ─── Feature Engineering ──────────────────────────────────────────────────────
TP_ATR_MULT      = 2.0
SL_ATR_MULT      = 1.0
MAX_HOLDING_BARS = 48    # bar H1 = 48 jam

# ── Swing-Based Labeling v3 ───────────────────────────────────────────────────
SWING_LABEL_MAX_HOLD = 48     # bar H1 = 48 jam
SWING_LABEL_MIN_RR   = 1.5
SWING_LABEL_MIN_TP   = 1.5
SWING_LABEL_MAX_SL   = 3.0
SWING_H4_LOOKBACK    = 5
SWING_ROLLING_BARS   = 24     # 24 jam rolling swing

# Volume Profile & FVG
VP_WINDOW       = 24
VP_BINS         = 50
FVG_MIN_GAP_ATR = 0.5
OB_LOOKBACK     = 30
SWING_LOOKBACK  = 5

# Synthetic OI (H1 Adjusted)
SYNTHETIC_OI_CVD_WINDOW  = 24
SYNTHETIC_OI_NORM_WINDOW = 168

# ─── Training & Purging ───────────────────────────────────────────────────────
N_FOLDS        = 8
PURGE_GAP_BARS = 5

# LightGBM Params
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}
LGBM_EARLY_STOPPING = 50

# LSTM Params
LSTM_SEQ_LEN    = 32
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 100
LSTM_PATIENCE   = 5
LSTM_BATCH_SIZE = 512
LSTM_LR         = 0.001

LABEL_MAP     = {"SHORT": 0, "FLAT": 1, "LONG": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = 3

# ─── ML Signal Thresholds ─────────────────────────────────────────────────────
CONFIDENCE_FULL = 0.75
CONFIDENCE_HALF = 0.60

# ─── Feature Columns v3 (H1 Base - 85 fitur) ─────────────────────────────────
FEATURE_COLS_V3 = [
    # OHLCV base
    "open", "high", "low", "close", "volume",

    # Volume flow
    "volume_delta", "cvd", "buy_volume", "sell_volume",

    # Market structure
    "MSB_BOS", "CHoCH", "bars_since_BOS",
    "FVG_up", "FVG_down", "Buy_Liq", "Sell_Liq", "SFP_sweep",

    # Open interest & funding
    "open_interest", "funding_rate",

    # EMA H1
    "ema_7_h1", "ema_21_h1", "ema_50_h1", "ema_200_h1",

    # EMA H4
    "ema_7_h4", "ema_21_h4", "ema_50_h4", "ema_200_h4",

    # Momentum
    "rsi_6", "stochrsi_k", "stochrsi_d",

    # ATR
    "atr_14_h1", "atr_14_h4",

    # Key levels
    "PDH", "PDL", "PWH", "PWL", "Fib_618", "Fib_786",

    # Volume profile
    "POC", "VAH", "VAL",

    # Macro
    "btc_dominance", "fear_greed", "market_session",

    # Returns & volume ratio
    "log_ret_1", "log_ret_5", "log_ret_20", "vol_ratio_20",

    # Time cyclical
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "time_to_funding_norm",

    # Long/short ratio
    "long_short_ratio",

    # Symbol encoding
    "symbol",

    # Swing structure (v2)
    "dist_swing_high", "dist_swing_low", "price_in_range", "swing_momentum",

    # Market regime (v2)
    "h4_trend", "trend_strength", "vol_regime",

    # Smart money v3
    "cvd_div_h4", "cvd_slope_h4",
    "vol_efficiency", "absorption_z",
    "funding_price_div",
    "rsi_h4", "rsi_divergence",
    "wyckoff_phase", "spring_upthrust",

    # Smart money v4 — OFI
    "ofi_raw", "ofi_acceleration", "ofi_z_score", "ofi_h4_delta",

    # Smart money v4 — VWDP
    "vwdp", "vwdp_smooth",

    # Smart money v4 — CVD hidden divergence
    "hidden_divergence", "cvd_momentum_adv",

    # Smart money v4 — Absorption at swing
    "absorption_at_swing",

    # Smart money v4 — VSA
    "spread_to_volume", "ultra_high_vol", "no_demand", "no_supply",
    "effort_vs_result",
]

# ─── Trading Simulation Parameters ───────────────────────────────────────────
MODAL_PER_TRADE            = 1000.0
LEVERAGE_SIM               = [3.0, 5.0]
FEE_PER_SIDE               = 0.0004
CONFIDENCE_THRESHOLD_ENTRY = 0.60
MIN_HOLD_BARS              = 2     # bar H1 = 4 jam minimum hold