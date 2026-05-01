"""
app/services/config_loader.py — Singleton loader untuk inference_config.json.

Source of truth untuk semua parameter model-spesifik yang di-generate
oleh pipeline pemodelan (bukan dikode keras di config.py).

  get_feature_cols()   → list[str]   kolom fitur dari feature_cols_v2.json
  get_lstm_seq_len()   → int         panjang sequence LSTM
  get_label_map_inv()  → dict        {0: "SHORT", 1: "FLAT", 2: "LONG"}
  get_symbol_map()     → dict        {"BTCUSDT": 0, ...}
  get_n_features()     → int         len(feature_cols)
  get_inference_config() → dict      seluruh inference_config.json

Semua hasil di-cache setelah baca pertama. Panggil reload_cache()
untuk force-reload saat replace model tanpa restart app.

Parameter infrastruktur (API key, DB URL, Telegram token, dll.)
tetap dibaca dari .env — TIDAK dipindahkan ke sini.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent.parent
REGISTRY = ROOT / "models" / "model_registry.json"

_config_cache: Optional[dict] = None
_feature_cols_cache: Optional[list] = None


# ── Internal ──────────────────────────────────────────────────────────────────

def _active_config_path() -> Path:
    with open(REGISTRY) as f:
        registry = json.load(f)
    for v in registry["versions"]:
        if v.get("status") in ("available", "active"):
            return ROOT / v["paths"]["inference_config"]
    raise RuntimeError("[config_loader] Tidak ada versi model aktif di registry")


# ── Public API ────────────────────────────────────────────────────────────────

def get_inference_config() -> dict:
    """Load dan cache seluruh inference_config.json versi aktif."""
    global _config_cache
    if _config_cache is None:
        path = _active_config_path()
        with open(path) as f:
            _config_cache = json.load(f)
        logger.info(
            f"[config_loader] Loaded {path.name} "
            f"(v={_config_cache.get('model_version', '?')})"
        )
    return _config_cache


def get_feature_cols() -> list[str]:
    """Return list nama kolom fitur dari model_files.features."""
    global _feature_cols_cache
    if _feature_cols_cache is None:
        cfg = get_inference_config()
        features_file = cfg.get("model_files", {}).get("features", "feature_cols_v2.json")
        config_path = _active_config_path()
        # Cari di direktori versioned dulu, fallback ke models/
        features_path = config_path.parent / features_file
        if not features_path.exists():
            features_path = ROOT / "models" / features_file
        with open(features_path) as f:
            _feature_cols_cache = json.load(f)
        logger.info(f"[config_loader] Feature cols: {len(_feature_cols_cache)} kolom")
    return _feature_cols_cache


def get_lstm_seq_len() -> int:
    """Return panjang sequence LSTM (default 32)."""
    return get_inference_config().get("inference", {}).get("seq_len", 32)


def get_label_map_inv() -> dict[int, str]:
    """Return mapping index → label: {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}."""
    raw = get_inference_config().get("inference", {}).get(
        "label_map_inv", {"0": "SHORT", "1": "FLAT", "2": "LONG"}
    )
    return {int(k): v for k, v in raw.items()}


def get_symbol_map() -> dict[str, int]:
    """Return mapping simbol → integer ID yang dipakai saat training."""
    return get_inference_config().get("symbol_map", {})


def get_n_features() -> int:
    """Return jumlah fitur = len(get_feature_cols())."""
    return len(get_feature_cols())


def reload_cache() -> None:
    """Force reload semua cache — gunakan setelah replace file model."""
    global _config_cache, _feature_cols_cache
    _config_cache = None
    _feature_cols_cache = None
    logger.info("[config_loader] Cache di-reset, akan reload pada akses berikutnya")
