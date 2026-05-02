"""
app/services/inference.py — Load model dan predict signal untuk satu koin.

Mendukung tiga mode (dari inference_config.json):
  lstm     → LSTM forward → softmax
  lgbm     → predict_proba
  ensemble → LGBM + LSTM → MetaLearner → Calibrator

Semua model di-lazy-load dan di-cache via TTLCache.
Guard wajib: model_meta.n_features == get_n_features() dari config_loader.
"""

import gc
import logging
import math
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from app.services.config_loader import get_feature_cols, get_label_map_inv, get_n_features
from core.models import TradingLSTM, load_lstm, ProbabilityCalibrator
from app.services.cache import model_cache
from app.services.model_registry import resolve_path, load_inference_config

logger = logging.getLogger(__name__)

N_FEATURES = get_n_features()


# ─── Model bundle ────────────────────────────────────────────────────────────

class _ModelBundle:
    """Container untuk semua model artifacts satu versi."""
    __slots__ = ("lstm", "lgbm", "scaler", "meta", "calibrator", "inference_config")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ─── InferenceService ─────────────────────────────────────────────────────────

class InferenceService:
    def __init__(self, model_meta_row):
        """
        model_meta_row : ORM ModelMeta instance (atau duck-typed dict)
        """
        self._meta   = model_meta_row
        self._config = load_inference_config()
        self._validate_n_features()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        symbol: str,
        features_df,          # DataFrame dari data_service
        model_type: str = "ensemble",
    ) -> Optional[dict]:
        """
        Return {direction, confidence, proba, entry_price, atr_value}
        atau None jika confidence < threshold atau cold start.
        """
        if features_df is None or len(features_df) == 0:
            return None

        try:
            bundle = self._get_bundle(model_type)
            proba  = self._run_model(features_df, bundle, model_type)  # ndarray (3,)

            threshold = self._config.get("inference", {}).get("confidence_threshold_entry", 0.60)
            pred_idx  = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
            direction  = get_label_map_inv()[pred_idx]

            logger.debug(
                f"[{symbol}] proba raw: SHORT={proba[0]:.4f} FLAT={proba[1]:.4f} LONG={proba[2]:.4f} "
                f"→ max={confidence:.4f} ({direction}) threshold={threshold:.2f} "
                f"→ {'PASS' if confidence >= threshold else 'FLAT (di bawah threshold)'}"
            )

            if confidence < threshold:
                direction = "FLAT"

            last = features_df.iloc[-1]
            h4_sh_val = last.get("h4_swing_high")
            h4_sl_val = last.get("h4_swing_low")
            h4_sh = float(h4_sh_val) if h4_sh_val is not None and not math.isnan(h4_sh_val) else 0.0
            h4_sl = float(h4_sl_val) if h4_sl_val is not None and not math.isnan(h4_sl_val) else 0.0

            return {
                "direction":   direction,
                "confidence":  round(confidence, 4),
                "proba":       proba.tolist(),
                "entry_price": float(last["close"]),
                "atr_value":   float(last.get("atr_14_h1", 0.0)),
                "h4_swing_high": h4_sh,
                "h4_swing_low":  h4_sl,
            }

        except Exception as e:
            logger.error(f"[{symbol}] inference error: {e}", exc_info=True)
            return None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _get_bundle(self, model_type: str) -> _ModelBundle:
        bundle = model_cache.get(model_type)
        if bundle is not None:
            return bundle

        logger.info(f"[inference] Loading {model_type} model...")
        bundle = self._load_bundle(model_type)
        model_cache.put(model_type, bundle)
        return bundle

    def _load_bundle(self, model_type: str) -> _ModelBundle:
        meta = self._meta
        cfg  = self._config.get("model_architecture", {})

        lstm_model = scaler = lgbm_model = meta_learner = calibrator = None

        if (model_type == "lstm" or model_type.startswith("ensemble")) and getattr(meta, "model_path", None):
            lstm_model = load_lstm(
                path        = resolve_path(meta.model_path),
                n_features  = N_FEATURES,
                hidden_size = cfg.get("lstm_hidden", 128),
                num_layers  = cfg.get("lstm_layers", 2),
                dropout     = cfg.get("lstm_dropout", 0.3),
                num_classes = cfg.get("num_classes", 3),
                device      = "cpu",
            )
            lstm_model.eval()

            if getattr(meta, "scaler_path", None):
                scaler = joblib.load(resolve_path(meta.scaler_path))

        if (model_type == "lgbm" or model_type.startswith("ensemble")) and getattr(meta, "model_path", None):
            lgbm_path = resolve_path(meta.model_path).parent / "lgbm_baseline.pkl"
            if lgbm_path.exists():
                lgbm_model = joblib.load(lgbm_path)

        if model_type == "ensemble" or model_type.startswith("ensemble"):
            if getattr(meta, "meta_learner_path", None):
                meta_path = resolve_path(meta.meta_learner_path)
                if meta_path.exists():
                    meta_learner = joblib.load(meta_path)
            if getattr(meta, "calibrator_path", None):
                cal_path = resolve_path(meta.calibrator_path)
                if cal_path.exists():
                    calibrator = ProbabilityCalibrator.load(cal_path)

        return _ModelBundle(
            lstm=lstm_model, lgbm=lgbm_model, scaler=scaler,
            meta=meta_learner, calibrator=calibrator,
            inference_config=self._config,
        )

    # ── Model forward pass ────────────────────────────────────────────────────

    def _run_model(
        self, df, bundle: _ModelBundle, model_type: str
    ) -> np.ndarray:
        seq_len = self._config.get("inference", {}).get("seq_len", 32)

        from app.services.data_service import InferenceDataService

        if model_type == "lstm":
            return self._lstm_proba(df, bundle, seq_len, InferenceDataService)

        if model_type == "lgbm":
            return self._lgbm_proba(df, bundle)

        # ensemble
        lgbm_p = self._lgbm_proba(df, bundle)
        lstm_p = self._lstm_proba(df, bundle, seq_len, InferenceDataService)
        combined = np.concatenate([lgbm_p, lstm_p]).reshape(1, -1)  # (1, 6)

        if bundle.meta is not None:
            proba = bundle.meta.predict_proba(combined)[0]
        else:
            proba = (lgbm_p + lstm_p) / 2

        if bundle.calibrator is not None:
            proba = bundle.calibrator.transform(proba.reshape(1, -1))[0]

        return proba

    def _lstm_proba(self, df, bundle: _ModelBundle, seq_len: int, data_svc_cls) -> np.ndarray:
        if bundle.lstm is None or bundle.scaler is None:
            raise RuntimeError("LSTM atau scaler tidak ter-load")
        seq = data_svc_cls.prepare_lstm_input(df, bundle.scaler)  # (1, 32, 85)
        tensor = torch.tensor(seq, dtype=torch.float32)
        with torch.no_grad():
            logits = bundle.lstm(tensor)
        return F.softmax(logits, dim=-1).numpy()[0]

    def _lgbm_proba(self, df, bundle: _ModelBundle) -> np.ndarray:
        if bundle.lgbm is None:
            raise RuntimeError("LGBM tidak ter-load")
        X = df[get_feature_cols()].fillna(0).iloc[[-1]]
        return bundle.lgbm.predict_proba(X)[0]

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_n_features(self) -> None:
        n = getattr(self._meta, "n_features", N_FEATURES)
        if n != N_FEATURES:
            raise ValueError(
                f"n_features mismatch: model={n}, config_loader={N_FEATURES}. "
                f"Gunakan model yang ditraining dengan {N_FEATURES} fitur."
            )

    @staticmethod
    def clear_cache() -> None:
        """Clear model cache. Panggil saat model di-switch."""
        model_cache.clear()
        gc.collect()
