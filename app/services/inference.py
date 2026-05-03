"""
app/services/inference.py — Load model dan predict signal untuk satu koin.

Mode yang didukung (dari inference_config.json → model_version):
  hierarchical → H4 LGBM bias → H1 LGBM entry → LSTM confirmation  ← DEFAULT BARU
  lstm         → LSTM forward → softmax (standalone)
  lgbm         → H1 LGBM predict_proba (standalone)
  ensemble     → LGBM + LSTM → MetaLearner (DEPRECATED — calibrator DISABLED)

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


def _resolve_model_path(relative_path: str):
    """Resolve path model; auto-fallback ke models/<filename> jika subfolder tidak ditemukan.

    Ini menangani kasus stale DB dengan path versi lama seperti
    'models/v20260425_170250/lstm_best.pt' padahal file sudah dipindah ke 'models/lstm_best.pt'.
    """
    from pathlib import Path
    path = resolve_path(relative_path)
    if not path.exists():
        fallback = resolve_path("models") / Path(relative_path).name
        if fallback.exists():
            logger.warning(
                f"[inference] Path '{path}' tidak ditemukan — "
                f"auto-fallback ke '{fallback}' (path DB lama?)"
            )
            return fallback
        logger.error(
            f"[inference] File tidak ditemukan: '{path}' "
            f"(fallback '{fallback}' juga tidak ada)"
        )
    return path


# ─── Model bundle ────────────────────────────────────────────────────────────

class _ModelBundle:
    """
    Container untuk semua model artifacts satu versi.

    Hierarchical cascade (model_version='hierarchical_v1'):
      h4_lgbm     : H4 LGBM regime filter
      lgbm        : H1 LGBM entry signal generator
      lstm        : LSTM momentum confirmation
      scaler      : LSTM feature scaler
      h4_feat_cols: feature columns untuk H4 LGBM

    Legacy ensemble (DEPRECATED):
      meta        : meta-learner (LogisticRegression) — tidak dipakai
      calibrator  : isotonic calibrator — DINONAKTIFKAN (over-calibrates FLAT)
    """
    __slots__ = (
        "lstm", "lgbm", "scaler", "meta", "calibrator",
        "h4_lgbm", "h4_feat_cols", "h4_calibrator", "inference_config"
    )

    def __init__(self, **kw):
        # Set defaults untuk slot yang tidak diberikan
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot, None))


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
        model_type: str = "hierarchical",
    ) -> Optional[dict]:
        """
        Return {direction, confidence, proba, entry_price, atr_value}
        atau None jika confidence < threshold atau cold start.

        model_type default berubah dari 'lstm' → 'hierarchical'.
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
        h4_model   = None
        h4_feat_cols = []

        # ── LSTM ─────────────────────────────────────────────────────────────
        if (model_type in ("lstm", "hierarchical") or model_type.startswith("ensemble")) \
                and getattr(meta, "model_path", None):
            lstm_model = load_lstm(
                path        = _resolve_model_path(meta.model_path),
                n_features  = N_FEATURES,
                hidden_size = cfg.get("lstm_hidden", 128),
                num_layers  = cfg.get("lstm_layers", 2),
                dropout     = cfg.get("lstm_dropout", 0.3),
                num_classes = cfg.get("num_classes", 3),
                device      = "cpu",
            )
            lstm_model.eval()

            if getattr(meta, "scaler_path", None):
                scaler = joblib.load(_resolve_model_path(meta.scaler_path))

        # ── H1 LGBM ──────────────────────────────────────────────────────────
        if (model_type in ("lgbm", "hierarchical") or model_type.startswith("ensemble")) \
                and getattr(meta, "model_path", None):
            lgbm_path = _resolve_model_path(meta.model_path).parent / "lgbm_baseline.pkl"
            if lgbm_path.exists():
                lgbm_model = joblib.load(lgbm_path)

        # ── H4 LGBM (hierarchical only) ───────────────────────────────────────
        h4_calibrator = None
        if model_type == "hierarchical":
            h4_path      = _resolve_model_path(meta.model_path).parent / "lgbm_h4.pkl"
            h4_feat_file = _resolve_model_path(meta.model_path).parent / "h4_feature_cols.json"
            h4_cal_file  = _resolve_model_path(meta.model_path).parent / "h4_calibrator.pkl"
            if h4_path.exists() and h4_feat_file.exists():
                h4_model = joblib.load(h4_path)
                import json
                with open(h4_feat_file) as f:
                    h4_feat_cols = json.load(f)
                # Load H4 calibrator (opsional — jika tidak ada, pakai raw proba)
                if h4_cal_file.exists():
                    h4_calibrator = ProbabilityCalibrator.load(h4_cal_file)
                    logger.info(f"[inference] H4 calibrator loaded: {h4_cal_file.name}")
                else:
                    logger.info("[inference] H4 calibrator tidak ditemukan — pakai raw proba")
                logger.info(f"[inference] H4 model loaded: {len(h4_feat_cols)} features")
            else:
                logger.warning(
                    "[inference] lgbm_h4.pkl tidak ditemukan — H4 bias = FLAT. "
                    "Jalankan 04_train_lgbm_h4.py untuk melatih H4 model."
                )

        # ── Ensemble legacy (DEPRECATED) ──────────────────────────────────────
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
            h4_lgbm=h4_model, h4_feat_cols=h4_feat_cols,
            h4_calibrator=h4_calibrator,
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

        # ── Hierarchical Cascade (DEFAULT) ─────────────────────────────────────
        if model_type == "hierarchical":
            return self._hierarchical_proba(df, bundle, seq_len, InferenceDataService)

        # ── Ensemble (DEPRECATED) ─────────────────────────────────────────────
        lgbm_p = self._lgbm_proba(df, bundle)
        lstm_p = self._lstm_proba(df, bundle, seq_len, InferenceDataService)
        combined = np.concatenate([lgbm_p, lstm_p]).reshape(1, -1)  # (1, 6)

        if bundle.meta is not None:
            proba = bundle.meta.predict_proba(combined)[0]
        else:
            proba = (lgbm_p + lstm_p) / 2

        # ═══════════════════════════════════════════════════════════════════════
        # CALIBRATOR — DINONAKTIFKAN SEMENTARA
        # ═══════════════════════════════════════════════════════════════════════
        # Penyebab: ProbabilityCalibrator (IsotonicRegression) over-calibrates
        # ke kelas FLAT karena ketidakseimbangan kelas di validation set.
        #
        # Bukti dari Railway log 15:05 UTC:
        #   LSTM standalone (BNBUSDT): FLAT=0.5104, LONG=0.489  ← wajar
        #   ensemble_v2 (SOLUSDT):     FLAT=0.985,  LONG=0.002  ← ekstrem
        #   ensemble_v2 (DOGEUSDT):    FLAT=0.998,  LONG=0.001  ← ekstrem
        #
        # Perbedaan hanya pada ada/tidaknya calibrator. LSTM standalone
        # tanpa calibrator menghasilkan distribusi yang jauh lebih baik.
        #
        # Untuk mengaktifkan kembali, hapus komentar di bawah ini.
        # ═══════════════════════════════════════════════════════════════════════
        logger.info(
            f"[ensemble] proba BEFORE calibrator: "
            f"SHORT={proba[0]:.4f} FLAT={proba[1]:.4f} LONG={proba[2]:.4f}"
        )

        # if bundle.calibrator is not None:
        #     proba_cal = bundle.calibrator.transform(proba.reshape(1, -1))[0]
        #     logger.info(
        #         f"[ensemble] proba AFTER calibrator: "
        #         f"SHORT={proba_cal[0]:.4f} FLAT={proba_cal[1]:.4f} LONG={proba_cal[2]:.4f}"
        #     )
        #     if proba_cal[1] > proba[1] + 0.1:
        #         logger.warning(
        #             f"[ensemble] Calibrator meningkatkan FLAT drastis "
        #             f"({proba[1]:.3f} → {proba_cal[1]:.3f}) — menggunakan raw meta-learner"
        #         )
        #     else:
        #         proba = proba_cal

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

    def _lstm_adjustment(self, h1_conf: float, lstm_dir: int, bias: int,
                         mode: str, agree_boost: float,
                         neutral_pen: float, opposite_pen: float) -> float:
        """
        Hitung LSTM confidence adjustment (sama dengan versi di backtest_utils).

        mode="relative":  adj = {agree/neutral/opposite} × h1_conf  (original)
        mode="absolute":  adj = fixed value
        mode="tiered":    adj bervariasi berdasarkan margin confidence
        """
        if lstm_dir == bias:       # agree → boost
            return agree_boost * (1.0 - h1_conf) if mode == "relative" else agree_boost
        elif lstm_dir == 1:        # neutral (FLAT) → slight reduce
            pen = neutral_pen
            return -pen * h1_conf if mode == "relative" else -pen
        else:                      # opposite → strong reduce
            pen = opposite_pen
            if mode == "tiered":
                margin = h1_conf - 0.62  # threshold reference
                if margin < 0.05:
                    return -pen * 1.5        # borderline → heavy
                elif margin < 0.10:
                    return -pen * 1.0        # moderate → medium
                else:
                    return -pen * 0.5        # confident → light
            return -pen * h1_conf if mode == "relative" else -pen

    def _hierarchical_proba(
        self, df, bundle: _ModelBundle, seq_len: int, data_svc_cls
    ) -> np.ndarray:
        """
        Hierarchical cascade untuk single-bar inference:

          STEP 1  H4 LGBM → bias (LONG/SHORT/FLAT) — dengan calibrator + percentile log
          STEP 2  H1 LGBM → entry probability
          STEP 3  LSTM    → soft proportional confidence adjustment (config-driven)
          STEP 4  Decision → emit signal jika adjusted_conf >= threshold

        LSTM adjustment modes (configurable via inference_config.json):
          "relative":  {agree/neutral/opposite} × h1_conf  (original)
          "absolute":  fixed values
          "tiered":    penalty varies by confidence margin

        Returns softmax-like probability array (3,): [SHORT, FLAT, LONG].
        Confidence dari winning class diteruskan ke predict() untuk threshold check.
        """
        from app.config_loader import load_inference_config as _load_cfg
        cfg = self._config.get("hierarchical_thresholds", {})
        h4_thr_long  = float(cfg.get("h4_binary_threshold_long",  0.55))
        h4_thr_short = float(cfg.get("h4_binary_threshold_short", 0.55))
        h1_thr_long  = float(cfg.get("h1_threshold_long",  0.62))
        h1_thr_short = float(cfg.get("h1_threshold_short", 0.62))
        lstm_confirm = bool(cfg.get("lstm_confirmation", True))
        # LSTM adjustment config
        lstm_mode     = str(cfg.get("lstm_adjust_mode", "tiered"))
        lstm_agree    = float(cfg.get("lstm_adjust_agree_boost", 0.05))
        lstm_neutral  = float(cfg.get("lstm_adjust_neutral_pen", 0.05))
        lstm_opposite = float(cfg.get("lstm_adjust_opposite_pen", 0.08))

        # Default: FLAT
        proba = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # ── STEP 1: H4 bias (dengan calibrator + percentile log) ──────────────
        bias = 1  # FLAT default
        if bundle.h4_lgbm is not None and bundle.h4_feat_cols:
            valid_h4 = [c for c in bundle.h4_feat_cols if c in df.columns]
            if valid_h4:
                h4_p = bundle.h4_lgbm.predict_proba(df[valid_h4].fillna(0).iloc[[-1]])[0]
                # Log proba BEFORE calibration
                prob_long_raw  = float(h4_p[1])
                prob_short_raw = float(h4_p[0])

                # Apply H4 calibrator jika tersedia
                if bundle.h4_calibrator is not None:
                    h4_p_cal = bundle.h4_calibrator.transform(h4_p.reshape(1, -1))[0]
                    prob_long_cal  = float(h4_p_cal[1])
                    prob_short_cal = float(h4_p_cal[0])
                    logger.debug(
                        f"[hierarchical] STEP1 H4 calibrator shift: "
                        f"SHORT {prob_short_raw:.3f}→{prob_short_cal:.3f} "
                        f"({(prob_short_cal-prob_short_raw)*100:+.1f}pp) | "
                        f"LONG {prob_long_raw:.3f}→{prob_long_cal:.3f} "
                        f"({(prob_long_cal-prob_long_raw)*100:+.1f}pp)"
                    )
                    h4_p = h4_p_cal
                else:
                    logger.debug(
                        f"[hierarchical] STEP1 H4 raw: SHORT={prob_short_raw:.3f} LONG={prob_long_raw:.3f} "
                        f"(no calibrator)"
                    )

                # Binary model: [prob_SHORT, prob_LONG] — index 1 = LONG, index 0 = SHORT
                prob_long  = float(h4_p[1])
                prob_short = float(h4_p[0])
                if prob_long >= h4_thr_long:
                    bias = 2  # LONG
                elif prob_short >= h4_thr_short:
                    bias = 0  # SHORT
                logger.debug(
                    f"[hierarchical] STEP1 H4: SHORT={prob_short:.3f} LONG={prob_long:.3f} "
                    f"→ bias={'LONG' if bias==2 else 'SHORT' if bias==0 else 'FLAT'}"
                )
        else:
            logger.debug("[hierarchical] STEP1 H4: model tidak tersedia — bias = FLAT")

        if bias == 1:
            return proba  # H4 = FLAT → selalu FLAT, tidak perlu cek H1

        # ── STEP 2: H1 entry signal ───────────────────────────────────────────
        h1_p = self._lgbm_proba(df, bundle)  # [SHORT, FLAT, LONG]
        h1_conf = float(h1_p[bias])  # confidence sesuai arah bias
        h1_thr  = h1_thr_long if bias == 2 else h1_thr_short
        logger.debug(
            f"[hierarchical] STEP2 H1: SHORT={h1_p[0]:.3f} FLAT={h1_p[1]:.3f} LONG={h1_p[2]:.3f} "
            f"→ {'LONG' if bias==2 else 'SHORT'}_conf={h1_conf:.4f} threshold={h1_thr:.2f} "
            f"→ {'PASS' if h1_conf >= h1_thr else 'FAIL (below threshold)'}"
        )

        if h1_conf < h1_thr:
            return proba  # H1 tidak confirm → FLAT

        # ── STEP 3: LSTM soft proportional adjustment (config-driven) ─────────
        lstm_adjustment = 0.0
        adjusted_conf   = h1_conf
        if lstm_confirm and bundle.lstm is not None and bundle.scaler is not None:
            lstm_p   = self._lstm_proba(df, bundle, seq_len, data_svc_cls)
            lstm_dir = int(np.argmax(lstm_p))

            lstm_adjustment = self._lstm_adjustment(
                h1_conf, lstm_dir, bias,
                lstm_mode, lstm_agree, lstm_neutral, lstm_opposite,
            )
            adjusted_conf = np.clip(h1_conf + lstm_adjustment, 0.0, 1.0)
            logger.debug(
                f"[hierarchical] STEP3 LSTM: SHORT={lstm_p[0]:.3f} FLAT={lstm_p[1]:.3f} LONG={lstm_p[2]:.3f} "
                f"→ dir={'LONG' if lstm_dir==2 else 'SHORT' if lstm_dir==0 else 'FLAT'} "
                f"mode={lstm_mode} adjustment={lstm_adjustment:+.4f} "
                f"conf={h1_conf:.4f}→{adjusted_conf:.4f} "
                f"→ {'PASS' if adjusted_conf >= h1_thr else 'FAIL'}"
            )
        else:
            logger.debug(
                f"[hierarchical] STEP3 LSTM: disabled (lstm_confirm={lstm_confirm}, "
                f"lstm={'available' if bundle.lstm else 'None'}, "
                f"scaler={'available' if bundle.scaler else 'None'})"
            )

        # ── STEP 4: Decision with adjusted confidence ─────────────────────────
        if adjusted_conf >= h1_thr:
            if bias == 2:
                proba = np.array([0.0, 1.0 - adjusted_conf, adjusted_conf], dtype=np.float32)
            else:  # bias == 0
                proba = np.array([adjusted_conf, 1.0 - adjusted_conf, 0.0], dtype=np.float32)
            logger.debug(
                f"[hierarchical] DECISION: {'LONG' if bias==2 else 'SHORT'} "
                f"adjusted_conf={adjusted_conf:.4f} → EMIT SIGNAL"
            )
        else:
            logger.debug(
                f"[hierarchical] DECISION: adjusted_conf={adjusted_conf:.4f} "
                f"< threshold={h1_thr:.2f} → FLAT (confidence too low)"
            )

        return proba

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

