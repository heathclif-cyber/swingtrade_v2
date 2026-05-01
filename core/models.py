"""
core/models.py — Definisi arsitektur model dan fungsi load/save
Dipakai bersama oleh pipeline/05_train_lstm.py, pipeline/06_ensemble.py,
dan Swing_Trade9.6/ml/ml_signal.py

PENTING: Arsitektur TradingLSTM TIDAK BOLEH diubah tanpa retraining.
         n_features=65, hidden_size=128, num_layers=2, dropout=0.3
"""

import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def _default_n_features() -> int:
    """Lazy: coba config_loader (web app), fallback ke 85 (default training)."""
    try:
        from app.services.config_loader import get_n_features
        return get_n_features()
    except Exception:
        return 85


# ─── TradingLSTM ─────────────────────────────────────────────────────────────

class TradingLSTM(nn.Module):
    """
    Unidirectional LSTM untuk prediksi sinyal trading multiclass.

    Input  : (batch, seq_len, n_features)
    Output : (batch, num_classes) — raw logits
    """

    def __init__(
        self,
        n_features:  Optional[int] = None,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        num_classes: int   = 3,
    ):
        if n_features is None:
            n_features = _default_n_features()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = False,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last   = self.norm(out[:, -1, :])
        last   = self.dropout(last)
        return self.fc(last)


# ─── Load / Save LSTM ────────────────────────────────────────────────────────

def save_lstm(model: TradingLSTM, path: Path) -> None:
    """Simpan state dict saja (bukan full checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_lstm(
    path: Path,
    n_features:  Optional[int] = None,
    hidden_size: int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    num_classes: int   = 3,
    device: str        = "cpu",
) -> TradingLSTM:
    """Load LSTM dari state dict."""
    model = TradingLSTM(n_features, hidden_size, num_layers, dropout, num_classes)
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Probability Calibrator ──────────────────────────────────────────────────

class ProbabilityCalibrator:
    """
    Kalibrasi probabilitas post-hoc untuk output ensemble.
    Difit pada validation set setelah ensemble selesai ditraining.

    Method "isotonic" (default) lebih fleksibel untuk distribusi
    non-Gaussian seperti trading signals. "platt" = Logistic Regression.

    Usage:
        cal = ProbabilityCalibrator()
        cal.fit(proba_val, y_val)        # proba: (n, 3), y: (n,) int
        cal_proba = cal.transform(proba) # output: (n, 3), setiap baris sum=1
        cal.save(path)
        cal = ProbabilityCalibrator.load(path)
    """

    def __init__(self, method: str = "isotonic"):
        self.method      = method
        self.calibrators = {}  # {class_idx: fitted_estimator}

    def fit(self, proba: "np.ndarray", y: "np.ndarray") -> None:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        for c in range(proba.shape[1]):
            y_bin = (y == c).astype(int)
            if self.method == "isotonic":
                est = IsotonicRegression(out_of_bounds="clip")
                est.fit(proba[:, c], y_bin)
            else:
                est = LogisticRegression(C=1.0)
                est.fit(proba[:, c].reshape(-1, 1), y_bin)
            self.calibrators[c] = est

    def transform(self, proba: "np.ndarray") -> "np.ndarray":
        import numpy as np

        cal = np.zeros_like(proba)
        for c, est in self.calibrators.items():
            if self.method == "isotonic":
                cal[:, c] = est.predict(proba[:, c])
            else:
                cal[:, c] = est.predict_proba(proba[:, c].reshape(-1, 1))[:, 1]

        row_sum = cal.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return cal / row_sum

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ProbabilityCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)