# architecture_RL_development.md — Reinforcement Learning untuk SwingTrade v2

## Overview

Paper trading bot saat ini menggunakan arsitektur **cascade**: **LightGBM** (scout) → **LSTM** (confirmer). LightGBM dijalankan dulu (cepat, 1 bar) — kalau hasilnya FLAT atau confidence rendah, LSTM tidak dijalankan (hemat RAM). Kalau LightGBM menghasilkan LONG/SHORT dengan confidence cukup, baru LSTM dijalankan (dalam, 32 bar sequence) sebagai konfirmasi akhir. Sinyal LONG/SHORT/FLAT dieksekusi dengan rule-based TP/SL dan position sizing.

Cascade thresholds (dari `inference_config.json`):
- `scout_flat_threshold` (0.60): LightGBM confidence < ini → FLAT, skip LSTM
- `scout_signal_threshold` (0.55): LightGBM confidence > ini → kandidat signal
- `confirmer_threshold` (0.65): LSTM confidence harus > ini untuk final LONG/SHORT

Hanya 1 model (LSTM + LightGBM) yang di-cache di memori dalam satu waktu — sebelum inferensi koin berikutnya, model sebelumnya di-unload (`gc.collect()`) untuk menjaga RAM ≤ 832 MB Railway.

RL agent akan menggantikan atau meng-augment decision layer — belajar kapan entry, sizing, dan exit dari data historis + pengalaman paper trading.

## Mengapa RL?

Supervised ML sekarang memprediksi arah, tapi tidak bisa:
- Mengoptimalkan **kapan** entry (mungkin signal benar tapi timing terlalu awal)
- Belajar dari **opportunity cost** (FLAT padahal seharusnya entry)
- Adapt terhadap **regime** (strategi berbeda untuk trending vs ranging)
- Learn from **counterfactual** (apa yang terjadi kalau keputusan berbeda?)

RL menyelesaikan ini karena belajar dari **cumulative reward**, bukan accuracy per bar.

## Action Space

```
Actions: { LONG_full, LONG_half, FLAT, SHORT_full, SHORT_half }
```

- `_full` = $100 modal (confidence > 0.75 equivalent)
- `_half` = $50 modal (confidence 0.60–0.75 equivalent)
- `FLAT` = tidak buka posisi
- Exit di-handle oleh TP/SL rule-based (untuk Phase 1). Phase 2 bisa ditambahkan `EXIT` action.

## State Representation

State per bar H1 terdiri dari 3 blok:

### Blok A: 85 Fitur ML (full feature vector)
```
Semua kolom dari features_df (VP, FVG, S/R, MA crosses, RSI, ATR, OBV, dll.)
```
Disimpan di `feature_snapshot` Signal record — sekarang hanya 4 field, perlu diubah ke 85.

### Blok B: Position Context
```
- has_open_position: bool
- position_direction: LONG | SHORT | None
- position_entry_price: float
- unrealized_pnl_pct: float
- bars_held: int
- mfe_pct: float  (max favorable excursion so far)
- mae_pct: float  (max adverse excursion so far)
```

### Blok C: Regime Context
```
- volatility_regime: low | med | high  (ATR percentile vs 100-bar)
- trend_regime: trending_up | trending_down | ranging  (ADX + market structure)
- vcb_active: bool
```

## Reward Function

### Phase 1: PnL-based (sederhana)

```
reward = pnl_net / modal  (per trade closed)
```

### Phase 2: Risk-adjusted (Sharpe-like)

```
reward = (pnl_net / modal) - (drawdown_penalty * max_drawdown_pct)
```

### Phase 3: Regime-aware

```
reward = regime_multiplier * (pnl_net / modal)
  - trending_up + LONG: multiplier 1.0
  - trending_up + SHORT: multiplier 0.5
  - ranging: multiplier 0.8 (semua arah)
```

## Data Pipeline — 5 Tahap

### Tahap 1: Full Feature Snapshot ✅ DONE

**File:** `app/jobs/generate_signals.py` (modifikasi)

**Implementasi aktual:** Full 85+ fitur dari `features_df.iloc[-1]` disimpan langsung di **DB** (`Signal.feature_snapshot` sebagai JSON):

- Setiap sinyal menyimpan semua kolom dari `features_df` (85 feature columns + h4_swing_high, h4_swing_low) + metadata asli (close, atr, confidence)
- Konversi numpy types → native Python (int/float/list) untuk JSON serialization
- Estimasi storage: ~3 KB/sinyal → ~40-50 MB/bulan → aman untuk Neon free tier (0.5 GB) dalam jangka menengah
- UI monitoring di `/rl-data` — ringkasan per bulan + preview 10 sinyal terakhir (query dari DB)
- `app/api/rl_data.py` menyediakan API scan dari Signal table

**Effort:** 1 file, ~15 baris perubahan. **Status: SELESAI.**
**Impact:** Fondasi semua data RL. Tanpa ini tidak bisa replay state.

### Tahap 2: TradeBar Table — Bar-by-Bar Trajectory

**File baru:** `app/models/trade_bar.py`

```python
class TradeBar(db.Model):
    __tablename__ = "trade_bars"

    id          = db.Column(db.Integer, primary_key=True)
    trade_id    = db.Column(db.Integer, db.ForeignKey("trades.id"), nullable=False)
    bar_index   = db.Column(db.Integer, nullable=False)    # 0, 1, 2, ...
    timestamp   = db.Column(db.DateTime, nullable=False)

    # Harga bar
    open        = db.Column(db.Float)
    high        = db.Column(db.Float)
    low         = db.Column(db.Float)
    close       = db.Column(db.Float)

    # Unrealized PnL di akhir bar ini
    unrealized_pnl_pct = db.Column(db.Float)

    # Cumulative excursion
    mfe_pct     = db.Column(db.Float)  # max favorable (highest high so far)
    mae_pct     = db.Column(db.Float)  # max adverse (lowest low so far)

    # TP/SL touch di bar ini?
    tp_touched  = db.Column(db.Boolean, default=False)
    sl_touched  = db.Column(db.Boolean, default=False)

    # Snapshot 85 fitur di bar ini
    feature_snapshot = db.Column(db.Text)  # JSON

    trade = db.relationship("Trade", backref="bars")
```

**Modifikasi `check_positions.py`:**
```python
# Setiap kali cek open trade → append 1 bar ke trade_bars
# Simpan unrealized PnL, MFE/MAE, dan 85 fitur
```

**Effort:** 1 model baru + 1 migrasi + modifikasi `check_positions.py`.
**Impact:** Data trajectory utuh. RL bisa replay sequence keputusan.

### Tahap 3: Regime Label per Bar

**File:** `app/services/regime.py` (baru)

```python
def classify_regime(features_df) -> dict:
    """
    Return:
      volatility_regime: "low" | "med" | "high"
      trend_regime: "trending_up" | "trending_down" | "ranging"
    """
    atr_series = features_df["atr_14_h1"]
    adx_series = features_df.get("adx_14", None)

    # Volatility: ATR percentile vs 100-bar
    atr_pct = (atr_series.iloc[-1] > atr_series.tail(100)).mean()
    if atr_pct > 0.8:
        vol = "high"
    elif atr_pct < 0.2:
        vol = "low"
    else:
        vol = "med"

    # Trend: ADX + market structure
    if adx_series is not None and adx_series.iloc[-1] > 25:
        # Cek HH/HL sequence
        highs = features_df["high_1d"].tail(20)
        lows = features_df["low_1d"].tail(20)
        hh = highs.iloc[-1] > highs.iloc[-5]
        ll = lows.iloc[-1] < lows.iloc[-5]
        if hh and not ll:
            trend = "trending_up"
        elif ll and not hh:
            trend = "trending_down"
        else:
            trend = "ranging"
    else:
        trend = "ranging"

    return {"volatility_regime": vol, "trend_regime": trend}
```

**Effort:** 1 file baru, ~40 baris.
**Impact:** State augmentation untuk RL. Juga berguna untuk analisis trade existing.

### Tahap 4: Counterfactual Tracking

**File:** `app/models/rejected_signal.py` (baru)

```python
class RejectedSignal(db.Model):
    __tablename__ = "rejected_signals"

    id            = db.Column(db.Integer, primary_key=True)
    coin_id       = db.Column(db.Integer, db.ForeignKey("coins.id"))
    direction     = db.Column(db.String(10))
    confidence    = db.Column(db.Float)
    entry_price   = db.Column(db.Float)
    reject_reason = db.Column(db.String(50))  # "low_conf", "vcb", "existing_position"
    signal_time   = db.Column(db.DateTime)
    feature_snapshot = db.Column(db.Text)

    # Outcome: apa yang terjadi 24 bar ke depan?
    outcome_24h_high  = db.Column(db.Float)  # highest price dalam 24 bar
    outcome_24h_low   = db.Column(db.Float)  # lowest price dalam 24 bar
    outcome_24h_close = db.Column(db.Float)  # close setelah 24 bar
    would_hit_tp      = db.Column(db.Boolean)  # apakah TP akan tersentuh?
    would_hit_sl      = db.Column(db.Boolean)  # apakah SL akan tersentuh?
```

**Logic di `generate_signals.py`:**
```python
if not trade:
    # Catat signal yang ditolak
    rejected = RejectedSignal(
        coin_id=coin_id,
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        reject_reason=reason,
        signal_time=utcnow(),
        feature_snapshot=json.dumps(feature_dict),
    )
    db.session.add(rejected)

# Background job: setelah 24 bar, update outcome
```

**Effort:** 1 model + 1 migrasi + modifikasi `generate_signals.py` + 1 job baru.
**Impact: Paling mahal tapi paling berharga untuk RL.** Counterfactual adalah data yang tidak bisa didapat dari backtest.

### Tahap 5: Outcome Decomposition

Modifikasi `_close_trade()` di `paper_trading.py` untuk menyimpan:

```python
# Tambahkan kolom ke Trade model:
trade.mfe_pct = max_favorable_pct   # dari trade_bars
trade.mae_pct = max_adverse_pct     # dari trade_bars
trade.volatility_regime = regime["volatility_regime"]  # saat entry
trade.trend_regime = regime["trend_regime"]             # saat entry
trade.entry_confidence = confidence
```

**Effort:** Tambah 5 kolom di Trade model + migrasi + modifikasi `_close_trade()`.
**Impact:** Memungkinkan post-hoc analysis — trade mana yang bagus karena skill vs karena kondisi pasar.

---

## Arsitektur RL Agent

### Phase 1: Offline RL (pakai data terkumpul)

```
Data collection (paper trading 3-6 bulan)
    ↓
Replay buffer dari TradeBar + RejectedSignal
    ↓
Train RL agent offline (DQN / PPO / CQL)
    ↓
Backtest agent vs supervised baseline
    ↓
Deploy sebagai "shadow mode" — agent memberi saran, eksekusi tetap rule-based
```

### Phase 2: Online Fine-tuning

```
Shadow mode diverifikasi > 1 bulan
    ↓
Agent mulai eksekusi dengan modal kecil ($10/trade)
    ↓
Online learning dari real-time feedback
    ↓
Scale up modal secara bertahap
```

### Model Choice

| Algo | Kelebihan | Kekurangan | Cocok untuk |
|------|-----------|------------|-------------|
| **DQN** | Discrete actions, sederhana | Overestimate Q-values, unstable | Phase 1 — baseline |
| **PPO** | Stable, on-policy, good for finance | Sample inefficient | Phase 2 — online |
| **CQL** | Conservative, offline-safe | Kompleks, butuh tuning | Offline training dari data historis |

Rekomendasi: **Mulai dengan DQN** untuk offline training. Pindah ke **PPO** untuk online fine-tuning.

---

## Prioritas Implementasi

| # | Tahap | Effort | Impact | Dependensi |
|---|-------|--------|--------|------------|
| 1 | Full 85 fitur snapshot ✅ | 2 file, rendah | **Fondasi** | Tidak ada |
| 2 | TradeBar table | 2 file, sedang | Trajectory data | Tahap 1 |
| 3 | Regime label | 1 file, rendah | State augmentation | Tahap 1 |
| 4 | Outcome decomposition | 1 file, rendah | Post-hoc analysis | Tahap 2 |
| 5 | Counterfactual tracking | 2 file, sedang | **Data paling mahal** | Tahap 1, 2 |

## Milestone

| Milestone | Waktu (estimasi) | Deliverable |
|-----------|-----------------|-------------|
| M1: Data foundation | ~~1-2 hari~~ SELESAI | Full feature snapshot ✅ + regime label |
| M2: Trajectory | 2-3 hari | TradeBar table + outcome decomposition |
| M3: Counterfactual | 3-5 hari | RejectedSignal + outcome backfill job |
| M4: Replay buffer | 3-5 hari | Dataset builder dari 3 tabel |
| M5: RL training | 1-2 minggu | DQN agent trained offline |
| M6: Shadow deploy | 1 minggu | Agent berjalan paralel, observasi-only |

---

## Catatan Teknis

### Railway 832 MB RAM constraint
- RL training TIDAK dijalankan di Railway
- Training dilakukan offline (local/Colab) menggunakan data yang di-export
- Railway hanya menjalankan inference + data collection

### Database
- `trade_bars` akan menjadi tabel terbesar (~20 bar × N trades × 18 coins)
- Estimasi: 100 trade/bulan × 18 koin × 20 bar = 36K rows/bulan → manageable
- `rejected_signals` sekitar 10-20× lebih banyak dari trades (kebanyakan FLAT) → ~200K rows/bulan
- Pertimbangkan partitioning atau retention policy (hapus > 6 bulan)

### Export untuk Training
```
Endpoint: GET /api/rl/export
Return: JSON/Parquet bundle dari:
  - signals (dengan full 85 fitur)
  - trade_bars (trajectory)
  - rejected_signals (counterfactual)
  - regime labels
```
