# Arsitektur End-to-End: Asisten Trading Pribadi

> **Versi:** 1.4 · **Tanggal:** 2026-04-30
> **Platform Target:** Railway Hobby Tier (832 MB RAM, Persistent Volume) + Neon PostgreSQL Free Tier (0.5 GB)
> **Model Deploy Fase 1:** LSTM + LGBM (`TradingLSTM` + scaler + LGBM .pkl) — plug-and-play via `model_registry.json`
> **Model Terdokumentasi:** Ensemble LGBM+LSTM+Meta+Calibrator (Fase 2)
> **n_features:** 85 (koreksi dari v1.3 yang salah tulis 83) · **Model source:** hasil training di project terpisah, copy via `deploy/prepare_deploy.py` (lihat §0)

---

## Daftar Isi

0. [Pre-Deploy Setup & Catatan Penting](#0-pre-deploy-setup--catatan-penting)
1. [Penjelasan Skema Data Historis](#1-penjelasan-skema-data-historis)
2. [Arsitektur End-to-End](#2-arsitektur-end-to-end)
3. [Struktur Folder Proyek](#3-struktur-folder-proyek)
4. [Skema Database PostgreSQL (Neon)](#4-skema-database-postgresql-neon)
5. [Preprocessing untuk Inference](#5-preprocessing-untuk-inference)
6. [Model Serving Strategy](#6-model-serving-strategy)
7. [Definisi Endpoint API](#7-definisi-endpoint-api)
8. [Dashboard UI](#8-dashboard-ui)
9. [Modul Paper Trading](#9-modul-paper-trading)
10. [Job Scheduler](#10-job-scheduler)
11. [Caching & Manajemen Memori](#11-caching--manajemen-memori)
12. [Deployment ke Railway](#12-deployment-ke-railway)

---

## 0. Pre-Deploy Setup & Catatan Penting

> Bagian ini wajib dibaca sebelum mulai build app. Berisi konteks model yang sudah ada, known issues, dan langkah persiapan sebelum `app/` ditulis.

### 0.1 Status Model (Sudah Ditraining)

Model telah selesai ditraining di **project terpisah**. Sebelum build app, file-file ini harus di-copy ke project ini dengan struktur folder versioned.

**Source (project training lama):**
```
<path_project_lama>/models/
├── lgbm_baseline.pkl
├── lstm_best.pt
├── lstm_scaler.pkl
├── ensemble_meta.pkl
├── calibrator.pkl
├── inference_config.json       ← perlu di-patch sebelum dipakai app
└── model_registry.json         ← format lama (flat), perlu migrasi
```

**Target (project ini setelah setup):**
```
models/
├── model_registry.json         ← format versioned baru (dibuat oleh migrate_registry.py)
└── v20260425_170250/           ← run_id dari inference_config["created_at"]
    ├── lgbm_baseline.pkl
    ├── lstm_best.pt
    ├── lstm_scaler.pkl
    ├── ensemble_meta.pkl
    ├── calibrator.pkl
    └── inference_config.json   ← versi yang sudah di-patch (coins_validated.recommended + fallback_tp_sl)
```

### 0.2 Langkah Pre-Deploy (Jalankan Sekali, Sebelum Build App)

```bash
# 1. Konversi model_registry.json lama (format flat) → format versioned
python deploy/migrate_registry.py \
  --src <path_project_lama>/models/model_registry.json \
  --inference-config <path_project_lama>/models/inference_config.json

# 2. Copy model files + patch inference_config + buat folder versioned
python deploy/prepare_deploy.py \
  --src <path_project_lama>/models
```

Setelah ini, `models/v{run_id}/` terbentuk dan `model_registry.json` di root `models/` siap dibaca app.

### 0.3 Known Issues & Fixes

Berikut issues yang ditemukan saat review model files hasil training, dan bagaimana cara penanganannya:

| # | Issue | Dampak ke App | Fix |
|---|-------|---------------|-----|
| 1 | `n_features` di plan v1.3 salah tulis "83", aktualnya **85** | Guard `n_features == len(FEATURE_COLS_V3)` akan gagal jika pakai nilai lama | Semua referensi di dokumen ini sudah dikoreksi ke **85** |
| 2 | `model_registry.json` format lama: `{"active": ..., "models": {...}}` | App tidak bisa baca — butuh format `{"versions": [...]}` | `deploy/migrate_registry.py` konversi ke format versioned |
| 3 | `inference_config.json` tidak punya key `coins_validated.recommended` | Job `generate_signals` akan `KeyError` saat start | `deploy/prepare_deploy.py` menambah key ini = `high_priority + medium_priority` |
| 4 | `inference_config["labeling"]` berisi `tp_atr_mult`/`sl_atr_mult` — ambigu (terkesan fixed ATR, padahal ini fallback) | Developer bisa salah baca dan tidak implementasi swing TP/SL | `prepare_deploy.py` rename section jadi `fallback_tp_sl` saat patching |
| 5 | Model paths di registry hanya filename (`lgbm_baseline.pkl`), bukan versioned path | App tidak bisa resolve lokasi file | `migrate_registry.py` buat paths dengan prefix `models/v{run_id}/` |

### 0.4 Struktur `coins_validated` Sebelum & Sesudah Patch

**Sebelum patch** (dari `inference_config.json` asli):
```json
"coins_validated": {
  "high_priority":   ["1000PEPEUSDT", "DOGEUSDT", "1000SHIBUSDT", "ADAUSDT", "TRXUSDT", "ETHUSDT"],
  "medium_priority": ["POLUSDT", "SUIUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "TONUSDT", "ARBUSDT", "AVAXUSDT"],
  "caution":         ["TAOUSDT", "NEARUSDT"]
}
```

**Sesudah patch** (oleh `prepare_deploy.py`):
```json
"coins_validated": {
  "high_priority":   [...sama...],
  "medium_priority": [...sama...],
  "caution":         [...sama...],
  "recommended":     ["1000PEPEUSDT", "DOGEUSDT", "1000SHIBUSDT", "ADAUSDT", "TRXUSDT",
                      "ETHUSDT", "POLUSDT", "SUIUSDT", "LINKUSDT", "SOLUSDT",
                      "XRPUSDT", "DOTUSDT", "TONUSDT", "ARBUSDT", "AVAXUSDT"]
}
```

Key `recommended` = `high_priority + medium_priority`. Koin `caution` (TAOUSDT, NEARUSDT) sengaja tidak dimasukkan karena drawdown tinggi.

### 0.5 Backtest Summary Model (Referensi)

Model `ensemble_v2` yang akan dideploy, dari `inference_config.json`:

| Metric | Value |
|--------|-------|
| Mean Win Rate (OOS 2025-05 → 2026-04) | **77.18%** |
| Mean Max Drawdown (leverage 3x) | 14.45% |
| Mean Trade per Month | 39.46 |
| Max Consecutive Loss | 10 |
| n_features | **85** |
| Labeling | swing_based_h4 |
| Seq Length (LSTM) | 32 |

---

## 1. Skema Data & Aliran Pipeline

### 1.1 Sumber Data

| Sumber | Data | Fungsi Fetch |
|--------|------|-------------|
| Binance Futures API | Klines 1h/4h/1d, Funding Rate, OI | [`core/fetchers.py`](core/fetchers.py) |
| CoinGecko | BTC Dominance | `fetch_btc_dominance()` |
| Alternative.me | Fear & Greed Index | `fetch_fear_greed()` |

Data mentah disimpan sebagai **Parquet** di `data/raw/`. Setelah cleaning + feature engineering: `data/labeled/{SYMBOL}_features_v3.parquet`.

### 1.2 Fitur & Label

**85 fitur** didefinisikan di [`config.py:FEATURE_COLS_V3`](config.py) — ini adalah sumber kebenaran tunggal. Jangan re-define di app.

Ringkasan grup fitur: OHLCV base (5) · Volume Flow (4) · Market Structure SMC (8) · OI & Funding (2) · EMA H1/H4 (8) · Momentum RSI/StochRSI (3) · ATR H1/H4 (2) · Key Levels (6) · Volume Profile (3) · Macro (3) · Returns & Vol Ratio (4) · Time Cyclical (5) · Long/Short Ratio (1, konstant 0 di inference) · Symbol Encoding (1) · Swing Structure v2 (4) · Market Regime v2 (3) · Smart Money v3 (9) · Smart Money v4 OFI/VWDP/CVD/VSA (14) · **Total = 85**

**Label:** `SHORT=0 / FLAT=1 / LONG=2` via **Swing-Based Labeling v3** ([`swing_based_labeling()`](core/features.py)) — TP/SL berbasis H4 swing high/low terdekat, bukan fixed ATR. Parameter: `MIN_RR=1.5`, `MIN_TP=1.5×ATR`, `MAX_SL=3.0×ATR`, `MAX_HOLD=48 bar H1`.

> **`long_short_ratio` = konstan 0 di inference** — kolom ini di-drop saat cleaning (`02_clean.py`) sehingga model belajar dari nilai 0. Jangan fetch live; set 0 langsung untuk konsistensi distribusi.

### 1.3 Aliran Data

```mermaid
flowchart LR
    A[Binance API] -->|fetch_klines| B[Raw Parquet<br/>OHLCV + Funding + OI]
    B -->|02_clean.py| C[Cleaned Parquet<br/>MTF Join H1+H4+H1D]
    C -->|03_engineer.py| D[Features V3 Parquet<br/>85 fitur + label]
    D -->|04_train_lgbm| E[LGBM Model .pkl]
    D -->|05_train_lstm| F[LSTM Model .pt<br/>+ scaler .pkl]
    E -->|06_ensemble| G[Ensemble Meta .pkl<br/>+ Calibrator]
    F -->|06_ensemble| G
    G -->|08_backtest| H[Backtest Results<br/>+ inference_config.json]
```

---

## 2. Arsitektur End-to-End

### 2.1 Diagram Arsitektur

```mermaid
flowchart TB
    subgraph OfflinePipeline[Offline Pipeline - Local/Colab]
        F1[01_fetch.py] --> F2[02_clean.py]
        F2 --> F3[03_engineer.py]
        F3 --> F4[04_train_lgbm.py]
        F3 --> F5[05_train_lstm.py]
        F4 --> F6[06_ensemble.py]
        F5 --> F6
        F6 --> F7[07_evaluate.py]
        F7 --> F8[08_backtest.py]
        F8 --> F9[09_holdout_backtest.py]
    end

    subgraph RailwayApp[Railway Hobby Tier - Single Service + Volume]
        subgraph FlaskApp[Flask + Gunicorn]
            API[API Routes<br/>Flask Blueprints]
            J2[Jinja2 Templates<br/>HTML Dashboard]
            SCH[APScheduler<br/>Periodic Jobs]
        end

        subgraph CoreModules[Core Modules - reuse existing]
            FEAT[features.py<br/>engineer_features]
            EVAL[evaluator.py<br/>simulate_trades]
            FETC[fetchers.py<br/>fetch latest data]
        end

        subgraph ModelServing[Model Serving - Lazy Load]
            LGBM_SRV[LGBM Service<br/>joblib load on demand]
            LSTM_SRV[LSTM Service<br/>torch load on demand]
            ENS_SRV[Ensemble Service<br/>meta + calibrator]
        end

        subgraph Storage[Storage Layer]
            NeonDB[(Neon PostgreSQL<br/>Free Tier 0.5 GB)]
            ModelFiles[Model Files<br/>PKL on Railway Volume]
            Cache[In-Memory Cache<br/>LRU dict]
        end

        subgraph PaperTrading[Paper Trading Engine]
            SIG[Signal Generator<br/>from model predictions]
            POS[Position Manager<br/>open/close/TP/SL]
            PNL[PnL Calculator<br/>realized + unrealized]
            LOG[Trade Logger<br/>to Neon PostgreSQL]
        end
    end

    OfflinePipeline -->|deploy models + scaler| ModelFiles
    API --> J2
    API --> ModelServing
    API --> PaperTrading
    SCH --> FETC
    SCH --> FEAT
    SCH --> SIG
    ModelServing --> Cache
    ModelServing --> ModelFiles
    PaperTrading --> NeonDB
    CoreModules --> FEAT
    CoreModules --> EVAL
    CoreModules --> FETC
```

### 2.2 Prinsip Desain

1. **Single-service constraint:** Semua komponen berjalan dalam satu proses Flask+Gunicorn di Railway
2. **Lazy loading:** Model dimuat ke memori hanya saat dibutuhkan, lalu di-cache dengan TTL
3. **Strategi model per fase:**
   - **Fase 1 (MVP Railway):** LSTM + LGBM — keduanya muat di 832 MB RAM (~200 MB total model memory). User bisa pilih per koin via `ModelSelection`.
   - **Fase 2:** Full Ensemble (LGBM + LSTM + MetaLearner + Calibrator) — jika RAM budget setelah Fase 1 stabil di bawah 600 MB
4. **Persistent volume:** Model files disimpan di Railway Volume — data tidak hilang saat redeploy
5. **Neon PostgreSQL sebagai single source of truth:** Semua state (trades, signals, model metrics) disimpan di Neon DB — terpisah dari Railway, tidak terpengaruh redeploy
6. **Pre-computed signals:** Scheduler generate sinyal periodik, bukan on-demand, untuk menghemat compute hours
7. **Plug-and-play model update:** Setiap run pipeline baru menghasilkan folder versi tersendiri (`models/v{run_id}/`). App mendeteksi versi baru dari `model_registry.json` dan memungkinkan aktivasi tanpa restart service.

---

## 3. Struktur Folder Proyek

Folder yang **sudah ada** (`core/`, `pipeline/`, `config.py`, `data/`) tidak diubah. Yang dibuat baru:

```
├── app/                         ← Web application (BARU)
│   ├── __init__.py              ← Flask app factory + init_scheduler()
│   ├── extensions.py            ← db (SQLAlchemy), utcnow() helper
│   ├── models/                  ← ORM: coin.py, model_meta.py, signal.py, trade.py, model_selection.py
│   ├── services/                ← inference.py, data_service.py, paper_trading.py, model_registry.py
│   ├── api/                     ← Blueprints: dashboard.py, coins.py, models_bp.py, signals.py, trades.py
│   ├── templates/               ← Jinja2: base.html, dashboard.html, coins.html, signals.html, trades.html, models.html
│   ├── static/                  ← style.css, app.js
│   └── jobs/                    ← fetch_latest.py, generate_signals.py, check_positions.py, update_metrics.py
│
├── models/                      ← Model files + versioning (sudah ada, diperbarui formatnya)
│   ├── model_registry.json      ← Index semua versi (format versioned, lihat §6.3)
│   └── v{YYYYMMDD_HHMMSS}/      ← Satu folder per run training
│       ├── lstm_best.pt
│       ├── lstm_scaler.pkl
│       └── inference_config.json
│
├── deploy/                      ← Deployment scripts
│   ├── migrate_registry.py      ← Konversi model_registry.json lama (flat) → format versioned (jalankan sekali, lihat §0.2)
│   ├── prepare_deploy.py        ← Copy model files dari source project → models/v{run_id}/, patch inference_config.json (tambah coins_validated.recommended, rename labeling → fallback_tp_sl)
│   └── seed_db.py               ← Seed Neon DB dengan data awal (coins, model_meta)
│
├── data/                        ← Railway Volume mount point
│   ├── inference/               ← Cached parquet per coin (250 bars)
│   └── models/                  ← Model files (symlink or copy from models/)
├── wsgi.py                      ← Gunicorn entry point
├── Procfile                     ← --workers 1 --threads 4
├── runtime.txt                  ← python-3.11.9
└── .env.example
```

---

## 4. Skema Database PostgreSQL (Neon)

> **Neon Free Tier:** 0.5 GB storage, auto-suspend after 5 min inactivity, built-in PgBouncer connection pooling. Database terpisah dari Railway — survive redeploy tanpa persistent volume.

### 4.1 Entity-Relationship Diagram

```mermaid
erDiagram
    COIN {
        int id PK
        string symbol UK
        string status
        datetime last_signal_at
        datetime last_updated
    }
    
    MODEL_META {
        int id PK
        int coin_id FK
        string model_type
        string run_id
        float accuracy
        float sharpe_ratio
        float profit_factor
        float f1_macro
        float win_rate
        int total_trades
        float max_drawdown
        string model_path
        string scaler_path
        string meta_learner_path
        string calibrator_path
        string inference_config_path
        string status
        datetime trained_at
        datetime evaluated_at
    }
    
    MODEL_SELECTION {
        int id PK
        int coin_id FK
        int model_meta_id FK
        datetime selected_at
    }
    
    SIGNAL {
        int id PK
        int coin_id FK
        int model_meta_id FK
        string direction
        float confidence
        float entry_price
        float tp_price
        float sl_price
        float atr_at_signal
        string timeframe
        json feature_snapshot
        datetime signal_time
        datetime created_at
    }
    
    TRADE {
        int id PK
        int signal_id FK
        int coin_id FK
        string direction
        float entry_price
        float exit_price
        float tp_price
        float sl_price
        float quantity
        float leverage
        float fee_total
        float pnl_gross
        float pnl_net
        float pnl_pct
        string exit_reason
        string status
        datetime opened_at
        datetime closed_at
        int hold_bars
    }
    
    COIN ||--o{ MODEL_META : has
    COIN ||--o{ SIGNAL : generates
    COIN ||--o{ TRADE : executes
    COIN ||--|| MODEL_SELECTION : active_model
    MODEL_META ||--o{ SIGNAL : produced_by
    MODEL_META ||--o{ MODEL_SELECTION : selected_as
    SIGNAL ||--o| TRADE : triggers
```

### 4.2 SQL DDL

```sql
-- Coins
CREATE TABLE coin (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL UNIQUE,
    status          TEXT DEFAULT 'active',  -- active, inactive, delisted
    last_signal_at  TIMESTAMPTZ,
    last_updated    TIMESTAMPTZ DEFAULT NOW()
);

-- Model metadata (one row per trained model per coin)
-- Untuk LSTM (Fase 1): isi model_path + scaler_path; meta_learner_path + calibrator_path = NULL
-- Untuk Ensemble (Fase 2): semua path diisi
CREATE TABLE model_meta (
    id                    SERIAL PRIMARY KEY,
    coin_id               INTEGER NOT NULL REFERENCES coin(id),
    model_type            TEXT NOT NULL,           -- lgbm, lstm, ensemble
    run_id                TEXT,                    -- format: YYYYMMDD_HHMMSS
    accuracy              DOUBLE PRECISION,
    sharpe_ratio          DOUBLE PRECISION,
    profit_factor         DOUBLE PRECISION,
    f1_macro              DOUBLE PRECISION,
    win_rate              DOUBLE PRECISION,
    total_trades          INTEGER,
    max_drawdown          DOUBLE PRECISION,
    model_path            TEXT,                    -- path ke .pt (LSTM) atau .pkl (LGBM)
    scaler_path           TEXT,                    -- path ke lstm_scaler.pkl (NULL untuk LGBM)
    meta_learner_path     TEXT,                    -- path ke ensemble_meta.pkl (NULL kecuali ensemble)
    calibrator_path       TEXT,                    -- path ke calibrator.pkl (NULL kecuali ensemble)
    inference_config_path TEXT,                    -- path ke inference_config.json versi ini
    n_features            INTEGER DEFAULT 85,      -- lock jumlah fitur saat training
    status                TEXT DEFAULT 'available', -- available, active, deprecated
    trained_at            TIMESTAMPTZ,
    evaluated_at          TIMESTAMPTZ,
    UNIQUE(coin_id, model_type, run_id)
);

-- Active model selection per coin
CREATE TABLE model_selection (
    id              SERIAL PRIMARY KEY,
    coin_id         INTEGER NOT NULL UNIQUE REFERENCES coin(id),
    model_meta_id   INTEGER NOT NULL REFERENCES model_meta(id),
    selected_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Trading signals
CREATE TABLE signal (
    id              SERIAL PRIMARY KEY,
    coin_id         INTEGER NOT NULL REFERENCES coin(id),
    model_meta_id   INTEGER REFERENCES model_meta(id),
    direction       TEXT NOT NULL,           -- LONG, SHORT, FLAT
    confidence      DOUBLE PRECISION,        -- 0.0 - 1.0
    entry_price     DOUBLE PRECISION,
    tp_price        DOUBLE PRECISION,
    sl_price        DOUBLE PRECISION,
    atr_at_signal   DOUBLE PRECISION,
    timeframe       TEXT DEFAULT '1h',
    feature_snapshot TEXT,                   -- JSON of key features at signal time
    signal_time     TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_signal_coin_time ON signal(coin_id, signal_time DESC);

-- Paper trades
CREATE TABLE trade (
    id              SERIAL PRIMARY KEY,
    signal_id       INTEGER REFERENCES signal(id),
    coin_id         INTEGER NOT NULL REFERENCES coin(id),
    direction       TEXT NOT NULL,           -- LONG, SHORT
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    tp_price        DOUBLE PRECISION,
    sl_price        DOUBLE PRECISION,
    quantity        DOUBLE PRECISION DEFAULT 1.0,
    leverage        DOUBLE PRECISION DEFAULT 3.0,
    fee_total       DOUBLE PRECISION DEFAULT 0.0,
    pnl_gross       DOUBLE PRECISION,
    pnl_net         DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    exit_reason     TEXT,                    -- tp_hit, sl_hit, time_exit, manual_close
    status          TEXT DEFAULT 'open',     -- open, closed
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    hold_bars       INTEGER
);
CREATE INDEX idx_trade_status ON trade(status);
CREATE INDEX idx_trade_coin ON trade(coin_id, opened_at DESC);

-- Performance summary (materialized view, updated periodically)
CREATE TABLE performance_summary (
    id              SERIAL PRIMARY KEY,
    coin_id         INTEGER NOT NULL REFERENCES coin(id),
    period          TEXT NOT NULL,           -- 7d, 30d, all
    total_trades    INTEGER,
    win_count       INTEGER,
    loss_count      INTEGER,
    win_rate        DOUBLE PRECISION,
    total_pnl       DOUBLE PRECISION,
    avg_pnl         DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(coin_id, period)
);
```

---

## 5. Preprocessing untuk Inference

### 5.1 Alur Inference

Inference menggunakan kembali modul yang sudah ada di [`core/features.py`](core/features.py) dan [`core/fetchers.py`](core/fetchers.py), dengan adaptasi untuk data real-time:

```mermaid
flowchart TD
    A[Scheduled Job Trigger] --> B[Fetch Latest Data<br/>fetchers.fetch_klines 1h/4h/1d<br/>fetchers.fetch_funding_rate<br/>fetchers.fetch_all_macro]
    B --> C[Clean + MTF Join<br/>reuse 02_clean logic]
    C --> D[Feature Engineering<br/>features.engineer_features<br/>85 fitur V3]
    D --> E[Prepare Model Input<br/>- LGBM: DataFrame FEATURE_COLS_V3<br/>- LSTM: Scale + Sequence LSTM_SEQ_LEN=32]
    E --> F[Model Prediction<br/>- LGBM: predict_proba<br/>- LSTM: forward pass<br/>- Ensemble: stack + logistic]
    F --> G[Post-Processing<br/>- Confidence check >= 0.60<br/>- Signal stability filter<br/>- Circuit breaker check]
    G --> H{Signal?}
    H -->|LONG/SHORT| I[Create Signal Record<br/>+ Calculate TP/SL from ATR]
    H -->|FLAT| J[No Action]
    I --> K[Paper Trading Engine<br/>Check cooldown → Open Position]
```

### 5.2 Adaptasi dari Pipeline Existing

Komponen yang **di-reuse langsung** tanpa modifikasi:

| Modul | Fungsi | Penggunaan |
|-------|--------|------------|
| [`core/binance_client.py`](core/binance_client.py) | `BinanceClient` | Fetch data dari Binance API |
| [`core/fetchers.py`](core/fetchers.py) | `fetch_klines()`, `fetch_funding_rate()`, `fetch_coin()` | Ambil data terbaru |
| [`core/features.py`](core/features.py) | `engineer_features()`, `calc_atr()`, `calc_rsi()` | Hitung 85 fitur |
| [`core/evaluator.py`](core/evaluator.py) | `simulate_trades_swing()`, `full_trading_report()` | Evaluasi performa (gunakan v3 swing, bukan v2 fixed ATR) |
| [`core/models.py`](core/models.py) | `TradingLSTM`, `load_lstm()`, `ProbabilityCalibrator` | Load + inferensi model |
| [`config.py`](config.py) | `FEATURE_COLS_V3`, `LABEL_MAP_INV`, semua parameter | Konfigurasi |

> **Catatan:** Gunakan `simulate_trades_swing()` (v3), bukan `simulate_trades()` (v2). Model dilatih dengan swing-based labeling sehingga evaluasi dan TP/SL inference harus konsisten menggunakan H4 swing high/low.

**Sumber data yang wajib di-fetch untuk inference (semua dibutuhkan `engineer_features()`):**

| Data | Fungsi Fetch | Fitur yang Dihasilkan | Catatan |
|------|-------------|----------------------|---------|
| Klines 1h, 4h, 1d | `fetch_coin()` | OHLCV, EMA, ATR, swing structure | |
| Funding rate | `fetch_coin()` → `fetch_funding_rate()` | `funding_rate`, `time_to_funding_norm` | |
| BTC dominance | `fetch_all_macro()` | `btc_dominance` | |
| Fear & Greed | `fetch_all_macro()` | `fear_greed` | |
| ~~Long/short ratio~~ | ~~Fetch terpisah~~ | `long_short_ratio` | **Set konstan 0** — lihat catatan di bawah |

> **`long_short_ratio` = 0 di inference (by design):** Di `02_clean.py` kolom ini di-drop secara eksplisit (`"tidak tersedia di H1"`). Saat training, kolom diisi NaN → ffill → 0. Untuk konsistensi distribusi fitur antara training dan inference, **jangan fetch live** — set saja `long_short_ratio = 0` di `InferenceDataService`. Fetch live akan menghasilkan distribusi berbeda dari apa yang model pelajari.

Komponen yang **perlu dibuat baru** untuk inference (`app/services/data_service.py`):

**`InferenceDataService`** — dua method utama:

| Method | Tanggung Jawab | Return |
|--------|---------------|--------|
| `prepare_latest_features(symbol, n_bars=250)` | Fetch klines 1h/4h/1d + macro → clean + MTF join → `engineer_features()` → set `long_short_ratio=0` | `DataFrame` (85 fitur + h4_swing_high/low) atau `None` jika data tidak cukup |
| `prepare_lstm_input(df, scaler)` | Scale `df[FEATURE_COLS_V3]` → sequence `LSTM_SEQ_LEN=32` terakhir → reshape | `ndarray (1, 32, 85)` |

> **Minimum bars:** `LSTM_SEQ_LEN (32) + SYNTHETIC_OI_NORM_WINDOW (168) + buffer = 250`. Return `None` jika data tidak cukup — caller wajib handle gracefully (skip, retry next interval).

> **n_features = 85** untuk `prepare_lstm_input` — `ndarray (1, 32, 85)`. Guard di `InferenceService`: `model_meta.n_features == len(FEATURE_COLS_V3)` harus == 85, jika beda raise error eksplisit.

### 5.3 Manajemen Data Inference

Database state disimpan di **Neon PostgreSQL** (external, survive redeploy). Model files dan inference cache disimpan di **Railway Volume**:

- **Neon PostgreSQL:** Semua state (trades, signals, model_meta, performance_summary) — terpisah dari Railway, auto-suspend setelah 5 menit inactivity, wake-up ~1 detik
- **Railway Volume `/data`:** Model files + inference parquet cache — tetap ada setelah redeploy
- **Strategi utama:** Simpan **last 250 bar** per koin dalam format parquet di `/data/inference/`
- **Restart / redeploy:** DB state tetap ada di Neon. Volume data tetap ada. Hanya jika volume kosong (fresh deploy), `prepare_latest_features()` return `None` → job skip → di-fetch ulang pada jadwal berikutnya.

> **Neon cold start:** Neon auto-suspend setelah 5 menit tidak ada query. Request pertama setelah suspend mengalami ~1 detik delay untuk wake-up. Ini acceptable untuk paper trading personal — tidak ada data loss.

---

## 6. Model Serving Strategy

### 6.1 Pilihan Model untuk Railway

| Model | RAM Saat Load | Inference Time | Fase | Status Railway |
|-------|---------------|----------------|------|----------------|
| LSTM (.pt + scaler.pkl) | ~100-150 MB | ~50 ms | **Fase 1** | ✅ **Default — sequence model utama** |
| LightGBM (.pkl) | ~30-50 MB | <10 ms | **Fase 1** | ✅ **Alternatif ringan — deploy bersama LSTM** |
| Ensemble (LGBM+LSTM+Meta+Calibrator) | ~200-250 MB | ~100 ms | Fase 2 | ⚠️ Perlu validasi RAM setelah Fase 1 stabil |

> **Rationale Fase 1 = LSTM + LGBM:** Dengan 832 MB RAM, kedua model muat bersamaan (~200 MB total). LSTM unggul untuk sequence context, LGBM unggul untuk speed dan koin tertentu. User pilih per koin via `ModelSelection`.

### 6.2 Strategi Lazy Loading & Inference Pipeline

**Model Cache** (`app/services/inference.py`):
- Thread-safe dict dengan TTL 30 menit, kapasitas max 2 model (LSTM + LGBM bersamaan)
- Key: `f"{model_type}_{run_id}"` — model yang berbeda run_id tidak konflik
- Saat model di-switch via `POST /models/select`: cache di-clear, model lama otomatis unload

**Keputusan desain kritis:**
- Semua parameter inference (threshold, seq_len, label_map) **dibaca dari `inference_config.json`** milik model yang aktif — tidak ada hardcode di app
- `FEATURE_COLS_V3` **diimport dari `config.py` pipeline** — tidak di-define ulang
- Guard wajib saat load: `model_meta.n_features == len(FEATURE_COLS_V3)` — jika mismatch, raise error eksplisit (bukan silent wrong prediction)
- `predict()` return `None` saat cold start (data belum tersedia) — caller wajib handle gracefully, tidak raise exception

**Pipeline per model type:**

| Model Type | Pipeline Inference |
|------------|-------------------|
| **lstm** (Fase 1) | `features_df` → scale → sequence `(1, 32, 85)` → LSTM forward → softmax → proba |
| **lgbm** (Fase 2) | `features_df[FEATURE_COLS_V3].iloc[-1]` → `predict_proba()` → proba |
| **ensemble** (Fase 3) | LGBM proba `(1,3)` + LSTM proba `(1,3)` → concat `(1,6)` → MetaLearner → ProbabilityCalibrator → proba |

### 6.3 Plug-and-Play Model Update

Setiap kali pipeline training baru selesai, model baru dapat diaktifkan **tanpa restart service** dan **tanpa overwrite model lama**.

#### Struktur Versioning

```
models/
├── model_registry.json           ← index semua versi, dibaca scheduler
├── v20250430_143000/             ← format: v{YYYYMMDD_HHMMSS}
│   ├── lstm_best.pt
│   ├── lstm_scaler.pkl
│   ├── lgbm_baseline.pkl
│   ├── ensemble_meta.pkl         ← NULL jika bukan ensemble run
│   ├── calibrator.pkl            ← NULL jika bukan ensemble run
│   └── inference_config.json     ← snapshot semua parameter saat itu
└── v20250501_090000/             ← versi berikutnya
    └── ...
```

#### Format `model_registry.json`

```json
{
  "versions": [
    {
      "run_id": "20260425_170250",
      "model_type": "ensemble",
      "n_features": 85,
      "trained_at": "2026-04-25T17:02:50Z",
      "evaluated_at": "2026-04-25T17:02:50Z",
      "backtest_summary": {
        "mean_winrate": 0.7718,
        "mean_drawdown_lev3x": 0.1445
      },
      "coins_validated": [
        "1000PEPEUSDT", "DOGEUSDT", "1000SHIBUSDT", "ADAUSDT", "TRXUSDT",
        "ETHUSDT", "POLUSDT", "SUIUSDT", "LINKUSDT", "SOLUSDT",
        "XRPUSDT", "DOTUSDT", "TONUSDT", "ARBUSDT", "AVAXUSDT"
      ],
      "paths": {
        "lgbm":             "models/v20260425_170250/lgbm_baseline.pkl",
        "lstm":             "models/v20260425_170250/lstm_best.pt",
        "scaler":           "models/v20260425_170250/lstm_scaler.pkl",
        "meta":             "models/v20260425_170250/ensemble_meta.pkl",
        "calibrator":       "models/v20260425_170250/calibrator.pkl",
        "inference_config": "models/v20260425_170250/inference_config.json"
      },
      "status": "available"
    }
  ]
}
```

> **Format ini dihasilkan oleh `deploy/migrate_registry.py`** dari `model_registry.json` lama. `run_id` diturunkan dari field `created_at` di `inference_config.json` (format `YYYYMMDD_HHMMSS`). `coins_validated` diambil dari `inference_config["coins_validated"]["recommended"]` setelah patching.

#### Alur Deteksi & Aktivasi

Job `update_metrics` (setiap 6 jam) membaca `model_registry.json`, lalu untuk setiap entry `status='available'` yang belum ada di `MODEL_META`:
1. Validasi `n_features == len(FEATURE_COLS_V3)` — tolak otomatis jika mismatch
2. Validasi file model ada di disk dan `evaluated_at` tidak None
3. Insert ke `MODEL_META` dengan `status='available'`
4. `/api/health` menampilkan badge "model baru tersedia"

User aktifkan manual via `POST /models/select` atau dropdown di UI `/models`.

### 6.4 Model Selection UI

Endpoint `POST /models/select` memungkinkan user memilih model aktif per koin:

```
POST /models/select
Body: {
    "symbol": "SOLUSDT",
    "model_type": "lstm",
    "run_id": "20250430_143000"   // ← tambahkan run_id agar bisa pilih versi spesifik
}
```

Saat model di-switch:
1. Validasi `n_features` model baru == `len(FEATURE_COLS_V3)`
2. Clear model cache (unload model lama)
3. Update `model_selection` table (set model lama jadi `deprecated` jika diminta)
4. Model baru akan di-load pada inference berikutnya

### 6.5 Konversi ke ONNX (Direkomendasikan sejak Fase 1)

Untuk menghindari dependency PyTorch CPU (~300MB disk) di Railway, konversi LSTM ke ONNX Runtime (~30MB). **Direkomendasikan sejak Fase 1** mengingat batas disk Railway 1 GB.

`deploy/prepare_deploy.py:export_lstm_to_onnx()` — load `TradingLSTM` → `torch.onnx.export()` dengan `dummy_input = torch.randn(1, 32, 85)`, `opset_version=17`, dynamic batch axis.

**Keputusan implementasi:**
- **Fase 1 (MVP):** Gunakan PyTorch langsung — lebih simple, sudah ada dependency
- **Fase 2 (Optimasi):** Konversi ke ONNX untuk menghemat ~500MB disk space dan ~30MB RAM

---

## 7. Definisi Endpoint API

### 7.1 Tabel Endpoint

| Method | Endpoint | Deskripsi | Template |
|--------|----------|-----------|----------|
| GET | `/` | Redirect ke dashboard | — |
| GET | `/dashboard` | Dashboard utama | `dashboard.html` |
| GET | `/coins` | Daftar koin + model metrics | `coins.html` |
| GET | `/coins/<symbol>` | Detail koin + sinyal terbaru | `coin_detail.html` |
| GET | `/models` | Daftar model tersedia | `models.html` |
| POST | `/models/select` | Pilih model aktif per koin | — (JSON) |
| GET | `/paper/signals` | Log sinyal trading | `signals.html` |
| GET | `/paper/signals/<id>` | Detail sinyal | — (JSON) |
| GET | `/paper/trades` | Riwayat trade + PnL | `trades.html` |
| GET | `/paper/trades/<id>` | Detail trade | — (JSON) |
| POST | `/paper/trades/<id>/close` | Tutup posisi manual | — (JSON) |
| GET | `/api/health` | Health check + badge model baru | — (JSON) |
| GET | `/api/stats` | Statistik ringkasan | — (JSON) |
| GET | `/api/models/available` | Daftar versi model di registry | — (JSON) |

### 7.2 Detail Endpoint

| Endpoint | Data yang Ditampilkan | Catatan |
|----------|----------------------|---------|
| `GET /dashboard` | Koin aktif, model aktif per koin, ringkasan PnL (7d/30d/all), sinyal terakhir, trade terbuka, PnL chart | Server-side render via Jinja2 |
| `GET /coins` | Tabel: Symbol, Active Model, Accuracy, Sharpe, PF, Win Rate, Last Signal, Status | Join `Coin` + `ModelSelection` + `ModelMeta` |
| `POST /models/select` | Body: `{symbol, model_type, run_id}` → validasi → update `ModelSelection` → clear model cache | Return JSON status |
| `GET /paper/signals` | Tabel sinyal + filter (coin, direction, date range) + pagination 50/page | Kolom: Time, Coin, Direction, Confidence, Entry, TP, SL, ATR |
| `GET /paper/trades` | Tabel trade + filter (coin, status, direction) + summary cards (PnL, WR, PF, Sharpe) | Join `performance_summary` untuk ringkasan |
| `POST /paper/trades/<id>/close` | Tutup posisi manual → hitung PnL → update DB | Return JSON status |

---

## 8. Dashboard UI

### 8.1 Layout

Menggunakan **Tailwind CSS via CDN** (tanpa build step) + **HTMX** untuk interaktivitas ringan tanpa framework JS berat:

```
┌─────────────────────────────────────────────────────────┐
│  🤖 Trading Assistant          [Dashboard] [Coins]      │
│                                 [Signals] [Trades]      │
│                                 [Models]                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ 20 Coins │ │ 3 Open   │ │ +12.5%   │ │ 0.87     │  │
│  │ Active   │ │ Trades   │ │ 30d PnL  │ │ Sharpe   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                         │
│  ┌─ Latest Signals ──────────────────────────────────┐  │
│  │ SOLUSDT  LONG   0.82  $148.50  TP=$155  SL=$142  │  │
│  │ ETHUSDT  SHORT  0.75  $2,850   TP=$2,750 SL=$2920│  │
│  │ BNBUSDT  FLAT   0.45  —        —        —        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─ Open Positions ──────────────────────────────────┐  │
│  │ SOLUSDT  LONG  Entry:$145  TP:$155  SL:$142  +2% │  │
│  │ XRPUSDT  SHORT Entry:$2.1  TP:$1.95 SL:$2.2  -1% │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─ Model Performance ───────────────────────────────┐  │
│  │ Symbol   │ Model │ Acc  │ Sharpe │ PF   │ WR     │  │
│  │ SOLUSDT  │ LSTM  │ 68%  │ 1.24   │ 1.85 │ 62%   │  │
│  │ ETHUSDT  │ LSTM  │ 65%  │ 0.98   │ 1.42 │ 58%   │  │
│  │ BNBUSDT  │ LGBM  │ 70%  │ 1.35   │ 2.01 │ 64%   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Teknologi Frontend

| Komponen | Teknologi | Alasan |
|----------|-----------|--------|
| CSS Framework | Tailwind CSS CDN | Ringan, tanpa build step |
| Interaktivitas | HTMX CDN | AJAX tanpa JS framework, ~14KB |
| Chart | Chart.js CDN | PnL chart sederhana |
| Template | Jinja2 | Built-in Flask, server-side rendering |

### 8.3 Model Selection UI

Di halaman `/models`, setiap koin memiliki dropdown HTMX: `<select hx-post="/models/select" hx-vals='{"symbol": "..."}'>` dengan opsi LSTM/LGBM/Ensemble. Status badge di-update via `hx-target` tanpa page reload.

---

## 9. Modul Paper Trading

### 9.1 Alur Lengkap Paper Trading

```mermaid
sequenceDiagram
    participant SCH as APScheduler
    participant FET as Fetchers
    participant FEAT as Features
    participant INF as InferenceService
    participant PT as PaperTradingEngine
    participant DB as Neon PostgreSQL

    Note over SCH: Setiap 5 menit (MONITOR_POLL_INTERVAL_SECS)

    SCH->>FET: 1. fetch latest klines 1h/4h/1d
    FET-->>SCH: raw data parquet

    SCH->>FEAT: 2. engineer_features
    FEAT-->>SCH: features DataFrame (85 cols)

    SCH->>INF: 3. predict(symbol, features)
    INF-->>SCH: {direction, confidence, tp, sl}

    SCH->>PT: 4. process_signal(signal)
    
    PT->>DB: 5. Save signal record
    
    alt direction = LONG or SHORT
        PT->>DB: 6a. Check cooldown (SAME_DIR_COOLDOWN_HOURS)
        PT->>DB: 6b. Check circuit breaker (VCB)
        
        alt Cooldown OK AND No circuit break
            PT->>DB: 7. Open trade (status=open)
            Note over PT: entry_price=close<br/>tp=entry + ATR*2<br/>sl=entry - ATR*1
        else Cooldown active OR Circuit break
            Note over PT: Skip - no new position
        end
    else direction = FLAT
        Note over PT: No action
    end

    Note over SCH: Setiap 1 menit (position check)

    SCH->>PT: 8. check_open_positions()
    
    loop For each open trade
        PT->>FET: 9a. Get current price
        FET-->>PT: current candle
        
        alt LONG position
            alt current_high >= tp_price
                PT->>DB: 10a. Close trade (exit_reason=tp_hit)
            else current_low <= sl_price
                PT->>DB: 10b. Close trade (exit_reason=sl_hit)
            else hold_bars >= MAX_HOLDING_BARS
                PT->>DB: 10c. Close trade (exit_reason=time_exit)
            end
        else SHORT position
            alt current_low <= tp_price
                PT->>DB: 10a. Close trade (exit_reason=tp_hit)
            else current_high >= sl_price
                PT->>DB: 10b. Close trade (exit_reason=sl_hit)
            else hold_bars >= MAX_HOLDING_BARS
                PT->>DB: 10c. Close trade (exit_reason=time_exit)
            end
        end
        
        PT->>PT: 11. Calculate PnL
        Note over PT: pnl_gross = direction * (exit - entry)<br/>fee = 2 * FEE_PER_SIDE * modal<br/>pnl_net = pnl_gross - fee
        PT->>DB: 12. Update trade record
    end

    SCH->>DB: 13. Update performance_summary
```

### 9.2 Paper Trading Engine

`app/services/paper_trading.py` — logika incremental (bar per bar), bukan batch seperti [`core/evaluator.py:simulate_trades_swing()`](core/evaluator.py).

**`process_signal(signal)` → buka posisi atau skip:**

1. Confidence < `CONFIDENCE_THRESHOLD_ENTRY` → skip
2. Direction = FLAT → skip
3. Cooldown: cek last closed trade same direction → jika < `SAME_DIR_COOLDOWN_HOURS` → skip
4. VCB: jika `VCB_ENABLED` dan `_circuit_breaker_active()` → skip
5. Sudah ada open position untuk coin ini → skip
6. Buka posisi: `entry_price=close`, `tp/sl` dari signal, `quantity=MODAL_PER_TRADE`, `leverage=LEVERAGE_SIM[0]`, `fee=2*FEE_PER_SIDE*MODAL_PER_TRADE`

**`check_open_positions()` → dipanggil scheduler setiap 5 menit:**

1. Untuk setiap open trade, fetch current candle (high/low/close)
2. LONG: `high >= tp` → tp_hit · `low <= sl` → sl_hit
3. SHORT: `low <= tp` → tp_hit · `high >= sl` → sl_hit
4. Jika tidak hit TP/SL dan `hold_bars >= MAX_HOLDING_BARS` → time_exit at close
5. PnL: `pnl_gross = direction * (exit - entry) / entry * qty * leverage` · `pnl_net = pnl_gross - fee`

**`_circuit_breaker_active(coin_id)` — buat dari scratch (belum ada di evaluator):**
- Ambil `VCB_LOOKBACK_BARS=24` bar H1 terakhir
- Jika `ATR_current > VCB_ATR_MULTIPLIER * ATR_mean` → return True (skip new positions)

### 9.3 Perhitungan TP/SL

> **Penting:** Model dilatih dengan **swing-based labeling** (v3), bukan fixed ATR multiplier. TP/SL saat inference harus menggunakan logika yang sama agar konsisten dengan asumsi training.

**`calculate_tp_sl_swing()` — logika:**

| Direction | TP | SL |
|-----------|----|----|
| LONG | Nearest H4 swing **high** above entry | Nearest H4 swing **low** below entry |
| SHORT | Nearest H4 swing **low** below entry | Nearest H4 swing **high** above entry |

**Validasi (jika gagal → return `None, None` → signal di-skip):**
- TP distance ≥ `SWING_LABEL_MIN_TP` (1.5) × ATR
- SL distance ≤ `SWING_LABEL_MAX_SL` (3.0) × ATR
- Risk-Reward = TP_dist / SL_dist ≥ `SWING_LABEL_MIN_RR` (1.5)

**Fallback** (hanya jika swing levels tidak tersedia): LONG `TP = entry + 2.0×ATR`, `SL = entry - 1.0×ATR` · SHORT sebaliknya.

> **`inference_config["fallback_tp_sl"]` = fallback values only:** Field `tp_atr_mult` dan `sl_atr_mult` di `inference_config.json` (setelah di-patch oleh `prepare_deploy.py`, section-nya bernama `fallback_tp_sl`) adalah nilai fallback ini — **bukan** parameter swing-based labeling. Jangan gunakan sebagai primary TP/SL logic; hanya invoke jika `calculate_tp_sl_swing()` return `None, None`.

---

## 10. Job Scheduler

### 10.1 Job Definitions

Menggunakan **APScheduler** (BackgroundScheduler) yang berjalan dalam proses yang sama dengan Flask:

| Job | Interval | Fungsi | Est. RAM Spike |
|-----|----------|--------|----------------|
| `fetch_latest` | Setiap 5 menit | Fetch klines terbaru + macro data | ~30 MB |
| `generate_signals` | Setiap 1 jam | Engineer fitur + inference + create signals | ~80 MB (LSTM) |
| `check_positions` | Setiap 1 menit | Cek TP/SL hit untuk open trades | ~5 MB |
| `update_metrics` | Setiap 6 jam | Refresh performance summary | ~10 MB |
| `cleanup_cache` | Setiap 30 menit | Evict expired cache entries | ~0 MB |

### 10.2 Implementasi Scheduler

`app/jobs/__init__.py` — `init_scheduler(app)` dipanggil sekali dari `create_app()`.

> **Kritis:** Procfile harus `--workers 1`. Multi-worker → setiap worker spawn scheduler sendiri → job jalan N kali → sinyal duplikat + RAM spike berlipat.

| Job ID | Interval | next_run_time | Catatan |
|--------|----------|---------------|---------|
| `fetch_latest` | 15 menit | immediate | H1 candle, tidak perlu lebih sering |
| `generate_signals` | 4 jam | +15 menit dari start | Pastikan fetch_latest sudah selesai sebelum generate pertama |
| `check_positions` | 5 menit | immediate | Cukup untuk paper trading H1 |
| `update_metrics` | 6 jam | immediate | Juga scan model baru dari registry |

### 10.3 Design Decisions: generate_signals Job

**Kenapa serial (bukan parallel)?**
- `ModelCache` hanya pegang 1 model sekaligus (RAM constraint 512MB)
- Parallelisme tidak menghemat RAM — justru memperparahnya
- Delay kecil antar koin memberi waktu GC untuk reclaim memory

**Kenapa hanya `recommended` coins?**
- Batasi ke koin dengan winrate ≥ 63% (dari `inference_config.json`)
- Mengurangi total inference time dan RAM pressure

**Alur per koin:**
1. Baca `inference_config.json` → ambil `coins_validated.recommended`
2. Untuk setiap symbol: `InferenceService.predict(symbol)` → jika `None` (cold start), skip
3. Simpan `Signal` record ke DB → `PaperTradingEngine.process_signal(signal)`
4. `gc.collect()` setelah setiap koin (reclaim tensor memory)
5. Jika error pada satu koin → log + lanjut ke berikutnya (jangan abort seluruh job)

> **`coins_validated.recommended` hanya ada setelah `prepare_deploy.py` dijalankan (§0.4).** Raw `inference_config.json` dari project training hanya punya `high_priority`, `medium_priority`, dan `caution` — tidak ada `recommended`. Script patching menggabungkan `high_priority + medium_priority` menjadi `recommended`. Tanpa langkah §0.2, job ini akan `KeyError` saat pertama kali jalan.

### 10.4 Staggering Timeline

```
Menit 0:   App start — fetch_latest BELUM jalan
Menit 1-14: Dashboard bisa diakses, tapi sinyal = kosong (cold start window)
Menit 15:  fetch_latest pertama — data H1 tersedia
Menit 15:  generate_signals pertama (dijadwal dari next_run_time)
Menit 20:  check_positions mulai berjalan bermakna

Per jam (contoh jam 08:00):
  08:00  check_positions
  08:05  check_positions
  08:10  check_positions
  08:15  fetch_latest + check_positions
  ...
  Setiap 4 jam: generate_signals (setelah fetch)
```

---

## 11. Caching & Manajemen Memori

### 11.1 Budget Memori (832 MB Total)

| Komponen | RAM Idle | RAM Peak | Keterangan |
|----------|----------|----------|------------|
| Flask + Gunicorn (**1 worker**, 4 threads) | ~25 MB | ~40 MB | 1 worker — wajib untuk APScheduler |
| Neon PostgreSQL client | ~10 MB | ~15 MB | Connection pool via PgBouncer |
| APScheduler | ~5 MB | ~10 MB | Berjalan dalam worker yang sama |
| LSTM Model (loaded) | ~100 MB | ~150 MB | Lazy loaded, TTL 30 min |
| LGBM Model (loaded) | ~30 MB | ~50 MB | Lazy loaded, TTL 30 min |
| Feature DataFrame (1 coin, **250 bars**, 85 cols) | ~2 MB | ~8 MB | |
| Inference temp tensors | ~5 MB | ~20 MB | Di-gc.collect() setelah setiap koin |
| Python runtime + libs | ~80 MB | ~120 MB | numpy, pandas, torch, lightgbm |
| **Total** | **~252 MB** | **~418 MB** | **Aman dalam 832 MB — sisa ~414 MB** |

> **Kedua model (LSTM + LGBM) di-load di Fase 1.** Ensemble (Fase 2) menambah ~50 MB untuk MetaLearner + Calibrator.

### 11.2 Strategi Caching

`app/services/cache.py` — Thread-safe LRU cache dengan TTL (OrderedDict + Lock).

| Cache Instance | max_size | TTL | Isi |
|----------------|----------|-----|-----|
| `model_cache` | 2 | 30 menit | Model objects (LSTM + LGBM) — max 2 loaded |
| `feature_cache` | 5 | 5 menit | Feature DataFrame per coin (250 bars × 85 cols) |
| `signal_cache` | 1 | 1 menit | Latest signals summary |

Operasi: `get(key)` → return value atau None (expired) · `put(key, value)` → evict oldest jika over capacity · `clear()` → dipanggil saat model switch atau memory pressure.

### 11.3 Manajemen Memori Kritis

`app/services/memory.py` — dipanggil sebelum setiap job berat:

1. Cek RSS via `psutil.Process().memory_info().rss`
2. Jika > 665 MB (80% dari 832 MB): clear `model_cache` + `feature_cache` → `gc.collect()` → log warning
3. Cek ulang RSS setelah cleanup → log hasilnya

`get_memory_status()` → return `{rss_mb, limit_mb, pct}` untuk dashboard dan `/api/health`.

### 11.4 PostgreSQL Connection & Timezone

`app/extensions.py` — SQLAlchemy config untuk Neon:

- **Connection pooling:** Neon menyediakan PgBouncer built-in — gunakan connection string pooling (`?sslmode=require`) dari Neon dashboard
- **Pool size:** `SQLALCHEMY_POOL_SIZE = 5`, `SQLALCHEMY_MAX_OVERFLOW = 2` — cukup untuk 1 worker + scheduler
- **Pool pre-ping:** `SQLALCHEMY_POOL_PRE_PING = True` — handle Neon auto-suspend gracefully (detect stale connections)

> **Timezone convention:** Semua kolom `TIMESTAMPTZ` menyimpan UTC secara otomatis. Gunakan helper `utcnow()` berbasis `datetime.now(timezone.utc)` — **jangan campur** dengan `datetime.utcnow()` (naive, deprecated) atau `datetime.now()` (local time). PostgreSQL enforce timezone pada `TIMESTAMPTZ`, tapi tetap best practice untuk selalu pass timezone-aware datetime.

---

## 12. Deployment ke Railway

### 12.1 File Konfigurasi Deployment

#### `Procfile`
```
web: gunicorn wsgi:app --workers 1 --threads 4 --timeout 120 --max-requests 500 --max-requests-jitter 50
```

**Penjelasan:**
- `--workers 1`: **Satu worker saja** — ini kritis. Multi-worker menyebabkan APScheduler diinisialisasi di setiap worker → job `generate_signals` jalan dua kali bersamaan → sinyal duplikat di DB. Dengan 832 MB RAM, 1 worker + 4 thread lebih dari cukup untuk personal assistant.
- `--threads 4`: 4 thread per worker untuk handle concurrent requests tanpa spawn proses baru
- `--timeout 120`: Timeout 120s untuk inference LSTM yang bisa butuh ~50-100ms
- `--max-requests 500`: Restart worker setelah 500 requests untuk mencegah memory leak bertahap
- `--max-requests-jitter 50`: Randomize restart timing

#### `runtime.txt`
```
3.11.9
```

#### `requirements.txt` (updated)
```
# Existing pipeline deps
pandas>=2.0
numpy>=1.26
pyarrow>=14.0
requests>=2.31
lightgbm>=4.0
scikit-learn>=1.4
joblib>=1.3
torch>=2.1
python-dotenv>=1.0

# Web app deps
flask>=3.0
flask-sqlalchemy>=3.1
psycopg2-binary>=2.9
gunicorn>=21.2
apscheduler>=3.10
psutil>=5.9
```

#### `.env.example`
```
# Flask
FLASK_ENV=production
SECRET_KEY=change-me-to-random-string

# Database (Neon PostgreSQL — dari Neon dashboard)
DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/trading_db?sslmode=require

# Binance API (public endpoints — tidak perlu API key untuk futures data)
BINANCE_BASE_URL=https://fapi.binance.com

# Scheduler
SIGNAL_INTERVAL_HOURS=4
POSITION_CHECK_INTERVAL_MINUTES=5
FETCH_INTERVAL_MINUTES=15

# Model
DEFAULT_MODEL_TYPE=lstm
MODEL_CACHE_TTL_SECONDS=1800
MAX_LOADED_MODELS=1

# Memory
MEMORY_LIMIT_MB=400
MEMORY_CHECK_ENABLED=true
```

#### `wsgi.py`
Standard Gunicorn entry: `from app import create_app; app = create_app()`. Port dari `PORT` env var.

### 12.2 Langkah Deployment

```mermaid
flowchart TD
    A[1. Jalankan training pipeline<br/>04→05→06→07→08→09] --> B[2. migrate_registry.py<br/>Konversi model_registry.json<br/>ke format versioned]
    B --> C[3. prepare_deploy.py<br/>- Buat models/v{run_id}/<br/>- Copy .pt + scaler.pkl<br/>- Copy inference_config.json ke folder versi]
    C --> D[4. Verifikasi lokal<br/>- inference_config.json terbentuk<br/>- n_features == 85<br/>- model_registry.json format baru]
    D --> E[5. Commit + Push ke GitHub]
    E --> F[6. Railway: Create New Service<br/>- Connect GitHub repo<br/>- Select branch]
    F --> G[7. Neon: Create Database<br/>- Sign up neon.tech<br/>- Create project + database<br/>- Copy connection string]
    G --> H[8. Railway: Create Volume<br/>- Mount at /data<br/>- For model files + inference cache]
    H --> I[9. Railway: Set Environment Variables<br/>- SECRET_KEY<br/>- DATABASE_URL=postgresql://...neon.tech<br/>- DEFAULT_MODEL_TYPE=lstm<br/>- WORKERS=1]
    I --> J[10. Railway: Deploy<br/>- Auto-detect Procfile --workers 1 --threads 4<br/>- Install requirements<br/>- Start gunicorn]
    J --> K[11. Verify /api/health<br/>- scheduler_running=true<br/>- models_loaded ok<br/>- Neon DB connected]
    K --> L[12. Monitor<br/>- /dashboard<br/>- Railway metrics RAM < 665MB<br/>- Neon dashboard: connection count]
```

### 12.3 Strategi Menghemat Compute Hours

Railway Hobby tier menggunakan **usage-based billing** ($5/bulan base, termasuk 832 MB RAM + 1 vCPU). Strategi efisiensi:

1. **Satu worker** (`--workers 1`) — cukup untuk personal assistant dengan traffic rendah

2. **Interval job yang sudah dikalibrasi:**
   - `check_positions`: **5 menit** (cukup untuk paper trading H1)
   - `fetch_latest`: **15 menit** (H1 candle baru tiap 60 menit — 15 menit sangat cukup)
   - `generate_signals`: **4 jam** (timeframe trading adalah H1, tidak perlu tiap jam)

3. **Pre-compute daripada on-demand:** Sinyal di-generate oleh scheduler, bukan saat user request. Dashboard hanya baca dari Neon PostgreSQL.

4. **Optimasi imports:** Lazy import `torch` hanya saat `InferenceService` pertama kali di-instantiate.

### 12.4 Disk Space Management (1 GB)

**PyTorch (direkomendasikan untuk Hobby tier — disk tidak terbatas seperti free tier):**

| Komponen | Estimasi Ukuran |
|----------|----------------|
| Python runtime + pip | ~200 MB |
| PyTorch CPU | ~300 MB |
| NumPy + Pandas + other libs | ~100 MB |
| Model files LSTM + LGBM + scaler (1 versi aktif) | ~30 MB |
| Neon PostgreSQL (external) | 0 MB local | DB di Neon, bukan di disk Railway |
| Flask + templates + static | ~5 MB |
| **Total** | **~655-665 MB** |

> **Hobby tier tidak memiliki batas disk ketat** — PyTorch langsung fine. ONNX konversi tetap didokumentasikan di §6.5 sebagai optimasi opsional jika ingin menghemat ~270 MB disk di masa depan.

**Signal rotation policy (Neon free tier = 0.5 GB, perlu dijaga agar tidak penuh):**
- Hapus `signal` records > 90 hari secara otomatis di job `update_metrics`
- Batas maksimum 10.000 rows di tabel `signal` (hapus oldest jika terlampaui)
- Estimasi: ~10K signals × ~1 KB = ~10 MB — jauh di bawah 0.5 GB limit
- Backup `trade` records ke separate JSON file sebelum hapus (opsional, trade count kecil)

### 12.5 Monitoring & Alerting

`GET /api/health` — return JSON: `{status, memory_mb, memory_pct, scheduler_running, active_trades, total_signals, models_loaded}`. Railway dapat memonitor endpoint ini untuk auto-restart jika `memory_pct > 80%` (>665 MB).

### 12.6 Fallback Strategy

Jika Railway Hobby tier perlu di-scale up:

| Opsi | Kapan | Cara |
|------|-------|------|
| **Upgrade ke Railway Pro** | RAM > 800 MB stabil, perlu lebih banyak | Scale up di Railway dashboard |
| **VPS murah ($5/bulan)** | Butuh full control, custom domain, lebih dari 1 service | Deploy manual via Docker |

---

## Ringkasan Implementasi

### Prioritas Pengembangan

| Fase | Scope | Target |
|------|-------|--------|
| **Fase 0: Pra-implementasi** | Jalankan pipeline 04→05→06→07→08→09, verifikasi `inference_config.json` + `model_registry.json` terbentuk | Semua model files ready |
| **Fase 1: MVP Railway** | Flask + Neon PostgreSQL + **LSTM + LGBM inference** + Swing TP/SL + VCB + Paper Trading + Dashboard + Plug-and-play versioning + Railway Volume | Deploy ke Railway Hobby + Neon Free |
| **Fase 2: Ensemble** | Full 4-stage inference (LGBM+LSTM+Meta+Calibrator) + Advanced caching | Akurasi sinyal lebih tinggi |
| **Fase 3: Polish** | Chart.js PnL chart + alert notifications + signal rotation + Performance analytics + optional ONNX | Production-ready |

### Checklist Sebelum Build Fase 1

**Pre-build (sebelum tulis satu baris kode app):**
- [x] Training pipeline selesai — model files sudah ada di project training terpisah (lihat §0.1)
- [ ] Jalankan `deploy/migrate_registry.py --src <path_lama>/model_registry.json` — konversi format lama ke versioned (lihat §0.3 issue #2)
- [ ] Jalankan `deploy/prepare_deploy.py --src <path_lama>/models` — copy + patch + buat `models/v{run_id}/` (lihat §0.2)
- [ ] Verifikasi `models/v{run_id}/inference_config.json` berisi `n_features: 85` dan key `coins_validated.recommended`
- [ ] Verifikasi `models/model_registry.json` format versioned: `{"versions": [{"run_id": ..., "paths": {...}}]}`
- [ ] Konfirmasi `long_short_ratio` = 0 di semua training data → **set konstan 0 di inference, tidak di-fetch**

**Saat build app:**
- [ ] `Procfile`: `--workers 1 --threads 4` (APScheduler constraint — bukan 2 workers)
- [ ] Neon DB: create project + database, copy `DATABASE_URL` connection string
- [ ] Railway Volume: create + mount at `/data` (for model files + inference cache, bukan DB)
- [ ] Semua `datetime` gunakan helper `utcnow()` berbasis `datetime.now(timezone.utc)` — PostgreSQL `TIMESTAMPTZ` otomatis handle timezone
- [ ] `SQLALCHEMY_POOL_PRE_PING = True` — handle Neon auto-suspend (stale connection detection)
- [ ] App baca parameter dari `inference_config.json` — **tidak ada hardcode** threshold, seq_len, label_map
- [ ] `FEATURE_COLS_V3` diimport dari `config.py` pipeline — tidak di-define ulang di app
- [ ] DB schema menyertakan: `meta_learner_path`, `calibrator_path`, `inference_config_path`, `n_features`
- [ ] `generate_signals` job: serial per koin, `gc.collect()` setelah setiap koin, hanya koin dari `recommended` list
- [ ] `prepare_latest_features()` return `None` jika data tidak cukup — semua caller handle `None` gracefully
- [ ] TP/SL menggunakan `calculate_tp_sl_swing()` berbasis H4 swing
- [ ] VCB `_circuit_breaker_active()` diimplementasikan
- [ ] Signal rotation: max 10K rows / hapus records > 90 hari di job `update_metrics`