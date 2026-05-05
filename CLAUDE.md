# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dev server
python wsgi.py                          # http://localhost:5000

# Production (gunicorn --workers 1 wajib, APScheduler in-process)
gunicorn wsgi:app --workers 1 --threads 4 --timeout 120

# Database
python reset_db.py                      # Drop & recreate + auto-seed

# Deploy
python deploy/prepare_deploy.py         # Copy model files + patch config
python deploy/deploy_model.py --source <path>  # Import training output

# Health check
curl http://localhost:5001/api/health
```

Tidak ada test suite atau linter.

## Arsitektur

SwingTrade v2 — paper trading bot untuk Binance Futures. ML ensemble (LSTM + LightGBM + MetaLearner + Calibrator) dengan 85 fitur dari H1/H4 candles. Flask + APScheduler + PostgreSQL (Neon) / SQLite fallback.

### Alur utama

```
APScheduler (in-process):
  fetch_latest (15m)       → cache engineered features ke parquet
  generate_signals (1h)    → inference per koin → Signal DB → paper_trading → Trade DB
  check_positions (5m)     → cek TP/SL semua open trades
  update_metrics (6h)      → refresh performance_summary + rotasi sinyal lama
```

### Inference pipeline

Semua model via `InferenceService` (`app/services/inference.py`), support 4 `model_type`:
- **lstm** — LSTM standalone, 32 bar sequence, softmax
- **lgbm** — LightGBM standalone, 1 bar
- **ensemble** — LGBM + LSTM → MetaLearner → Calibrator
- **cascade** — LGBM scout (cepat, 1 bar) → LSTM confirmer (dalam, 32 bar). Kalau LGBM bilang FLAT / tidak yakin, skip LSTM

### Paper trading flow (`app/services/paper_trading.py`)

```
signal → confidence check → FLAT check → VCB → existing trade? → TP/SL calc → sizing tier → open Trade
```

TP/SL: swing-based (H4 high/low) dengan minimum distance enforcement (TP ≥ 2%, SL ≥ 1.5% dari entry). Fallback ke ATR multiplier jika swing tidak tersedia.

Position sizing by confidence: conf > 0.75 → full size ($100), conf 0.60–0.75 → half ($50), conf < 0.60 → skip.

### Model Registry

`models/model_registry.json` — source of truth untuk versi model yang tersedia. `models/inference_config.json` — semua parameter (confidence thresholds, feature list, cascade thresholds, backtest stats, VCB settings). `config_loader.py` singleton yang membaca file ini; panggil `reload_cache()` untuk reload tanpa restart.

### Database

Tidak ada Alembic. Migrasi via `_run_migrations()` di `app/__init__.py` — `ALTER TABLE ADD COLUMN IF NOT EXISTS` pada startup.

Model: `Coin` → `Signal` (1:N), `Coin` → `Trade` (1:N), `Coin` → `ModelMeta` (1:N), `Coin` → `ModelSelection` (1:1), `Signal` → `ModelMeta` (N:1).

### Constraints produksi

- Railway Hobby 832 MB RAM → serial per-coin, `gc.collect()` tiap koin, max 2 model di cache
- `--workers 1` wajib — APScheduler in-process, multi-worker = duplicate jobs

### WITA

Semua waktu tampilan pakai WITA (UTC+8). Jinja filter `wita_fmt` untuk konversi.

## Slash Commands

### /audit

Ketika user mengetik `/audit`, lakukan audit menyeluruh pada kode di branch saat ini:
1. Cek semua perhitungan persentase (PnL%, TP%, SL%, drawdown%) — pastikan berbasis modal, bukan hasil leverage
2. Cek TP/SL calculation — pastikan minimum distance enforcement aktif
3. Cek cascade thresholds — pastikan scout_flat, scout_signal, confirmer threshold terkonfigurasi
4. Cek notifikasi Telegram — pastikan semua field (model_type, TP%, SL%) muncul
5. Cek query N+1 — pastikan semua query pakai `joinedload` untuk relasi yang diakses di template
6. Report temuan dalam format checklist (✅/⚠️/❌)
