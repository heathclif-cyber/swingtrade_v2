# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dev server
python wsgi.py                          # http://localhost:5000

# Production (gunicorn --workers 1 wajib, APScheduler in-process)
gunicorn wsgi:app --workers 1 --threads 4 --timeout 120 --max-requests 500 --max-requests-jitter 50

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
  update_metrics (6h)      → insert snapshot performance_summary + rotasi sinyal lama
```

### Inference pipeline

Semua model via `InferenceService` (`app/services/inference.py`), support 4 `model_type`:
- **lstm** — LSTM standalone, 32 bar sequence, softmax. Direkomendasikan — LGBM terlalu bias FLAT.
- **lgbm** — LightGBM standalone, 1 bar. Prediksi FLAT ~77% waktu (sesuai distribusi label training).
- **ensemble** — LGBM + LSTM → MetaLearner → Calibrator
- **cascade** — LGBM scout (1 bar) → LSTM confirmer (32 bar). 3-stage threshold dari config:
  - `scout_flat_threshold` (default 0.85): LGBM FLAT & confidence ≥ ini → skip LSTM
  - `scout_signal_threshold` (default 0.55): LGBM LONG/SHORT & confidence < ini → FLAT
  - `confirmer_threshold` (default 0.65): LSTM confidence ≥ ini → terkonfirmasi

**Catatan cascade:** Label training ~77% FLAT. LGBM belajar distribusi ini dan hampir selalu predict FLAT dengan confidence 0.9+. Akibatnya LSTM jarang dipanggil. Naikkan `scout_flat_threshold` ke 0.95+ agar lebih banyak sinyal lolos ke LSTM, atau pakai `lstm` standalone.

Cascade config dapat diedit via UI di halaman `/models` → "Cascade Configuration" atau via API `POST /api/cascade-config`.

### Paper trading flow (`app/services/paper_trading.py`)

```
signal → confidence check → cooldown check → FLAT check → VCB → existing trade? → TP/SL calc → sizing tier → open Trade
```

**TP/SL calculation** (`_calculate_tp_sl`): Swing-based (H4 high/low) dengan ATR sebagai floor.
- Stage 1: Hitung ATR-based TP/SL (`fallback_tp_sl.tp_atr_mult=2.0`, `sl_atr_mult=1.5`)
- Stage 2: Hitung swing-based TP/SL (H4 high/low)
- Final: TP = yang lebih jauh dari entry antara swing vs ATR, SL = yang lebih jauh juga.  
  Ini memastikan saat H4 sideways (swing terlalu dekat), ATR mengambil alih.

**Position sizing:** conf > 0.75 → full size ($100), conf 0.60–0.75 → half ($50), conf < 0.60 → skip. Fee selalu $0.08 (2 × 0.0004 × $100).

**Cooldown** (setelah trade close, mencegah re-entry arah sama): `tp_hit=2h`, `time_exit=2h`, `sl_hit=4h`, default=4h. Dikonfigurasi via `inference_config.cooldown`.

**VCB (Volatility Circuit Breaker):** Skip trade jika ATR saat ini > `vcb.atr_multiplier` × ATR mean 24 bar. Config di `volatility_circuit_breaker`.

**PnL:** `pnl_net = price_change × modal × leverage - fee`, `pnl_pct = pnl_net / modal × 100`. Berbasis modal, bukan harga mentah.

**TP%/SL% display:** Semua tampilan TP% dan SL% (signals page, CSV, API, Telegram) sudah dikalikan leverage (default 5x), jadi menampilkan return ke modal, bukan pergerakan harga.

### Performance Monitoring

**Charts (Chart.js CDN):**
- Dashboard (`/dashboard`): Equity Curve 60d (cumulative PnL $) + Rolling Win Rate 10-day (% WR + baseline 50%)
- Coin detail (`/coins/<symbol>`): Equity Curve + Drawdown overlay (dual-axis)

**PerformanceSummary snapshot history:**
- `update_metrics` job (6h) INSERT baris baru, bukan overwrite — history terlacak
- Cleanup: 1 snapshot/hari/coin/period, hapus >90 hari
- Query selalu ambil snapshot terbaru via subquery `max(snapshot_at)`
- Coin detail menampilkan tabel "Performance Trend" (14 snapshot terakhir) dengan indikator ↗/↘/→

**API endpoints:**
- `GET /api/equity-curve` — daily PnL + rolling WR (semua koin)
- `GET /api/equity-curve/<symbol>` — daily equity + drawdown% (per koin)
- `GET /api/cascade-config` / `POST /api/cascade-config` — baca/edit threshold cascade
- `GET /api/health` — status scheduler, memory, model cache, signal count

### Model Registry

`models/model_registry.json` — source of truth untuk versi model yang tersedia. `models/inference_config.json` — semua parameter (confidence thresholds, feature list, cascade thresholds, backtest stats, VCB settings, TP/SL multipliers). `config_loader.py` singleton yang membaca file ini; panggil `reload_cache()` untuk reload tanpa restart.

### Database

Tidak ada Alembic. Migrasi via `_run_migrations()` di `app/__init__.py` — `ALTER TABLE ADD COLUMN IF NOT EXISTS` pada startup.

Model: `Coin` → `Signal` (1:N), `Coin` → `Trade` (1:N), `Coin` → `ModelMeta` (1:N), `Coin` → `ModelSelection` (1:1), `Coin` → `PerformanceSummary` (1:N, snapshot history), `Signal` → `ModelMeta` (N:1).

Unique constraint `uq_perf_coin_period` dihapus (via migration) untuk mendukung snapshot history.

### UI / Frontend

- **Tailwind CSS CDN** + **HTMX** + **Chart.js** (CDN)
- Container: `max-w-[1400px]` (diperbesar dari 1280px untuk tabel 16 kolom)
- Copy CSV: dropdown pilihan baris (50/100/200/400/All) di semua halaman (signals, trades, coins)
- Signals page: multi-select filter coin & arah (comma-separated URL params, tahan refresh)
- Signals page: FLAT ditampilkan secara default (sebelumnya difilter)
- Trades table: kolom Closed, TP, SL, Hold bars ditambahkan

### Constraints produksi

- Railway Hobby 832 MB RAM → serial per-coin, `gc.collect()` tiap koin, max 2 model di cache
- `--workers 1` wajib — APScheduler in-process, multi-worker = duplicate jobs
- Model cache TTL = 30 menit (`MODEL_CACHE_TTL_SECONDS`), reload setiap run karena interval 60 menit > TTL

### WITA

Semua waktu tampilan pakai WITA (UTC+8). Jinja filter `wita_fmt` untuk konversi.

## Slash Commands

### /audit

Ketika user mengetik `/audit`, lakukan audit menyeluruh pada kode di branch saat ini:
1. Cek semua perhitungan persentase (PnL%, TP%, SL%, drawdown%) — pastikan berbasis modal, bukan hasil leverage. TP%/SL% display harus dikali leverage.
2. Cek TP/SL calculation — pastikan ATR sebagai floor aktif (swing vs ATR, yang lebih jauh yang dipakai)
3. Cek cascade thresholds — pastikan `scout_flat`, `scout_signal`, `confirmer` threshold terkonfigurasi di `inference_config.json`
4. Cek notifikasi Telegram — pastikan semua field (model_type, TP%, SL%) muncul, TP%/SL% dikali leverage
5. Cek query N+1 — pastikan semua query pakai `joinedload` untuk relasi yang diakses di template. Dashboard dan coin detail pakai subquery untuk snapshot terbaru.
6. Cek PerformanceSummary — pastikan job INSERT (bukan upsert), snapshot_at terisi, cleanup berjalan
7. Cek grafik — pastikan API equity-curve mengisi semua hari (termasuk hari kosong = 0), tidak ada null/undefined di tooltip
8. Report temuan dalam format checklist (✅/⚠️/❌)
