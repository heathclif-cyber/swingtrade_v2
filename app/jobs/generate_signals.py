"""
app/jobs/generate_signals.py — Inference per koin → simpan Signal → proses paper trading.

Dipanggil scheduler setiap 1 jam.
Serial per koin (bukan paralel) — RAM constraint 832 MB.
gc.collect() setelah setiap koin untuk reclaim tensor memory.
"""

import gc
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
from flask import Flask

logger = logging.getLogger(__name__)


def run(app: Flask) -> None:
    with app.app_context():
        from app.extensions import db, utcnow
        from app.models.coin import Coin
        from app.models.signal import Signal
        from app.models.model_meta import ModelMeta
        from app.models.model_selection import ModelSelection
        from app.services.data_service import InferenceDataService
        from app.services.inference import InferenceService
        from app.services.paper_trading import PaperTradingEngine
        from app.services.model_registry import get_active_version
        from app.services.memory import check_and_free

        check_and_free()

        version = get_active_version()
        logger.info(
            f"[generate_signals] Mulai — versi={version['model_type'] if version else 'NONE'}, "
            f"run_id={version['run_id'] if version else 'N/A'}"
        )

        if not version:
            logger.error("[generate_signals] Tidak ada versi model aktif di registry")
            return

        # Ambil semua koin aktif — tidak hanya recommended
        coins = Coin.query.filter(
            Coin.status == "active",
        ).all()

        if not coins:
            logger.warning("[generate_signals] Tidak ada koin aktif di database")
            return

        logger.info(f"[generate_signals] {len(coins)} koin aktif akan diproses")
        data_svc = InferenceDataService()
        engine   = PaperTradingEngine()
        ok = skip = fail = 0

        for coin in coins:
            try:
                status = _process_coin(coin, data_svc, engine, db, utcnow,
                                       Signal, ModelMeta, ModelSelection)
                if status == "skip":
                    skip += 1
                elif status == "fail":
                    fail += 1
                    logger.warning(f"[generate_signals] {coin.symbol} inference gagal (predict=None)")
                else:
                    ok += 1
            except Exception as e:
                logger.error(f"[generate_signals] {coin.symbol} error tak tertangani: {e}", exc_info=True)
                fail += 1
            finally:
                gc.collect()
                time.sleep(0.3)

        logger.info(f"[generate_signals] Selesai: {ok} prediksi tersimpan, {skip} skip, {fail} gagal / {len(coins)} koin")


def _process_coin(coin, data_svc, engine, db, utcnow,
                  Signal, ModelMeta, ModelSelection):
    from app.services.inference import InferenceService  # Import di sini

    symbol = coin.symbol

    # Ambil model_meta aktif untuk koin ini
    sel = ModelSelection.query.filter_by(coin_id=coin.id).first()
    if not sel:
        logger.warning(f"[{symbol}] Tidak ada ModelSelection — skip")
        return "skip"
    logger.debug(f"[{symbol}] ModelSelection id={sel.id} → model_meta_id={sel.model_meta_id}")

    meta = ModelMeta.query.get(sel.model_meta_id)
    if not meta:
        logger.warning(f"[{symbol}] ModelMeta id={sel.model_meta_id} tidak ditemukan — skip")
        return "skip"
    logger.debug(f"[{symbol}] ModelMeta: type={meta.model_type!r}")

    # Fetch + engineer features
    logger.debug(f"[{symbol}] Mulai fetch features...")
    features_df = data_svc.prepare_latest_features(symbol)
    if features_df is None:
        logger.warning(f"[{symbol}] features_df=None — kemungkinan data Binance tidak cukup atau engineer_features gagal")
        return "skip"
    logger.info(f"[{symbol}] Features OK: {features_df.shape[0]} bars × {features_df.shape[1]} cols")

    # Inference — gunakan model_type dari ModelMeta (single source of truth)
    model_type = meta.model_type or "lstm"
    logger.debug(f"[{symbol}] Mulai inference model_type={model_type!r}...")
    svc = InferenceService(meta)
    result = svc.predict(symbol, features_df, model_type=model_type)
    if result is None:
        logger.warning(f"[{symbol}] predict=None — inference gagal atau exception (lihat log di atas)")
        return "fail"

    direction  = result["direction"]
    confidence = result["confidence"]
    entry      = result["entry_price"]
    atr        = result["atr_value"]
    proba      = result.get("proba", [])
    logger.info(
        f"[{symbol}] Prediksi: direction={direction} conf={confidence:.4f} "
        f"proba={[f'{p:.3f}' for p in proba]} entry={entry:.4f} atr={atr:.4f}"
    )

    # Simpan Signal
    swing_high = result.get("h4_swing_high") or 0.0
    swing_low  = result.get("h4_swing_low")  or 0.0

    # Full 85+ fitur dari baris terakhir untuk RL training
    feature_dict = {}
    last_row = features_df.iloc[-1]
    for col in features_df.columns:
        val = last_row[col]
        if isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = float(val) if not np.isnan(val) else None
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        feature_dict[col] = val

    signal = Signal(
        coin_id          = coin.id,
        model_meta_id    = meta.id,
        direction        = direction,
        confidence       = confidence,
        entry_price      = entry,
        atr_at_signal    = atr,
        h4_swing_high    = swing_high if swing_high > 0 else None,
        h4_swing_low     = swing_low  if swing_low  > 0 else None,
        tp_price         = None,
        sl_price         = None,
        timeframe        = "1h",
        feature_snapshot = json.dumps(feature_dict),
        signal_time = utcnow(),
    )
    db.session.add(signal)
    db.session.flush()  # dapat signal.id sebelum commit

    # Update last_signal_at di coin
    coin.last_signal_at = utcnow()

    # Paper trading — buka posisi jika signal bukan FLAT
    if direction in ("LONG", "SHORT"):
        # Hitung TP/SL untuk sinyal (selalu, bahkan jika trade tidak dibuka)
        last_row = features_df.iloc[-1] if features_df is not None else None
        tp, sl = engine._calculate_tp_sl(direction, entry, atr, last_row)
        if tp is not None:
            signal.tp_price = tp
        if sl is not None:
            signal.sl_price = sl

        logger.debug(f"[{symbol}] Mengirim ke PaperTradingEngine arah={direction}...")
        trade = engine.process_signal(signal, features_df)
        if trade:
            signal.tp_price = trade.tp_price
            signal.sl_price = trade.sl_price
            logger.info(f"[{symbol}] Trade DIBUKA: tp={trade.tp_price:.4f} sl={trade.sl_price:.4f}")
        else:
            logger.warning(
                f"[{symbol}] Trade TIDAK dibuka meski signal {direction} — "
                f"cek: cooldown / VCB / TP-SL / posisi terbuka"
            )
    else:
        logger.debug(f"[{symbol}] Signal FLAT — tidak ada trade")

    db.session.commit()
    logger.info(f"[{symbol}] Signal={direction} conf={confidence:.2f} entry={entry:.4f} → tersimpan (id={signal.id})")

    # Kirim notifikasi Telegram untuk signal LONG/SHORT
    if direction in ("LONG", "SHORT"):
        try:
            from app.services.telegram import get_telegram_service
            tg = get_telegram_service()
            tg.send_signal_alert(signal, symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Gagal kirim notifikasi Telegram: {e}")

    return "ok"
