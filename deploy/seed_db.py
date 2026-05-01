"""
deploy/seed_db.py

Seed Neon DB dengan data awal:
  - Insert coins dari inference_config.json (coins_validated.recommended + acceptable)
  - Insert ModelMeta dari model_registry.json (versi pertama)
  - Set ModelSelection default (ensemble) untuk setiap coin

Usage:
    python deploy/seed_db.py
    python deploy/seed_db.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


def load_configs():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    version = registry["versions"][0]
    run_id = version["run_id"]
    config_path = MODELS_DIR / f"v{run_id}" / "inference_config.json"
    if not config_path.exists():
        config_path = MODELS_DIR / "inference_config.json"

    with open(config_path) as f:
        inference_config = json.load(f)

    return version, inference_config


def get_coins_to_seed(inference_config: dict) -> list[str]:
    cv = inference_config.get("coins_validated", {})
    recommended = cv.get("recommended", [])
    acceptable = cv.get("acceptable", [])
    return list(dict.fromkeys(recommended + acceptable))  # deduplicate, preserve order


def seed(dry_run: bool) -> None:
    version, inference_config = load_configs()
    symbols = get_coins_to_seed(inference_config)

    print(f"[seed_db] model_type={version.get('model_type')}")
    print(f"[seed_db] Coins to seed: {symbols}")

    if dry_run:
        print("\n[DRY-RUN] Tidak ada perubahan ke DB.")
        return

    from app import create_app
    from app.extensions import db, utcnow
    from app.models import Coin, ModelMeta, ModelSelection

    app = create_app()
    bs = inference_config.get("backtest_summary", {})
    paths = version.get("paths", {})
    trained_at = datetime.fromisoformat(
        inference_config.get("created_at", "2026-04-25T17:02:50+00:00")
    )

    with app.app_context():
        inserted_coins = 0
        inserted_metas = 0
        inserted_selections = 0

        for symbol in symbols:
            # Upsert coin
            coin = Coin.query.filter_by(symbol=symbol).first()
            if not coin:
                coin = Coin(symbol=symbol, status="active")
                db.session.add(coin)
                db.session.flush()
                inserted_coins += 1
                print(f"  [coin] INSERT {symbol}")
            else:
                print(f"  [coin] SKIP {symbol} (sudah ada)")

            # Insert ModelMeta jika belum ada
            model_type_val = version.get("model_type", "ensemble")
            meta = ModelMeta.query.filter_by(
                coin_id=coin.id, model_type=model_type_val
            ).first()

            if not meta:
                per_coin = inference_config.get("backtest_per_coin", {}).get(symbol, {})
                meta = ModelMeta(
                    coin_id=coin.id,
                    model_type=model_type_val,
                    win_rate=per_coin.get("winrate", bs.get("mean_winrate")),
                    total_trades=per_coin.get("total_trades"),
                    max_drawdown=per_coin.get("dd_lev3x", bs.get("mean_drawdown_lev3x")),
                    n_features=version.get("n_features", 85),
                    model_path=paths.get("lstm"),
                    scaler_path=paths.get("scaler"),
                    meta_learner_path=paths.get("meta"),
                    calibrator_path=paths.get("calibrator"),
                    inference_config_path=paths.get("inference_config"),
                    status="available",
                    trained_at=trained_at,
                    evaluated_at=trained_at,
                )
                db.session.add(meta)
                db.session.flush()
                inserted_metas += 1
                print(f"  [meta] INSERT ModelMeta {model_type_val} for {symbol}")
            else:
                print(f"  [meta] SKIP ModelMeta (sudah ada) for {symbol}")

            # Set ModelSelection jika belum ada
            sel = ModelSelection.query.filter_by(coin_id=coin.id).first()
            if not sel:
                sel = ModelSelection(coin_id=coin.id, model_meta_id=meta.id)
                db.session.add(sel)
                inserted_selections += 1
                print(f"  [sel]  INSERT ModelSelection for {symbol}")
            else:
                print(f"  [sel]  SKIP ModelSelection (sudah ada) for {symbol}")

        db.session.commit()
        print(f"\n[OK] Selesai. Coins={inserted_coins} ModelMeta={inserted_metas} Selections={inserted_selections}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
