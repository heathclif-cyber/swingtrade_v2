"""
deploy/prepare_deploy.py

Copy model files dari models/ root → models/v{run_id}/ dan patch inference_config.json:
  - Rename section "labeling" → "fallback_tp_sl"
  - Pastikan coins_validated.recommended tersedia

Usage:
    python deploy/prepare_deploy.py            # pakai models/ local
    python deploy/prepare_deploy.py --dry-run  # preview tanpa menulis file
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

REQUIRED_FILES = [
    "lgbm_baseline.pkl",
    "lstm_best.pt",
    "lstm_scaler.pkl",
    "ensemble_meta.pkl",
    "calibrator.pkl",
    "inference_config.json",
]


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        sys.exit(f"[ERROR] {REGISTRY_PATH} tidak ditemukan.")
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    if "versions" not in registry or not registry["versions"]:
        sys.exit("[ERROR] model_registry.json bukan format versioned. Jalankan migrate_registry.py dulu.")
    return registry


def patch_inference_config(config: dict) -> dict:
    patched = dict(config)

    # Rename "labeling" → "fallback_tp_sl"
    if "labeling" in patched and "fallback_tp_sl" not in patched:
        patched["fallback_tp_sl"] = patched.pop("labeling")
        print("  [patch] 'labeling' -> 'fallback_tp_sl'")
    elif "fallback_tp_sl" in patched:
        print("  [skip]  'fallback_tp_sl' sudah ada")
    else:
        print("  [warn]  key 'labeling' tidak ditemukan — skip rename")

    # Pastikan coins_validated.recommended ada
    cv = patched.get("coins_validated", {})
    if isinstance(cv, dict) and "recommended" not in cv:
        high = cv.get("high_priority", [])
        med = cv.get("medium_priority", [])
        cv["recommended"] = high + med
        patched["coins_validated"] = cv
        print(f"  [patch] coins_validated.recommended ditambah ({len(high + med)} koin)")
    else:
        print("  [skip]  coins_validated.recommended sudah ada")

    return patched


def prepare(run_id: str, src_dir: Path, dry_run: bool) -> Path:
    versioned_dir = MODELS_DIR / f"v{run_id}"

    print(f"\n[prepare_deploy] run_id={run_id}")
    print(f"  src : {src_dir}")
    print(f"  dest: {versioned_dir}")

    # Validasi file sumber
    missing = [f for f in REQUIRED_FILES if not (src_dir / f).exists()]
    if missing:
        sys.exit(f"[ERROR] File tidak ditemukan di {src_dir}: {missing}")

    if dry_run:
        print("\n[DRY-RUN] File yang akan di-copy:")
        for fname in REQUIRED_FILES:
            print(f"  {src_dir / fname} -> {versioned_dir / fname}")
        print("\n[DRY-RUN] inference_config.json akan di-patch:")
        with open(src_dir / "inference_config.json") as f:
            config = json.load(f)
        patch_inference_config(config)
        return versioned_dir

    # Buat folder versioned
    versioned_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[copy] Membuat {versioned_dir}")

    # Copy semua model files
    for fname in REQUIRED_FILES:
        src = src_dir / fname
        dst = versioned_dir / fname
        if fname == "inference_config.json":
            continue  # ditangani terpisah setelah patch
        shutil.copy2(src, dst)
        print(f"  [copy] {fname}")

    # Patch dan simpan inference_config.json
    with open(src_dir / "inference_config.json") as f:
        config = json.load(f)

    print("\n[patch] inference_config.json:")
    patched = patch_inference_config(config)

    dst_config = versioned_dir / "inference_config.json"
    with open(dst_config, "w") as f:
        json.dump(patched, f, indent=2)
    print(f"  [save] {dst_config}")

    return versioned_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare model versioned folder untuk deploy")
    parser.add_argument("--dry-run", action="store_true", help="Preview tanpa menulis file")
    args = parser.parse_args()

    registry = load_registry()
    version = registry["versions"][0]
    run_id = version["run_id"]

    versioned_dir = prepare(run_id, src_dir=MODELS_DIR, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n[OK] Selesai. Versioned folder: {versioned_dir}")
        print("     Langkah berikutnya: jalankan deploy/seed_db.py untuk setup Neon PostgreSQL.")


if __name__ == "__main__":
    main()
