"""
deploy/deploy_model.py — Deployment Bridge Script

Jembatan antara training pipeline output dan web app.
Membaca model_registry.json dari folder source, menyalin 7 file model ke
root models/, dan menulis model_registry.json dengan format yang web app mengerti.

Tidak ada folder backup — semua file langsung di models/ (seamless).

Usage:
    python deploy/deploy_model.py                                    # deploy dari folder kustom
    python deploy/deploy_model.py --source D:/Download/workspace/models
    python deploy/deploy_model.py --dry-run                          # preview
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY   = MODELS_DIR / "model_registry.json"

REQUIRED_FILES = [
    "lgbm_baseline.pkl",
    "lstm_best.pt",
    "lstm_scaler.pkl",
    "ensemble_meta.pkl",
    "calibrator.pkl",
    "inference_config.json",
    "feature_cols_v2.json",
]


def load_new_registry(path: Path) -> dict:
    """Baca model_registry.json format baru: {active, models}."""
    with open(path) as f:
        data = json.load(f)

    # Deteksi format
    if "versions" in data:
        print("  [info] Format LAMA (versions) — tidak perlu konversi")
        return data

    if "models" not in data:
        print(f"  [error] Format tidak dikenal di {path}")
        print(f"  Key yang ada: {list(data.keys())}")
        sys.exit(1)

    return data


def get_active_model_info(data: dict) -> dict:
    """Ekstrak informasi model aktif dari registry format baru/lama."""
    if "versions" in data:
        # Format lama — ambil versi aktif pertama
        for v in data["versions"]:
            if v.get("status") in ("available", "active"):
                return v
        print("  [error] Tidak ada versi dengan status active/available")
        sys.exit(1)

    # Format baru
    active_name = data.get("active")
    if not active_name:
        print("  [error] Key 'active' tidak ditemukan di registry")
        sys.exit(1)

    model_data = data.get("models", {}).get(active_name)
    if not model_data:
        print(f"  [error] Model '{active_name}' tidak ditemukan di registry['models']")
        sys.exit(1)

    return {
        "run_id":      active_name,
        "model_type":  active_name,
        "status":      model_data.get("status", "active"),
        "n_features":  model_data.get("n_features", 85),
        "version":     model_data.get("version", ""),
        "trained_at":  model_data.get("trained_date", ""),
    }


def patch_inference_config(config: dict) -> dict:
    """Patch inference_config agar kompatibel (rename labeling → fallback_tp_sl)."""
    patched = dict(config)

    if "labeling" in patched and "fallback_tp_sl" not in patched:
        patched["fallback_tp_sl"] = patched.pop("labeling")
        print("  [patch] 'labeling' -> 'fallback_tp_sl'")

    # Pastikan coins_validated.recommended ada
    cv = patched.get("coins_validated", {})
    if isinstance(cv, dict) and "recommended" not in cv:
        high = cv.get("high_priority", [])
        med  = cv.get("medium_priority", [])
        cv["recommended"] = high + med
        patched["coins_validated"] = cv
        print(f"  [patch] coins_validated.recommended ditambah ({len(high + med)} koin)")

    # Normalisasi field drawdown: rename lev3x → lev5x jika pipeline menggunakan format lama
    bpc = patched.get("backtest_per_coin", {})
    changed_bpc = False
    for data in bpc.values():
        if "dd_lev3x" in data and "dd_lev5x" not in data:
            data["dd_lev5x"] = data.pop("dd_lev3x")
            changed_bpc = True
    if changed_bpc:
        print("  [patch] backtest_per_coin: 'dd_lev3x' -> 'dd_lev5x'")

    bs = patched.get("backtest_summary", {})
    if "mean_drawdown_lev3x" in bs and "mean_drawdown_lev5x" not in bs:
        bs["mean_drawdown_lev5x"] = bs.pop("mean_drawdown_lev3x")
        patched["backtest_summary"] = bs
        print("  [patch] backtest_summary: 'mean_drawdown_lev3x' -> 'mean_drawdown_lev5x'")

    return patched


def write_registry(active_info: dict, dest_dir: Path, version_str: str):
    """Tulis model_registry.json dengan format versions.

    Semua path menunjuk langsung ke root models/ (tanpa subfolder versi).

    Args:
        active_info: Informasi model aktif dari registry.
        dest_dir: Direktori tujuan (models/).
        version_str: Timestamp string untuk run_id.
    """
    paths = {
        "lstm":             "models/lstm_best.pt",
        "scaler":           "models/lstm_scaler.pkl",
        "meta":             "models/ensemble_meta.pkl",
        "calibrator":       "models/calibrator.pkl",
        "inference_config": "models/inference_config.json",
    }

    registry = {
        "versions": [
            {
                "run_id":      version_str,
                "model_type":  active_info.get("model_type", "lstm"),
                "status":      "active",
                "n_features":  active_info.get("n_features", 85),
                "version":     active_info.get("version", ""),
                "trained_at":  active_info.get("trained_at", ""),
                "paths":       paths,
            }
        ]
    }

    dest_path = dest_dir / "model_registry.json"
    with open(dest_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  [write] {dest_path}")
    return registry


def deploy(source_dir: Path, dry_run: bool) -> Path:
    """Deploy model dari source_dir ke models/ root.

    Semua file langsung ditimpa di models/, tanpa folder backup.
    model_registry.json ditulis dengan path langsung ke root models/.

    Args:
        source_dir: Direktori sumber file model.
        dry_run: Jika True, hanya preview.

    Returns:
        MODELS_DIR.
    """
    source_dir = source_dir.resolve()
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  DEPLOY MODEL — Seamless (tanpa backup)")
    print(f"{'='*60}")
    print(f"  Source  : {source_dir}")
    print(f"  Dry run : {'YES' if dry_run else 'NO'}")
    print(f"{'='*60}\n")

    # ── 1. Validasi file sumber ──────────────────────────────────────────────
    missing = [f for f in REQUIRED_FILES if not (source_dir / f).exists()]
    if missing:
        print(f"  [error] File tidak ditemukan di {source_dir}:")
        for f in missing:
            print(f"         - {f}")
        sys.exit(1)
    print(f"  [ok] Semua {len(REQUIRED_FILES)} file model ditemukan")

    # ── 2. Baca registry + aktifkan model ────────────────────────────────────
    src_registry = source_dir / "model_registry.json"
    if src_registry.exists():
        data = load_new_registry(src_registry)
        active_info = get_active_model_info(data)
        print(f"  [registry] Model aktif: {active_info.get('run_id')}")
        print(f"             Status    : {active_info.get('status')}")
        print(f"             Trained at: {active_info.get('trained_at')}")
    else:
        print(f"  [info] Tidak ada model_registry.json di source — menggunakan default")
        active_info = {
            "run_id":      timestamp,
            "model_type":  "lstm",
            "status":      "active",
            "n_features":  85,
            "version":     "",
            "trained_at":  datetime.now(timezone.utc).isoformat(),
        }

    if dry_run:
        print(f"\n  [dry-run] Akan menulis:")
        print(f"    - {MODELS_DIR / 'model_registry.json'} (format versions)")
        print(f"    - 7 file model akan ditimpa di {MODELS_DIR}/")
        return MODELS_DIR

    # ── 3. Patch inference_config ────────────────────────────────────────────
    with open(source_dir / "inference_config.json") as f:
        config = json.load(f)
    patched = patch_inference_config(config)

    # ── 4. Copy semua file kecuali inference_config ──────────────────────────
    print(f"\n  Menyalin file ke {MODELS_DIR} ...")
    for fname in REQUIRED_FILES:
        src = source_dir / fname
        dst = MODELS_DIR / fname
        if fname == "inference_config.json":
            continue  # handle terpisah dengan patch
        shutil.copy2(src, dst)
        print(f"    [copy] {fname}")

    # ── 5. Simpan inference_config.json yang sudah di-patch ──────────────────
    with open(MODELS_DIR / "inference_config.json", "w") as f:
        json.dump(patched, f, indent=2)
    print(f"    [save] models/inference_config.json (patched)")

    # ── 6. Tulis model_registry.json ─────────────────────────────────────────
    registry = write_registry(active_info, MODELS_DIR, timestamp)

    print(f"\n{'='*60}")
    print(f"  DEPLOY BERHASIL")
    print(f"{'='*60}")
    print(f"  Registry  : {MODELS_DIR / 'model_registry.json'}")
    print(f"  Model     : {MODELS_DIR}/ (langsung, tanpa backup)")
    print(f"\n  Jalankan deploy/seed_db.py untuk update database:")
    print(f"    python deploy/seed_db.py")
    print(f"{'='*60}")

    return MODELS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Deploy model training output — seamless, tanpa backup"
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Folder source (default: root models/)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview tanpa menulis file",
    )
    args = parser.parse_args()

    # Tentukan source directory
    if args.source:
        source_dir = Path(args.source)
    else:
        source_dir = MODELS_DIR

    if not source_dir.exists():
        print(f"[error] Source directory tidak ditemukan: {source_dir}")
        sys.exit(1)

    deploy(source_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
