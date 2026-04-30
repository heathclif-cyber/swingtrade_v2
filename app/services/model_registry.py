"""
app/services/model_registry.py — Baca model_registry.json dan resolusi path model.
"""

import json
from pathlib import Path
from typing import Optional

ROOT        = Path(__file__).parent.parent.parent
REGISTRY    = ROOT / "models" / "model_registry.json"


def load_registry() -> list[dict]:
    """Return list of version entries dari model_registry.json."""
    with open(REGISTRY) as f:
        data = json.load(f)
    return data.get("versions", [])


def get_active_version() -> Optional[dict]:
    """Return versi pertama dengan status 'available' atau 'active'."""
    versions = load_registry()
    for v in versions:
        if v.get("status") in ("available", "active"):
            return v
    return None


def get_version_by_run_id(run_id: str) -> Optional[dict]:
    for v in load_registry():
        if v.get("run_id") == run_id:
            return v
    return None


def resolve_path(relative: str) -> Path:
    """Konversi path relatif dari registry ke path absolut."""
    return ROOT / relative


def load_inference_config(run_id: str) -> dict:
    version = get_version_by_run_id(run_id)
    if not version:
        raise ValueError(f"run_id {run_id} tidak ditemukan di registry")
    config_path = resolve_path(version["paths"]["inference_config"])
    with open(config_path) as f:
        return json.load(f)
