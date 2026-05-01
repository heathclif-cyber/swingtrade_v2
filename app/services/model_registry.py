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
    """Return versi terbaru (berdasarkan trained_at) dengan status 'available' atau 'active'."""
    versions = [v for v in load_registry() if v.get("status") in ("available", "active")]
    if not versions:
        return None
    return max(versions, key=lambda v: v.get("trained_at", ""), default=None)


def resolve_path(relative: str) -> Path:
    """Konversi path relatif dari registry ke path absolut."""
    return ROOT / relative


def load_inference_config() -> dict:
    """Load inference config dari versi terbaru yang aktif."""
    version = get_active_version()
    if not version:
        raise ValueError("Tidak ada versi model aktif di registry")
    config_path = resolve_path(version["paths"]["inference_config"])
    with open(config_path) as f:
        return json.load(f)
