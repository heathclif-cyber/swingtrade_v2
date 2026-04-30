"""
app/services/memory.py — Monitor RSS dan trigger cleanup jika mendekati batas.
"""

import gc
import logging
import os

import psutil

logger = logging.getLogger(__name__)

LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", 665))


def get_memory_status() -> dict:
    rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return {
        "rss_mb":   round(rss_mb, 1),
        "limit_mb": LIMIT_MB,
        "pct":      round(rss_mb / LIMIT_MB * 100, 1),
    }


def check_and_free(force: bool = False) -> bool:
    """Return True jika cleanup dilakukan."""
    status = get_memory_status()
    if not force and status["pct"] < 80:
        return False

    logger.warning(f"[memory] RSS={status['rss_mb']} MB ({status['pct']}%) — clearing caches")

    from app.services.cache import model_cache, feature_cache
    model_cache.clear()
    feature_cache.clear()
    gc.collect()

    after = get_memory_status()
    logger.info(f"[memory] After cleanup: RSS={after['rss_mb']} MB ({after['pct']}%)")
    return True
