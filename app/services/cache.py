"""
app/services/cache.py — Thread-safe LRU cache dengan TTL.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock  = threading.Lock()
        self.max_size    = max_size
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            value, ts = self._store[key]
            if time.time() - ts > self.ttl_seconds:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [k for k, (_, ts) in self._store.items()
                       if now - ts > self.ttl_seconds]
            for k in expired:
                del self._store[k]
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


import os

model_cache   = TTLCache(max_size=2,  ttl_seconds=int(os.getenv("MODEL_CACHE_TTL_SECONDS", 1800)))
feature_cache = TTLCache(max_size=5,  ttl_seconds=300)
signal_cache  = TTLCache(max_size=1,  ttl_seconds=60)
