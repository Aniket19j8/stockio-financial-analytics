"""
cache_manager.py
================
Cache-aware pipeline for rapid re-analysis.
Supports TTL-based expiry and memory/disk caching.
"""

import hashlib
import json
import os
import time
import pickle
from typing import Any, Optional
import pandas as pd


class CacheManager:
    """
    In-memory + disk cache for stock data and computation results.
    Supports TTL-based expiry for time-sensitive financial data.
    """

    def __init__(self, cache_dir: str = ".stockio_cache", default_ttl: int = 900):
        """
        Parameters
        ----------
        cache_dir : str - Directory for disk cache
        default_ttl : int - Default time-to-live in seconds (15 min)
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self._memory_cache = {}
        self._access_log = {}

        os.makedirs(cache_dir, exist_ok=True)

    def _make_key(self, key: str) -> str:
        """Create a safe cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve from cache (memory first, then disk).
        Returns None if not found or expired.
        """
        ttl = ttl or self.default_ttl
        hashed = self._make_key(key)

        # Check memory cache
        if hashed in self._memory_cache:
            entry = self._memory_cache[hashed]
            if time.time() - entry["timestamp"] < ttl:
                self._access_log[hashed] = time.time()
                return entry["data"]
            else:
                del self._memory_cache[hashed]

        # Check disk cache
        disk_path = os.path.join(self.cache_dir, f"{hashed}.pkl")
        if os.path.exists(disk_path):
            try:
                mtime = os.path.getmtime(disk_path)
                if time.time() - mtime < ttl:
                    with open(disk_path, "rb") as f:
                        data = pickle.load(f)
                    # Promote to memory
                    self._memory_cache[hashed] = {
                        "data": data,
                        "timestamp": mtime,
                    }
                    self._access_log[hashed] = time.time()
                    return data
                else:
                    os.remove(disk_path)
            except Exception:
                pass

        return None

    def set(self, key: str, data: Any) -> None:
        """Store data in both memory and disk cache."""
        hashed = self._make_key(key)
        now = time.time()

        # Memory cache
        self._memory_cache[hashed] = {
            "data": data,
            "timestamp": now,
        }

        # Disk cache
        try:
            disk_path = os.path.join(self.cache_dir, f"{hashed}.pkl")
            with open(disk_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Disk write failure is non-fatal

        self._access_log[hashed] = now

        # Evict old entries if memory cache is too large
        if len(self._memory_cache) > 100:
            self._evict()

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        hashed = self._make_key(key)
        self._memory_cache.pop(hashed, None)

        disk_path = os.path.join(self.cache_dir, f"{hashed}.pkl")
        if os.path.exists(disk_path):
            os.remove(disk_path)

    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._access_log.clear()

        for f in os.listdir(self.cache_dir):
            if f.endswith(".pkl"):
                os.remove(os.path.join(self.cache_dir, f))

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": len([f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]),
            "total_accesses": len(self._access_log),
        }

    def _evict(self) -> None:
        """Evict least recently accessed entries from memory cache."""
        if not self._access_log:
            return

        # Sort by access time, remove oldest 20%
        sorted_keys = sorted(self._access_log.items(), key=lambda x: x[1])
        n_evict = max(1, len(sorted_keys) // 5)

        for key, _ in sorted_keys[:n_evict]:
            self._memory_cache.pop(key, None)
            self._access_log.pop(key, None)
