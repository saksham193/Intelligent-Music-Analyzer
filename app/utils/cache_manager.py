"""File cache helpers for lyrics, uploads, and temporary artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def safe_cache_key(*parts: str) -> str:
    """Create a filesystem-safe cache key from user/API text."""
    raw = "|".join(part.strip().lower() for part in parts if part)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_text_cache(path: str | Path) -> str | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    return cache_path.read_text(encoding="utf-8")


def write_text_cache(path: str | Path, text: str) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return cache_path

