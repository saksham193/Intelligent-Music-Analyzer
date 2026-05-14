"""Cleanup helpers for generated temporary files."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from app.config.settings import CACHE_DIR, CACHE_RETENTION_HOURS


def cleanup_cache(cache_dir: str | Path = CACHE_DIR, retention_hours: int = CACHE_RETENTION_HOURS) -> int:
    """Delete old cache files while keeping marker files and directories."""
    root = Path(cache_dir)
    if not root.exists():
        return 0

    cutoff = datetime.now().timestamp() - timedelta(hours=retention_hours).total_seconds()
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
    return removed
