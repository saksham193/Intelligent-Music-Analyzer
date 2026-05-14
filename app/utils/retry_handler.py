"""Small retry utility for API calls and optional integrations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 1.6,
) -> T:
    """Run a callable with simple exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds * (backoff**attempt))
    raise RuntimeError(f"Operation failed after {attempts} attempts") from last_error

