"""Central logging helpers for the application."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger.

    Streamlit apps can rerun modules often, so this avoids adding duplicate
    handlers every time a file is imported.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

