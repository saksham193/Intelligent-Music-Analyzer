"""Lyrics cleanup helpers."""

from __future__ import annotations

import re


def clean_lyrics(raw_lyrics: str) -> str:
    """Remove common Genius annotations and noisy metadata."""
    text = raw_lyrics or ""
    text = re.sub(r"^\d+\s*Contributors?.*?Lyrics", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\d*Embed$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preview_lyrics(lyrics: str, max_chars: int = 600) -> str:
    """Return a short preview for compact UI areas."""
    if len(lyrics) <= max_chars:
        return lyrics
    return f"{lyrics[:max_chars].rstrip()}..."

