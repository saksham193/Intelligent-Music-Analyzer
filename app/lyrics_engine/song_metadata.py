"""Song metadata extraction for lyrics queries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


def _split_filename(stem: str) -> tuple[str, str]:
    """Infer artist/title from common filenames like Artist - Title.mp3."""
    cleaned = re.sub(r"[_]+", " ", stem).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if " - " in cleaned:
        artist, title = cleaned.split(" - ", 1)
        return title.strip(), artist.strip()
    return cleaned.strip(), ""


def extract_song_metadata(audio_path: str | Path) -> Dict[str, str]:
    """Extract title and artist from tags, falling back to the filename."""
    path = Path(audio_path)
    title = ""
    artist = ""

    try:
        from mutagen import File

        tags = File(path, easy=True)
        if tags:
            title = (tags.get("title") or [""])[0]
            artist = (tags.get("artist") or [""])[0]
    except Exception:
        title = ""
        artist = ""

    if not title:
        title, inferred_artist = _split_filename(path.stem)
        artist = artist or inferred_artist

    return {
        "title": title.strip(),
        "artist": artist.strip(),
        "query": f"{title} {artist}".strip(),
    }

