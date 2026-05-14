"""Preference learning helpers."""

from __future__ import annotations


def preference_boost(song: dict[str, str], profile: dict[str, object], mood: str = "", genre: str = "") -> float:
    """Return a simple ranking boost based on profile matches."""
    boost = 0.0
    favorite_moods = {item[0] for item in profile.get("favorite_moods", [])}
    favorite_genres = {item[0] for item in profile.get("favorite_genres", [])}
    if mood in favorite_moods:
        boost += 0.15
    if genre in favorite_genres:
        boost += 0.1
    if song.get("title"):
        boost += 0.02
    return boost

