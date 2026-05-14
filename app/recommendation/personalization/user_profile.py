"""User profile summaries from local feedback."""

from __future__ import annotations

from collections import Counter

from app.recommendation.personalization.recommendation_memory import fetch_feedback


def build_user_profile() -> dict[str, object]:
    """Build a lightweight profile from liked/skipped songs."""
    feedback = fetch_feedback()
    liked = [item for item in feedback if item.get("action") == "liked"]
    skipped = [item for item in feedback if item.get("action") == "skipped"]

    favorite_moods = Counter(item.get("mood") for item in liked if item.get("mood"))
    favorite_genres = Counter(item.get("genre") for item in liked if item.get("genre"))
    preferred_languages = Counter(item.get("language") for item in liked if item.get("language"))

    return {
        "liked_count": len(liked),
        "skipped_count": len(skipped),
        "favorite_moods": favorite_moods.most_common(5),
        "favorite_genres": favorite_genres.most_common(5),
        "preferred_languages": preferred_languages.most_common(3),
    }

