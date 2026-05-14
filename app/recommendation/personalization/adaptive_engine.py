"""Adaptive recommendation ranking."""

from __future__ import annotations

from app.recommendation.personalization.preference_learner import preference_boost
from app.recommendation.personalization.user_profile import build_user_profile


def rank_recommendations(
    recommendations: list[dict[str, str]],
    mood: str = "",
    genre: str = "",
) -> list[dict[str, str]]:
    """Rank recommendations with lightweight personalization."""
    profile = build_user_profile()
    ranked = []
    for index, song in enumerate(recommendations):
        score = 1.0 - (index * 0.01) + preference_boost(song, profile, mood=mood, genre=genre)
        enriched = dict(song)
        enriched["personalized_score"] = round(score, 3)
        ranked.append(enriched)
    return sorted(ranked, key=lambda item: item["personalized_score"], reverse=True)
