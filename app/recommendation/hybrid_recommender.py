"""Hybrid recommendation intelligence and explainable emotion fusion."""

from __future__ import annotations

from app.config.settings import HYBRID_FUSION_WEIGHTS


EMOTION_SCORES = {
    "happy": 0.85,
    "positive": 0.8,
    "energetic upbeat": 0.8,
    "calm": 0.58,
    "relaxed ambient": 0.55,
    "neutral": 0.5,
    "balanced expressive": 0.5,
    "sad": 0.2,
    "negative": 0.18,
    "emotional melancholic": 0.25,
    "angry": 0.12,
    "dark aggressive": 0.1,
    "anxious": 0.18,
}


def _score(label: str | None, fallback: float = 0.5) -> float:
    if not label:
        return fallback
    return EMOTION_SCORES.get(str(label).lower(), fallback)


def fuse_multimodal_emotion(
    speech_emotion: str | None = None,
    speech_sentiment: str | None = None,
    lyrics_sentiment: str | None = None,
    music_mood: str | None = None,
    genre: str | None = None,
    energy_score: float = 0.5,
    valence_score: float = 0.5,
    weights: dict[str, float] = HYBRID_FUSION_WEIGHTS,
) -> dict[str, object]:
    """Fuse separated pipeline outputs into one explainable interpretation."""
    channel_scores = {
        "speech_emotion": _score(speech_emotion),
        "speech_sentiment": _score(speech_sentiment),
        "lyrics_sentiment": _score(lyrics_sentiment),
        "music_mood": _score(music_mood, fallback=valence_score),
    }
    total_weight = sum(weights.values()) or 1.0
    weighted_score = sum(channel_scores[key] * weights.get(key, 0.0) for key in channel_scores) / total_weight

    spread = max(channel_scores.values()) - min(channel_scores.values())
    dissonance_score = float(min(max(spread, 0.0), 1.0))
    confidence = float(max(0.0, 1.0 - (dissonance_score * 0.55)))

    if dissonance_score > 0.55:
        interpretation = "Emotionally contradictory reflective state"
    elif weighted_score >= 0.68 and energy_score >= 0.55:
        interpretation = "Positive high-energy state"
    elif weighted_score >= 0.58:
        interpretation = "Positive balanced state"
    elif weighted_score <= 0.32 and energy_score >= 0.55:
        interpretation = "Intense negative state"
    elif weighted_score <= 0.4:
        interpretation = "Melancholic reflective state"
    else:
        interpretation = "Neutral mixed emotional state"

    return {
        "final_interpretation": interpretation,
        "emotional_complexity": "high" if dissonance_score > 0.55 else "moderate" if dissonance_score > 0.3 else "low",
        "dissonance_score": dissonance_score,
        "confidence": confidence,
        "weighted_score": float(weighted_score),
        "channel_scores": channel_scores,
        "genre_context": genre or "unknown",
        "explanation": (
            "Weighted fusion combines speech emotion, speech sentiment, lyrics sentiment, "
            "and music mood while measuring contradiction between channels."
        ),
    }
