"""Fusion helpers for speech emotion and text sentiment results."""

from __future__ import annotations


def fuse_emotions(voice_emotion: str, text_sentiment: str) -> str:
    """Combine voice emotion and text sentiment into a recommendation mood."""
    voice = (voice_emotion or "neutral").lower()
    sentiment = (text_sentiment or "neutral").lower()

    fusion_rules = {
        ("calm", "negative"): "emotionally tired",
        ("sad", "negative"): "sad",
        ("angry", "negative"): "angry",
        ("fearful", "negative"): "anxious",
        ("happy", "positive"): "happy",
        ("calm", "positive"): "calm",
        ("neutral", "positive"): "happy",
        ("neutral", "negative"): "sad",
    }
    return fusion_rules.get((voice, sentiment), voice)

