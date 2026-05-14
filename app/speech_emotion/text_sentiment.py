"""Text sentiment analysis for typed and transcribed prompts."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def _load_sentiment_pipeline():
    """Load a multilingual sentiment model through HuggingFace transformers."""
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
    )


def _map_label_to_sentiment(label: str) -> str:
    label = label.lower()
    if "1" in label or "2" in label or "negative" in label:
        return "negative"
    if "4" in label or "5" in label or "positive" in label:
        return "positive"
    return "neutral"


def analyze_text_sentiment(text: str) -> Dict[str, object]:
    """Analyze transcribed text and return positive/negative/neutral sentiment."""
    if not text or not text.strip():
        return {"sentiment": "neutral", "confidence": 0.0, "raw_label": "empty"}

    try:
        result = _load_sentiment_pipeline()(text[:512])[0]
    except Exception as exc:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "raw_label": f"unavailable: {exc}",
        }

    return {
        "sentiment": _map_label_to_sentiment(str(result["label"])),
        "confidence": float(result["score"]),
        "raw_label": result["label"],
    }

