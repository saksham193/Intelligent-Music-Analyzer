"""Language detection for lyrics text."""

from __future__ import annotations


def detect_language(text: str) -> dict[str, object]:
    """Detect lyrics language, returning unknown when detection is unavailable."""
    if not text or not text.strip():
        return {"language": "unknown", "confidence": 0.0}

    try:
        from langdetect import detect_langs

        candidates = detect_langs(text[:2000])
        if not candidates:
            return {"language": "unknown", "confidence": 0.0}
        best = candidates[0]
        return {"language": best.lang, "confidence": float(best.prob)}
    except Exception as exc:
        return {"language": "unknown", "confidence": 0.0, "error": str(exc)}

