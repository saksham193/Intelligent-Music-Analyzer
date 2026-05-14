"""Language routing for English, Hindi, and Hinglish lyric transcription."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


HINGLISH_HINTS = {
    "aaj",
    "dil",
    "hai",
    "ho",
    "hun",
    "ishq",
    "jaan",
    "kabhi",
    "mera",
    "meri",
    "pyaar",
    "pyar",
    "sajna",
    "sanam",
    "tera",
    "tere",
    "teri",
    "tum",
    "yaar",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower(), flags=re.UNICODE)


def detect_text_language(text: str) -> Dict[str, object]:
    """Detect likely language from text without relying on one external model."""
    tokens = _tokens(text)
    if not tokens:
        return {"language": "unknown", "is_mixed": False, "mixed_probability": 0.0, "confidence": 0.0}

    devanagari = sum(1 for token in tokens if re.search(r"[\u0900-\u097F]", token))
    ascii_tokens = sum(1 for token in tokens if re.search(r"[a-zA-Z]", token))
    hinglish_hits = sum(1 for token in tokens if token in HINGLISH_HINTS)

    devanagari_ratio = devanagari / len(tokens)
    english_ratio = ascii_tokens / len(tokens)
    hinglish_ratio = hinglish_hits / max(ascii_tokens, 1)
    mixed_probability = max(
        min(devanagari_ratio, english_ratio) * 2,
        min(hinglish_ratio, 1.0),
    )

    if devanagari_ratio > 0.35 and english_ratio > 0.2:
        language = "mixed"
    elif devanagari_ratio > 0.35:
        language = "hi"
    elif hinglish_ratio > 0.12:
        language = "hinglish"
    else:
        language = "en"

    confidence = max(devanagari_ratio, english_ratio, hinglish_ratio)
    return {
        "language": language,
        "is_mixed": language in {"mixed", "hinglish"} or mixed_probability > 0.35,
        "mixed_probability": round(min(mixed_probability, 1.0), 2),
        "confidence": round(min(confidence, 1.0), 2),
    }


def route_whisper_language(audio_path: str | Path, hint_text: str = "") -> Dict[str, object]:
    """Choose Whisper language settings from metadata/filename/text hints."""
    path = Path(audio_path)
    combined_hint = f"{path.stem} {hint_text}".replace("_", " ").replace("-", " ")
    detection = detect_text_language(combined_hint)
    language = str(detection["language"])

    whisper_language: Optional[str]
    if language == "hi":
        whisper_language = "hi"
    elif language == "en":
        whisper_language = "en"
    else:
        whisper_language = None

    return {
        "whisper_language": whisper_language,
        "detected_language": language,
        "is_mixed": bool(detection["is_mixed"]),
        "mixed_probability": float(detection["mixed_probability"]),
        "confidence": float(detection["confidence"]),
        "reason": "filename/text hint" if combined_hint.strip() else "auto multilingual fallback",
    }


def display_language(language: str, is_mixed: bool = False) -> str:
    """Human-friendly language label for the UI."""
    if is_mixed or language in {"mixed", "hinglish"}:
        return "Hindi-English / Hinglish"
    if language == "hi":
        return "Hindi"
    if language == "en":
        return "English"
    return language.title() if language else "Unknown"
