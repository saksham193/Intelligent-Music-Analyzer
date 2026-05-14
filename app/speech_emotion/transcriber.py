"""Offline speech-to-text with local Whisper models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from app.config.settings import DEFAULT_WHISPER_MODEL


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    import whisper

    return whisper.load_model(model_name)


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
) -> Dict[str, object]:
    """Transcribe Hindi/English audio locally with Whisper.

    Set language to "hi" or "en" to force a language, or leave it as None for
    automatic language detection.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        model = _load_whisper_model(model_name)
        options = {"fp16": False}
        if language:
            options["language"] = language
        result = model.transcribe(str(path), **options)
    except Exception as exc:
        return {
            "text": "",
            "language": "unknown",
            "segments": [],
            "error": str(exc),
        }

    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
        "segments": result.get("segments", []),
        "error": "",
    }

