"""Translation helpers for lyrics."""

from __future__ import annotations


LANGUAGE_NAMES = {
    "en": "english",
    "hi": "hindi",
}


def translate_text(text: str, target_language: str = "en", source_language: str = "auto") -> dict[str, str]:
    """Translate lyrics with deep-translator when available."""
    if not text or not text.strip():
        return {"text": "", "source": source_language, "target": target_language, "error": ""}

    try:
        from deep_translator import GoogleTranslator

        target = LANGUAGE_NAMES.get(target_language, target_language)
        source = LANGUAGE_NAMES.get(source_language, source_language)
        translated = GoogleTranslator(source=source, target=target).translate(text[:4500])
        return {"text": translated, "source": source_language, "target": target_language, "error": ""}
    except Exception as exc:
        return {"text": "", "source": source_language, "target": target_language, "error": str(exc)}

