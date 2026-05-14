"""Readable lyrics cleanup for English, Hindi, and Hinglish transcripts."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


NOISE_LINES = {
    "music",
    "song",
    "lyrics",
    "instrumental",
    "applause",
    "noise",
    "silence",
    "humming",
    "background music",
}

NOISE_TOKENS = {
    "uh",
    "um",
    "hmm",
    "mmm",
    "ah",
    "ohh",
    "yeah yeah",
    "la la la la",
}


def normalize_hinglish_text(text: str) -> str:
    """Normalize common Roman Hindi spellings without changing meaning."""
    replacements = {
        r"\bhain\b": "hai",
        r"\bhoon\b": "hun",
        r"\bhu\b": "hun",
        r"\bmera\b": "mera",
        r"\bmeraa\b": "mera",
        r"\btere\b": "tere",
        r"\bteraa\b": "tera",
        r"\btumhi\b": "tum hi",
        r"\bdilbar\b": "dilbar",
    }
    cleaned = text
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _dedupe_repeated_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    for line in lines:
        if not line:
            continue
        if deduped and (_similar(line, deduped[-1]) > 0.88 or line == deduped[-1]):
            continue
        deduped.append(line)
    return deduped


def clean_extracted_lyrics(raw_text: str) -> str:
    """Clean timestamps, noise markers, repeated fragments, and random symbols."""
    text = raw_text or ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\]", " ", text)
    text = re.sub(r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?", " ", text)
    text = re.sub(r"\[(verse|chorus|hook|intro|outro|bridge).*?\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F.,!?'\n-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1 \1", text, flags=re.IGNORECASE)
    text = normalize_hinglish_text(text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip(" -_.,")
        lowered = line.lower()
        if not line or lowered in NOISE_LINES or lowered in NOISE_TOKENS:
            continue
        line_words = re.findall(r"[\w\u0900-\u097F]+", lowered, flags=re.UNICODE)
        if line_words and all(word in NOISE_TOKENS for word in line_words):
            continue
        if len(line) <= 2:
            continue
        cleaned_lines.append(line)

    cleaned_lines = _dedupe_repeated_lines(cleaned_lines)
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def lyric_snippets(lyrics: str, max_lines: int = 3) -> list[str]:
    """Return short searchable transcript fragments for hybrid Genius lookup."""
    snippets: list[str] = []
    for line in clean_extracted_lyrics(lyrics).splitlines():
        words = line.split()
        if len(words) < 3:
            continue
        snippets.append(" ".join(words[:10]))
        if len(snippets) >= max_lines:
            break
    return snippets


def transcript_quality_score(text: str, segments: list[dict[str, object]] | None = None) -> float:
    """Estimate transcript usefulness from length, segment confidence, and noise."""
    cleaned = clean_extracted_lyrics(text)
    words = re.findall(r"[\w\u0900-\u097F]+", cleaned, flags=re.UNICODE)
    if not words:
        return 0.0

    length_score = min(len(words) / 80, 1.0)
    unique_score = min(len(set(word.lower() for word in words)) / max(len(words), 1) * 1.8, 1.0)
    noise_count = sum(1 for word in words if word.lower() in NOISE_TOKENS)
    noise_penalty = min(noise_count / max(len(words), 1), 0.4)

    confidence_score = 0.7
    if segments:
        no_speech_probs = [
            float(segment.get("no_speech_prob", 0.0))
            for segment in segments
            if isinstance(segment, dict) and "no_speech_prob" in segment
        ]
        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
        confidence_score = max(0.0, 1.0 - avg_no_speech)

    score = (0.36 * length_score) + (0.34 * unique_score) + (0.30 * confidence_score) - noise_penalty
    return round(max(0.0, min(score, 1.0)), 2)
