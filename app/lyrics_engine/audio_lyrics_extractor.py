"""Audio-first lyric extraction for mobile clips, songs, and voice notes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
from scipy.io import wavfile

from app.config.settings import DEFAULT_SAMPLE_RATE, LYRICS_CACHE_DIR
from app.lyrics_engine.language_router import (
    detect_text_language,
    display_language,
    route_whisper_language,
)
from app.lyrics_engine.lyrics_cleaner import (
    clean_extracted_lyrics,
    transcript_quality_score,
)
from app.music_intelligence.stem_separator import separate_stems
from app.speech_emotion.transcriber import transcribe_audio


LYRICS_AUDIO_CACHE_DIR = LYRICS_CACHE_DIR / "audio"


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _json_cache_path(audio_path: Path) -> Path:
    return LYRICS_AUDIO_CACHE_DIR / f"{_file_digest(audio_path)}.json"


def _preprocessed_path(audio_path: Path) -> Path:
    return (
        LYRICS_AUDIO_CACHE_DIR
        / f"{_file_digest(audio_path)}_preprocessed.wav"
    )


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _looks_like_real_song(text: str) -> bool:
    """
    Reject podcasts/interviews/comedy transcripts.
    """
    if not text:
        return False

    text = text.lower()

    banned_patterns = [
        "ladies and gentlemen",
        "thank you for coming",
        "air date",
        "applause",
        "stand-up",
        "audience",
        "comedy",
        "interview",
    ]

    for pattern in banned_patterns:
        if pattern in text:
            return False

    lines = text.splitlines()

    repeated_short_lines = 0

    for line in lines:
        words = line.strip().split()

        if 2 <= len(words) <= 8:
            repeated_short_lines += 1

    return repeated_short_lines >= 3


def preprocess_audio_for_lyrics(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:

    source = Path(audio_path)
    output = _preprocessed_path(source)

    if output.exists():
        return output

    LYRICS_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from pydub import AudioSegment, effects

        segment = AudioSegment.from_file(source)
        segment = effects.normalize(segment)
        segment = segment.set_channels(1)
        segment = segment.set_frame_rate(sample_rate)

        pydub_wav = output.with_name(f"{output.stem}_pydub.wav")

        segment.export(pydub_wav, format="wav")

        load_source = pydub_wav

    except Exception:
        load_source = source

    y, _ = librosa.load(
        str(load_source),
        sr=sample_rate,
        mono=True,
    )

    if y.size == 0:
        raise ValueError("Audio file is empty.")

    y, _ = librosa.effects.trim(y, top_db=28)

    if y.size == 0:
        raise ValueError("Audio contains only silence.")

    y = librosa.util.normalize(y)
    y = np.clip(y * 0.95, -1.0, 1.0)

    wavfile.write(
        output,
        sample_rate,
        (y * 32767).astype(np.int16),
    )

    return output


def _try_vocal_path(audio_path: Path) -> tuple[Path, str]:
    try:
        import demucs  # noqa: F401
    except Exception:
        return audio_path, "original_audio"

    stems = separate_stems(audio_path)

    if stems.get("ok") and stems.get("stems", {}).get("vocals"):
        return (
            Path(str(stems["stems"]["vocals"])),
            "separated_vocals",
        )

    return audio_path, "original_audio"


def _transcribe_with_optional_faster_whisper(
    audio_path: Path,
    language: Optional[str],
) -> Dict[str, object]:

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8",
        )

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
        )

        segment_rows = []
        text_parts = []

        for segment in segments:
            text_parts.append(segment.text.strip())

            segment_rows.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "no_speech_prob": getattr(
                        segment,
                        "no_speech_prob",
                        0.0,
                    ),
                }
            )

        return {
            "text": " ".join(
                part
                for part in text_parts
                if part
            ).strip(),
            "language": getattr(
                info,
                "language",
                language or "unknown",
            ),
            "segments": segment_rows,
            "error": "",
            "engine": "faster-whisper",
        }

    except Exception:
        result = transcribe_audio(
            audio_path,
            language=language,
        )

        result["engine"] = "openai-whisper"

        return result


def extract_lyrics_from_audio(
    audio_path: str | Path,
    hint_text: str = "",
    use_vocals: bool = True,
) -> Dict[str, object]:

    source = Path(audio_path)

    cache_path = _json_cache_path(source)

    cached = _read_json(cache_path)

    if cached:
        cached["cache_path"] = str(cache_path)
        return cached

    try:
        working_path = preprocess_audio_for_lyrics(source)

        transcription_path, vocal_source = (
            _try_vocal_path(working_path)
            if use_vocals
            else (working_path, "original_audio")
        )

        route = route_whisper_language(
            source,
            hint_text=hint_text,
        )

        transcript = _transcribe_with_optional_faster_whisper(
            transcription_path,
            language=(
                "hi"
                if str(
                    route.get("detected_language", "")
                ).lower() in ["hindi", "hinglish"]
                else route.get("whisper_language")
            ),
        )

        cleaned = clean_extracted_lyrics(
            str(transcript.get("text", ""))
        )

        if not _looks_like_real_song(cleaned):
            cleaned = ""

        text_language = detect_text_language(cleaned)

        detected_language = (
            str(text_language["language"])
            if text_language.get("confidence", 0) > 0.18
            else str(
                transcript.get("language")
                or route.get("detected_language")
                or "unknown"
            )
        )

        is_mixed = bool(
            text_language.get("is_mixed")
            or route.get("is_mixed")
        )

        quality = transcript_quality_score(
            cleaned,
            transcript.get("segments", []),
        )

        confidence = (
            round(min(0.92, (quality * 0.72) + 0.18), 2)
            if cleaned
            else 0.0
        )

        result = {
            "lyrics": cleaned,
            "raw_transcript": transcript.get("text", ""),
            "source": "audio_transcription",
            "audio_source": vocal_source,
            "confidence": confidence,
            "quality_score": quality,
            "language": display_language(
                detected_language,
                is_mixed=is_mixed,
            ),
            "language_code": detected_language,
            "is_mixed_language": is_mixed,
            "transcription_engine": transcript.get(
                "engine",
                "openai-whisper",
            ),
            "error": transcript.get("error", ""),
        }

    except Exception as exc:
        result = {
            "lyrics": "",
            "raw_transcript": "",
            "source": "audio_transcription",
            "audio_source": "none",
            "confidence": 0.0,
            "quality_score": 0.0,
            "language": "Unknown",
            "language_code": "unknown",
            "is_mixed_language": False,
            "transcription_engine": "none",
            "error": str(exc),
        }

    _write_json(cache_path, result)

    result["cache_path"] = str(cache_path)

    return result