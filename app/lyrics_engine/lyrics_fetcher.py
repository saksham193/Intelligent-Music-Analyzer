"""Hybrid lyrics fetching with Genius, audio transcription, and local caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.config.settings import GENIUS_ACCESS_TOKEN, LYRICS_CACHE_DIR
from app.lyrics_engine.audio_lyrics_extractor import extract_lyrics_from_audio
from app.lyrics_engine.lyrics_cleaner import clean_extracted_lyrics, lyric_snippets
from app.lyrics_engine.parser import clean_lyrics as clean_genius_lyrics
from app.lyrics_engine.song_metadata import extract_song_metadata
from app.utils.cache_manager import read_text_cache, safe_cache_key, write_text_cache
from app.utils.retry_handler import retry_call


def _lyrics_cache_path(title: str, artist: str) -> Path:
    return LYRICS_CACHE_DIR / f"{safe_cache_key(title, artist)}.txt"


def _score_match(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0

    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(query.lower(), candidate.lower())) / 100
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, query.lower(), candidate.lower()).ratio()


def _source_label(source: str) -> str:
    labels = {
        "genius": "Metadata",
        "cache": "Metadata",
        "filename": "Filename",
        "hybrid_match": "Hybrid Match",
        "audio_transcription": "Audio Transcription",
        "raw_transcript": "Raw Transcript",
        "none": "None",
    }
    return labels.get(source, source.replace("_", " ").title())


def _language_matches(audio_lang: str, lyrics: str) -> bool:
    """
    Prevent random English Genius matches for Hindi audio.
    """
    if not lyrics:
        return False

    audio_lang = (audio_lang or "").lower()
    lyrics_lower = lyrics.lower()

    hindi_markers = [
        "dil",
        "ishq",
        "pyaar",
        "tera",
        "meri",
        "mohabbat",
        "tum",
        "hai",
        "nahi",
        "kyun",
    ]

    english_words = [
        "the",
        "and",
        "is",
        "are",
        "you",
        "what",
        "when",
        "where",
        "because",
    ]

    hindi_hits = sum(word in lyrics_lower for word in hindi_markers)
    english_hits = sum(word in lyrics_lower for word in english_words)

    if "hindi" in audio_lang or "hinglish" in audio_lang:
        return hindi_hits >= english_hits

    return True


def fetch_lyrics_by_metadata(
    title: str,
    artist: str = "",
    api_token: str = GENIUS_ACCESS_TOKEN,
) -> Dict[str, object]:
    title = title.strip()
    artist = artist.strip()

    if not title:
        return {
            "lyrics": "",
            "source": "none",
            "source_label": "None",
            "confidence": 0.0,
            "error": "Song title is required.",
        }

    cache_path = _lyrics_cache_path(title, artist)
    cached = read_text_cache(cache_path)

    if cached:
        return {
            "lyrics": cached,
            "source": "cache",
            "source_label": _source_label("cache"),
            "confidence": 0.88,
            "error": "",
            "cache_path": str(cache_path),
        }

    if not api_token:
        return {
            "lyrics": "",
            "source": "none",
            "source_label": "None",
            "confidence": 0.0,
            "error": "GENIUS_ACCESS_TOKEN is not configured.",
            "cache_path": str(cache_path),
        }

    try:
        import lyricsgenius

        genius = lyricsgenius.Genius(
            api_token,
            timeout=8,
            retries=1,
            remove_section_headers=True,
        )

        def _search():
            return genius.search_song(title=title, artist=artist or None)

        song = retry_call(_search, attempts=2, delay_seconds=1.0)

        if not song or not song.lyrics:
            return {
                "lyrics": "",
                "source": "genius",
                "source_label": _source_label("genius"),
                "confidence": 0.0,
                "error": "No lyrics found.",
                "cache_path": str(cache_path),
            }

        lyrics = clean_genius_lyrics(song.lyrics)

        write_text_cache(cache_path, lyrics)

        return {
            "lyrics": lyrics,
            "source": "genius",
            "source_label": _source_label("genius"),
            "confidence": 0.9,
            "error": "",
            "cache_path": str(cache_path),
        }

    except Exception as exc:
        return {
            "lyrics": "",
            "source": "genius",
            "source_label": _source_label("genius"),
            "confidence": 0.0,
            "error": str(exc),
            "cache_path": str(cache_path),
        }


def _fetch_lyrics_by_transcript_snippets(
    transcript: str,
    api_token: str = GENIUS_ACCESS_TOKEN,
) -> Dict[str, object]:

    snippets = lyric_snippets(transcript, max_lines=3)

    if not snippets:
        return {
            "lyrics": "",
            "source": "hybrid_match",
            "source_label": _source_label("hybrid_match"),
            "confidence": 0.0,
            "error": "No searchable transcript snippets.",
        }

    if not api_token:
        return {
            "lyrics": "",
            "source": "hybrid_match",
            "source_label": _source_label("hybrid_match"),
            "confidence": 0.0,
            "error": "GENIUS_ACCESS_TOKEN is not configured.",
        }

    try:
        import lyricsgenius

        genius = lyricsgenius.Genius(
            api_token,
            timeout=8,
            retries=1,
            remove_section_headers=True,
            skip_non_songs=True,
        )

        best: Dict[str, object] = {
            "lyrics": "",
            "source": "hybrid_match",
            "source_label": _source_label("hybrid_match"),
            "confidence": 0.0,
            "error": "No hybrid Genius match found.",
        }

        joined_query = " ".join(snippets)

        for snippet in snippets:
            song = retry_call(
                lambda: genius.search_song(title=snippet),
                attempts=1,
                delay_seconds=0.5,
            )

            if not song or not song.lyrics:
                continue

            lyrics = clean_genius_lyrics(song.lyrics)
            match_score = _score_match(joined_query, lyrics)

            if match_score > float(best["confidence"]):
                best = {
                    "lyrics": lyrics,
                    "source": "hybrid_match",
                    "source_label": _source_label("hybrid_match"),
                    "confidence": round(match_score, 2),
                    "error": "",
                    "matched_title": getattr(song, "title", ""),
                    "matched_artist": getattr(song, "artist", ""),
                }

        if float(best["confidence"]) >= 0.78:
            cache_path = _lyrics_cache_path(
                str(best.get("matched_title", "")),
                str(best.get("matched_artist", "")),
            )

            write_text_cache(cache_path, str(best["lyrics"]))
            best["cache_path"] = str(cache_path)

            return best

        return best

    except Exception as exc:
        return {
            "lyrics": "",
            "source": "hybrid_match",
            "source_label": _source_label("hybrid_match"),
            "confidence": 0.0,
            "error": str(exc),
        }


def fetch_lyrics_for_audio(
    audio_path: str | Path,
    api_token: str = GENIUS_ACCESS_TOKEN,
) -> Dict[str, object]:

    metadata = extract_song_metadata(audio_path)

    hint_text = " ".join(
        part
        for part in [metadata.get("title", ""), metadata.get("artist", "")]
        if part
    )

    metadata_result = fetch_lyrics_by_metadata(
        metadata["title"],
        metadata["artist"],
        api_token=api_token,
    )

    if metadata_result.get("lyrics"):
        metadata_result["metadata"] = metadata
        metadata_result.setdefault("language", "Unknown")
        metadata_result.setdefault("quality_score", 1.0)
        metadata_result.setdefault("message", "Lyrics found from metadata.")
        return metadata_result

    audio_result = extract_lyrics_from_audio(
        audio_path,
        hint_text=hint_text,
        use_vocals=True,
    )

    cleaned_transcript = clean_extracted_lyrics(
        str(
            audio_result.get("lyrics")
            or audio_result.get("raw_transcript", "")
        )
    )

    hybrid_result = _fetch_lyrics_by_transcript_snippets(
        cleaned_transcript,
        api_token=api_token,
    )

    if (
        hybrid_result.get("lyrics")
        and float(hybrid_result.get("confidence", 0.0)) >= 0.78
        and audio_result.get("quality_score", 0.0) >= 0.55
        and _language_matches(
            str(audio_result.get("language", "")),
            str(hybrid_result.get("lyrics", "")),
        )
    ):
        hybrid_result["metadata"] = metadata
        hybrid_result["language"] = audio_result.get("language", "Unknown")
        hybrid_result["quality_score"] = audio_result.get("quality_score", 0.0)
        hybrid_result["audio_transcript"] = cleaned_transcript
        hybrid_result["message"] = "Lyrics matched using audio transcript snippets."

        return hybrid_result

    if cleaned_transcript:
        confidence = float(audio_result.get("confidence", 0.0))

        fallback_source = (
            "audio_transcription"
            if confidence >= 0.45
            else "raw_transcript"
        )

        return {
            "lyrics": cleaned_transcript,
            "source": fallback_source,
            "source_label": _source_label(fallback_source),
            "confidence": confidence,
            "quality_score": audio_result.get("quality_score", 0.0),
            "language": audio_result.get("language", "Unknown"),
            "language_code": audio_result.get("language_code", "unknown"),
            "is_mixed_language": audio_result.get("is_mixed_language", False),
            "audio_source": audio_result.get("audio_source", "original_audio"),
            "transcription_engine": audio_result.get(
                "transcription_engine",
                "openai-whisper",
            ),
            "metadata": metadata,
            "error": "",
            "message": (
                "Lyrics reconstructed from audio."
                if confidence >= 0.45
                else "Partial lyrics extracted with low confidence."
            ),
        }

    return {
        "lyrics": "",
        "source": "raw_transcript",
        "source_label": _source_label("raw_transcript"),
        "confidence": 0.0,
        "quality_score": 0.0,
        "language": audio_result.get("language", "Unknown"),
        "metadata": metadata,
        "error": (
            audio_result.get("error")
            or metadata_result.get("error")
            or hybrid_result.get("error", "")
        ),
        "message": "Low confidence transcription. Try a clearer or longer vocal clip.",
    }