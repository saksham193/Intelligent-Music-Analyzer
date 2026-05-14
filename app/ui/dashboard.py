"""Streamlit dashboard for the emotion-aware music platform."""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from html import escape
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import streamlit as st

from app.audio_processing.features import extract_features
from app.config.settings import CACHE_DIR, DEFAULT_RECORD_SECONDS, SUPPORTED_AUDIO_TYPES
from app.database.db import fetch_history, insert_history
from app.lyrics_engine.language_detector import detect_language
from app.lyrics_engine.language_router import detect_text_language, display_language, route_whisper_language
from app.lyrics_engine.lyrics_cleaner import clean_extracted_lyrics
from app.lyrics_engine.lyrics_fetcher import fetch_lyrics_for_audio
from app.lyrics_engine.song_metadata import extract_song_metadata
from app.lyrics_engine.translator import translate_text
from app.music_genre.genre_predict import predict_genre
from app.music_genre.visualizer import (
    mel_spectrogram_figure,
    spectrogram_figure as music_spectrogram_figure,
    waveform_figure as music_waveform_figure,
)
from app.music_intelligence.mood_analysis import analyze_music_mood
from app.music_intelligence.stem_separator import separate_stems
from app.recommendation.hybrid_recommender import fuse_multimodal_emotion
from app.recommendation.personalization.adaptive_engine import rank_recommendations
from app.recommendation.personalization.recommendation_memory import (
    fetch_feedback,
    remember_feedback,
)
from app.recommendation.personalization.user_profile import build_user_profile
from app.recommendation.recommender import PLACEHOLDER_COVER_URL, recommend_songs
from app.speech_emotion.fusion import fuse_emotions
from app.speech_emotion.predict import predict_emotion
from app.speech_emotion.text_sentiment import analyze_text_sentiment
from app.speech_emotion.transcriber import transcribe_audio
from app.ui.components.chat_input import render_chat_input
from app.ui.components.voice_recorder import (
    init_voice_recorder_state,
    render_recording_feedback,
    start_recording,
    stop_recording,
)
from app.utils.validators import validate_audio_upload
from app.visualization.advanced_dashboard import (
    daily_mood_timeline,
    language_preference_chart,
    mood_heatmap,
    recommendation_feedback_chart,
)
from app.visualization.charts import (
    bpm_chart,
    emotion_confidence_chart,
    mood_analytics_chart,
)
from app.visualization.music_galaxy import music_galaxy_figure


LOGGER = logging.getLogger(__name__)
VISIBLE_RECOMMENDATION_COUNT = 5


def _save_uploaded_file(uploaded_file, prefix: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    output_path = CACHE_DIR / f"{prefix}_{safe_name}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def _fallback_voice_result(audio_path: Path) -> Dict[str, object]:
    features = extract_features(audio_path)
    return {
        "emotion": "neutral",
        "confidence": 0.0,
        "confidence_scores": {"neutral": 1.0},
        "features": features,
    }


def _normalize_voice_transcript(text: str) -> str:
    """Clean Whisper text while preserving valid Hindi Unicode characters."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\ufffd", " ")
    normalized = "".join(
        char for char in normalized
        if char == "\n" or char == "\t" or not unicodedata.category(char).startswith("C")
    )
    normalized = clean_extracted_lyrics(normalized)
    return " ".join(normalized.split())


def _transcribe_voice(audio_path: Path, language: str) -> Dict[str, object]:
    if language == "Auto":
        route = route_whisper_language(audio_path)
        whisper_language = route.get("whisper_language")
    else:
        route = {"is_mixed": False}
        whisper_language = {"English": "en", "Hindi": "hi"}[language]

    result = transcribe_audio(audio_path, language=whisper_language)
    normalized_text = _normalize_voice_transcript(str(result.get("text", "")))
    detected = detect_text_language(normalized_text)
    language_code = str(detected.get("language") or result.get("language") or "unknown")
    is_mixed = bool(detected.get("is_mixed") or route.get("is_mixed"))

    result["text"] = normalized_text
    result["language"] = display_language(language_code, is_mixed=is_mixed)
    result["language_code"] = language_code
    result["is_mixed_language"] = is_mixed
    return result


def _text_weighted_final_mood(
    voice_emotion: str,
    text_sentiment: str,
    sentiment_confidence: float,
    transcript_text: str,
) -> str:
    """Prefer clear user language over noisy vocal tone for recommendation mood."""
    base_mood = fuse_emotions(voice_emotion, text_sentiment)
    text = transcript_text.lower()
    sentiment = (text_sentiment or "neutral").lower()

    if sentiment_confidence < 0.45 or not transcript_text.strip():
        return base_mood

    negative_cues = {
        "stress": "anxious",
        "stressed": "anxious",
        "anxiety": "anxious",
        "tension": "anxious",
        "lonely": "sad",
        "alone": "sad",
        "sad": "sad",
        "down": "sad",
        "angry": "angry",
        "gussa": "angry",
        "dukhi": "sad",
        "tanha": "sad",
        "pareshan": "anxious",
    }
    positive_cues = {
        "relax": "calm",
        "relaxing": "calm",
        "calm": "calm",
        "peace": "calm",
        "happy": "happy",
        "excited": "happy",
        "khush": "happy",
        "sukoon": "calm",
    }

    for cue, mood in {**negative_cues, **positive_cues}.items():
        if cue in text:
            return mood

    if sentiment == "negative":
        return "sad" if voice_emotion.lower() not in {"angry", "fearful"} else base_mood
    if sentiment == "positive":
        return "happy" if voice_emotion.lower() != "calm" else "calm"
    return base_mood


def _analyze_voice_prompt(
    prompt_text: str,
    voice_path: Optional[Path],
    language: str,
) -> Dict[str, object]:
    transcript = {"text": prompt_text.strip(), "language": language, "segments": [], "error": ""}
    voice_result = {
        "emotion": "neutral",
        "confidence": 0.0,
        "confidence_scores": {"neutral": 1.0},
        "features": {"tempo_bpm": 0.0},
    }

    if voice_path:
        if not transcript["text"]:
            transcript = _transcribe_voice(voice_path, language)

        try:
            voice_result = predict_emotion(voice_path)
        except Exception as exc:
            st.warning(f"Voice model unavailable, using neutral fallback. Details: {exc}")
            voice_result = _fallback_voice_result(voice_path)
    else:
        transcript["text"] = _normalize_voice_transcript(transcript["text"])
        detected = detect_text_language(transcript["text"])
        language_code = str(detected.get("language", language))
        is_mixed = bool(detected.get("is_mixed"))
        transcript["language"] = display_language(language_code, is_mixed=is_mixed)
        transcript["language_code"] = language_code
        transcript["is_mixed_language"] = is_mixed

    sentiment = analyze_text_sentiment(str(transcript.get("text", "")))
    final_emotion = _text_weighted_final_mood(
        str(voice_result["emotion"]),
        str(sentiment["sentiment"]),
        float(sentiment.get("confidence", 0.0)),
        str(transcript.get("text", "")),
    )
    recommendations = recommend_songs(
        final_emotion,
        limit=12,
        language=str(transcript.get("language", language)),
        sentiment=str(sentiment["sentiment"]),
        bpm=float(voice_result["features"].get("tempo_bpm", 0.0)),
    )

    insert_history(
        voice_emotion=str(voice_result["emotion"]),
        text_sentiment=str(sentiment["sentiment"]),
        final_emotion=final_emotion,
        transcript=str(transcript.get("text", "")),
        language=str(transcript.get("language", language)),
        bpm=float(voice_result["features"].get("tempo_bpm", 0.0)),
        recommended_songs=recommendations,
    )

    return {
        "voice": voice_result,
        "transcript": transcript,
        "sentiment": sentiment,
        "final_emotion": final_emotion,
        "recommendations": recommendations,
    }


def _apply_lightweight_css() -> None:
    """Apply lightweight custom CSS."""
    chatbot_css_path = Path(__file__).parent / "styles" / "chatbot.css"
    chatbot_styles = ""
    if chatbot_css_path.exists():
        chatbot_styles = chatbot_css_path.read_text(encoding="utf-8")

    custom_css = """
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.8rem;
        margin-right: 0.35rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    """

    st.markdown(
        f"""
        <style>
        {chatbot_styles}
        {custom_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _song_key(song: dict[str, object]) -> str:
    value = f"{song.get('title', '')}|{song.get('artist', '')}".lower()
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _init_recommendation_state() -> None:
    st.session_state.setdefault("liked_recommendation_keys", set())
    st.session_state.setdefault("skipped_recommendation_keys", set())


def _apply_session_preference_boost(recommendations: list[dict[str, str]]) -> list[dict[str, str]]:
    _init_recommendation_state()
    liked_keys = set(st.session_state.liked_recommendation_keys)
    boosted = []
    for song in recommendations:
        enriched = dict(song)
        if _song_key(song) in liked_keys:
            enriched["personalized_score"] = round(float(enriched.get("personalized_score", 1.0)) + 0.2, 3)
        boosted.append(enriched)
    return sorted(boosted, key=lambda item: float(item.get("personalized_score", 1.0)), reverse=True)


def _dedupe_recommendations(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for song in candidates:
        title = str(song.get("title") or "").strip()
        artist = str(song.get("artist") or "").strip()
        if not title and not artist:
            continue
        key = (title.lower(), artist.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(song)
    return unique


def _is_valid_image_url(url: str) -> bool:
    if not url:
        return False
    if url.startswith("data:image/"):
        return True
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _resolve_cover_url(song: dict[str, object]) -> tuple[str, str]:
    candidates = [
        ("spotify", song.get("spotify_album_image")),
        ("spotify", song.get("spotify_image")),
        ("spotify", song.get("album_image")),
        ("lastfm", song.get("cover_url")),
        ("lastfm", song.get("image")),
        ("youtube", song.get("youtube_thumbnail")),
        ("youtube", song.get("thumbnail_url")),
    ]
    for source, value in candidates:
        url = str(value or "").strip()
        if url == PLACEHOLDER_COVER_URL:
            continue
        if _is_valid_image_url(url):
            LOGGER.debug("Resolved recommendation cover: source=%s url=%s", source, url)
            return url, source
        if url:
            LOGGER.debug("Rejected invalid recommendation cover URL: source=%s url=%s", source, url)

    LOGGER.debug(
        "Using fallback recommendation cover: title=%s artist=%s",
        song.get("title"),
        song.get("artist"),
    )
    return PLACEHOLDER_COVER_URL, "fallback"


def _render_music_card(song: dict[str, object], mood: str, language: str, index: int) -> str:
    title = escape(str(song.get("title") or "Unknown title"))
    artist = escape(str(song.get("artist") or "Unknown artist"))
    url = escape(str(song.get("url") or "#"), quote=True)
    card_mood = escape(str(song.get("mood") or mood or "Mood match"))
    card_language = escape(str(song.get("language") or language or "Music"))
    score = float(song.get("personalized_score", 1.0) or 1.0)
    cover, cover_source = _resolve_cover_url(song)
    cover = escape(cover, quote=True)
    if cover_source == "fallback":
        cover_html = '<div class="music-cover-fallback" aria-label="Music cover">&#9834;</div>'
    else:
        cover_html = f'<img src="{cover}" class="music-cover" alt="{title} cover" loading="lazy" />'

    html = f"""
    <div class="music-card" data-card-index="{index}">
        <div class="music-cover-wrap">
            {cover_html}
        </div>
        <div class="music-info">
            <div class="music-meta">
                <span>{card_mood}</span>
                <span>{card_language}</span>
                <span>{score:.2f} match</span>
                <span>{escape(cover_source)} art</span>
            </div>
            <div class="music-title">{title}</div>
            <div class="music-artist">{artist}</div>
        </div>
        <div class="music-card-play">
            <a class="music-play-link" href="{url}" target="_blank" rel="noopener noreferrer" title="Play">
                &#9654;
            </a>
        </div>
    </div>
    """
    if html.count("<div") != html.count("</div>"):
        LOGGER.debug("Malformed recommendation card HTML detected: title=%s artist=%s", title, artist)
    else:
        LOGGER.debug("Rendered recommendation card HTML: title=%s artist=%s", title, artist)
    return html


def _visible_recommendations(
    recommendations: list[dict[str, str]],
    mood: str,
    language: str,
    genre: str = "",
) -> list[dict[str, str]]:
    _init_recommendation_state()
    skipped_keys = set(st.session_state.skipped_recommendation_keys)
    candidates = list(recommendations)
    backup_contexts = [mood, genre, "neutral", "happy", "calm", "sad", "angry"]
    for context in backup_contexts:
        if context:
            candidates.extend(recommend_songs(str(context), limit=12, language=language))

    visible = [
        song for song in _dedupe_recommendations(candidates)
        if _song_key(song) not in skipped_keys
    ]
    LOGGER.debug(
        "Rendering recommendations: received=%s visible=%s sources=%s skipped=%s",
        len(recommendations),
        min(len(visible), VISIBLE_RECOMMENDATION_COUNT),
        sorted({str(song.get("source_detail", song.get("source", "unknown"))) for song in visible}),
        len(skipped_keys),
    )
    return visible[:VISIBLE_RECOMMENDATION_COUNT]


def _render_recommendations(recommendations, mood: str = "", genre: str = "", language: str = "") -> None:
    _init_recommendation_state()
    visible = _visible_recommendations(list(recommendations or []), mood, language, genre=genre)
    if len(visible) < VISIBLE_RECOMMENDATION_COUNT:
        st.info("Recommendations are still warming up. Try another prompt if fewer than five songs appear.")

    for index, song in enumerate(visible):
        key = _song_key(song)
        title = str(song.get("title") or "Unknown title")
        artist = str(song.get("artist") or "Unknown artist")
        st.markdown(_render_music_card(song, mood, language, index), unsafe_allow_html=True)
        st.markdown('<div class="music-action-row-marker"></div>', unsafe_allow_html=True)
        like_col, skip_col, spacer_col = st.columns([0.12, 0.12, 0.76])
        if like_col.button("❤️", key=f"like_song_{index}_{key}", help="Like this song"):
            st.session_state.liked_recommendation_keys.add(key)
            remember_feedback(title, artist, mood, genre, language, "liked")
            st.toast("Preference saved.")
            st.rerun()
        if skip_col.button("⏭", key=f"skip_song_{index}_{key}", help="Skip this song"):
            st.session_state.skipped_recommendation_keys.add(key)
            remember_feedback(title, artist, mood, genre, language, "skipped")
            st.toast("Skipped. Loading another recommendation.")
            st.rerun()

    LOGGER.debug(
        "Rendered recommendation cards: count=%s titles=%s",
        len(visible),
        [f"{song.get('title')} - {song.get('artist')}" for song in visible],
    )


def _render_result_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <span>{escape(label)}</span>
            <strong>{escape(value)}</strong>
            <small>{escape(detail)}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_processed_voice_path() -> None:
    """Clear temporary recorded/uploaded audio from active state after analysis."""
    st.session_state.voice_path = None
    st.session_state.voice_source = ""


def _render_voice_analysis() -> None:
    st.markdown(
        """
        <div class="voice-hero">
            <h2>Voice Emotion Analysis</h2>
            <p>Your conversational music assistant for typed prompts, spoken feelings, and human voice notes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "voice_analysis" not in st.session_state:
        st.session_state.voice_analysis = None
    if "voice_path" not in st.session_state:
        st.session_state.voice_path = None
    if "prompt_text" not in st.session_state:
        st.session_state.prompt_text = ""
    if "voice_source" not in st.session_state:
        st.session_state.voice_source = ""
    if "voice_processing" not in st.session_state:
        st.session_state.voice_processing = False
    if "pending_voice_auto_analyze" not in st.session_state:
        st.session_state.pending_voice_auto_analyze = False
    init_voice_recorder_state()

    option_cols = st.columns([0.42, 0.58])
    language = option_cols[0].selectbox("Prompt language", ["Auto", "English", "Hindi"])
    duration = option_cols[1].slider(
        "Mic duration",
        3,
        20,
        DEFAULT_RECORD_SECONDS,
        help="Suggested recording length. Click Mic again whenever you are done speaking.",
    )

    if st.session_state.recorder_active:
        render_recording_feedback(duration)

    mic_clicked, analyze_clicked, uploaded_voice = render_chat_input(
        text_key="prompt_text",
        mic_active=bool(st.session_state.recorder_active),
        disabled=bool(st.session_state.voice_processing),
    )

    if mic_clicked:
        if st.session_state.recorder_active:
            with st.spinner("Stopping recording and transcribing your voice..."):
                try:
                    recorded_path = stop_recording()
                    if recorded_path:
                        st.session_state.voice_path = recorded_path
                        st.session_state.voice_source = "recorded"
                        transcript = _transcribe_voice(recorded_path, language)
                        st.session_state.prompt_text = str(transcript.get("text", "")).strip()
                        st.session_state.pending_voice_auto_analyze = True
                        st.toast("Recording saved and transcribed.")
                        if st.session_state.get("recorder_error"):
                            st.warning(f"Microphone warning: {st.session_state.recorder_error}")
                    else:
                        st.warning("No microphone audio was captured. Please try again.")
                except Exception as exc:
                    st.error(f"Recording could not be saved: {exc}")
            st.rerun()
        else:
            try:
                start_recording()
                st.toast("Recording started. Speak now.")
            except Exception as exc:
                st.error(str(exc))
            st.rerun()

    if uploaded_voice:
        ok, error = validate_audio_upload(uploaded_voice)
        if not ok:
            st.error(error)
        else:
            st.session_state.voice_path = _save_uploaded_file(uploaded_voice, "voice")
            st.session_state.voice_source = "uploaded"
            st.success("Voice note attached. Press Send to analyze it.")

    should_analyze = bool(analyze_clicked or st.session_state.pending_voice_auto_analyze)
    if should_analyze and not st.session_state.voice_processing:
        st.session_state.pending_voice_auto_analyze = False
        prompt = st.session_state.prompt_text.strip()
        voice_path = Path(st.session_state.voice_path) if st.session_state.voice_path else None
        if not prompt and not voice_path:
            st.error("Type a prompt, record with Mic, or upload a voice note first.")
        else:
            st.session_state.voice_processing = True
            try:
                with st.spinner("Analyzing speech emotion, sentiment, and recommendations..."):
                    st.session_state.voice_analysis = _analyze_voice_prompt(prompt, voice_path, language)
                _clear_processed_voice_path()
            except Exception as exc:
                st.error(f"Could not complete voice analysis: {exc}")
            finally:
                st.session_state.voice_processing = False

    analysis = st.session_state.voice_analysis
    if not analysis:
        st.info("Start with a short message, a recorded prompt, or an uploaded human voice note.")
        return

    voice = analysis["voice"]
    sentiment = analysis["sentiment"]
    transcript = analysis["transcript"]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        _render_result_card("Voice Emotion", str(voice["emotion"]).title(), f"Confidence {voice['confidence']:.0%}")
    with metric_cols[1]:
        _render_result_card("Text Sentiment", str(sentiment["sentiment"]).title(), f"Confidence {sentiment['confidence']:.0%}")
    with metric_cols[2]:
        _render_result_card("Final Mood", str(analysis["final_emotion"]).title(), "Music recommendation mood")
    with metric_cols[3]:
        _render_result_card("BPM", f"{voice['features'].get('tempo_bpm', 0):.1f}", "Voice tempo estimate")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Transcript")
        if transcript.get("error"):
            st.warning(f"Transcription unavailable: {transcript['error']}")
        st.markdown(
            f"""
            <div class="transcript-bubble">
                {escape(str(transcript.get("text") or st.session_state.prompt_text or "No speech text detected."))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Detected language: {transcript.get('language', 'unknown')}")

        st.subheader("Recommended songs")
        ranked = rank_recommendations(analysis["recommendations"], mood=str(analysis["final_emotion"]))
        ranked = _apply_session_preference_boost(ranked)
        _render_recommendations(ranked, mood=str(analysis["final_emotion"]), language=str(transcript.get("language", "")))

    with right:
        st.subheader("Emotion confidence")
        st.plotly_chart(emotion_confidence_chart(voice["confidence_scores"]), width="stretch")
        st.plotly_chart(bpm_chart(float(voice["features"].get("tempo_bpm", 0.0))), width="stretch")


def _render_music_genre_analyzer() -> None:
    st.header("Music Genre Analyzer")
    st.caption("Upload full songs here for genre prediction and audio feature visualization.")

    uploaded_music = st.file_uploader(
        "Upload music file",
        type=list(SUPPORTED_AUDIO_TYPES),
        key="music_upload",
    )
    if not uploaded_music:
        st.info("Upload a song or music clip to analyze genre.")
        return

    ok, error = validate_audio_upload(uploaded_music)
    if not ok:
        st.error(error)
        return

    music_path = _save_uploaded_file(uploaded_music, "music")
    st.audio(str(music_path))

    try:
        with st.spinner("Predicting genre and extracting music intelligence..."):
            result = predict_genre(music_path)
            mood_result = analyze_music_mood(music_path)
    except Exception as exc:
        st.error(f"Could not analyze music file: {exc}")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted genre", str(result["genre"]).title(), f"{result['confidence']:.0%}")
    metric_cols[1].metric("Music mood", str(mood_result["mood"]))
    metric_cols[2].metric("Valence", f"{mood_result['valence_score']:.2f}")
    metric_cols[3].metric("Energy", f"{mood_result['energy_score']:.2f}")

    left, right = st.columns([0.9, 1.1])
    with left:
        st.subheader("Genre confidence")
        st.plotly_chart(emotion_confidence_chart(result["confidence_scores"]), width="stretch")
    with right:
        st.subheader("Mood intelligence")
        st.progress(float(mood_result["energy_score"]), text="Energy")
        st.progress(float(mood_result["valence_score"]), text="Valence")
        st.progress(float(mood_result["emotional_intensity"]), text="Emotional intensity")
        st.plotly_chart(bpm_chart(float(mood_result["tempo_bpm"])), width="stretch")

    metadata = extract_song_metadata(music_path)
    st.subheader("Lyrics Intelligence")
    title_col, artist_col = st.columns(2)
    title = title_col.text_input("Song title", value=metadata["title"])
    artist = artist_col.text_input("Artist", value=metadata["artist"])

    lyric_col, translate_col = st.columns(2)
    if "lyrics_result" not in st.session_state:
        st.session_state.lyrics_result = None
    if "translated_lyrics" not in st.session_state:
        st.session_state.translated_lyrics = None

    if lyric_col.button("Extract Lyrics", width="stretch"):
        with st.spinner("Extracting lyrics from metadata, vocals, and transcript..."):
            manual_result = None
            if title:
                from app.lyrics_engine.lyrics_fetcher import fetch_lyrics_by_metadata

                manual_result = fetch_lyrics_by_metadata(title, artist)
            if manual_result and manual_result.get("lyrics"):
                st.session_state.lyrics_result = manual_result
                st.session_state.lyrics_result["metadata"] = {"title": title, "artist": artist, "query": f"{title} {artist}".strip()}
                st.session_state.lyrics_result.setdefault("message", "Lyrics found from manual metadata.")
            else:
                st.session_state.lyrics_result = fetch_lyrics_for_audio(music_path)
        st.session_state.translated_lyrics = None

    lyrics_text = ""
    lyrics_sentiment = {"sentiment": "neutral", "confidence": 0.0}
    lyrics_language = {"language": "unknown", "confidence": 0.0}
    lyrics_source_language_code = "auto"
    if st.session_state.lyrics_result:
        lyrics_result = st.session_state.lyrics_result
        if lyrics_result.get("error") and not lyrics_result.get("lyrics"):
            st.warning(f"Lyrics unavailable: {lyrics_result['error']}")
        lyrics_text = str(lyrics_result.get("lyrics", ""))
        source_label = str(lyrics_result.get("source_label") or lyrics_result.get("source", "unknown")).replace("_", " ").title()
        result_language = str(lyrics_result.get("language") or "Unknown")
        confidence = float(lyrics_result.get("confidence", 0.0) or 0.0)
        quality = float(lyrics_result.get("quality_score", 0.0) or 0.0)
        st.caption(str(lyrics_result.get("message") or "Lyrics intelligence result."))
        lyric_metrics = st.columns(4)
        with lyric_metrics[0]:
            _render_result_card("Lyrics Source", source_label, str(lyrics_result.get("audio_source", "")))
        with lyric_metrics[1]:
            _render_result_card("Confidence", f"{confidence:.0%}", "Match/transcription confidence")
        with lyric_metrics[2]:
            _render_result_card("Detected Language", result_language, "English, Hindi, or Hinglish")
        with lyric_metrics[3]:
            _render_result_card("Transcript Quality", f"{quality:.0%}", str(lyrics_result.get("transcription_engine", "")))
        if lyrics_text:
            lyrics_language = detect_language(lyrics_text)
            lyrics_source_language_code = str(lyrics_language.get("language", "auto"))
            result_language_code = str(lyrics_result.get("language_code", ""))
            if result_language_code in {"en", "hi"}:
                lyrics_source_language_code = result_language_code
            if result_language != "Unknown":
                lyrics_language = {"language": result_language, "confidence": confidence}
            lyrics_sentiment = analyze_text_sentiment(lyrics_text[:512])
            st.caption(
                f"Lyrics source: {source_label} | "
                f"Language: {lyrics_language.get('language', 'unknown')} | "
                f"Sentiment: {lyrics_sentiment['sentiment']}"
            )
            st.markdown(f"<div class='lyrics-box'>{escape(lyrics_text)}</div>", unsafe_allow_html=True)
        else:
            st.info(str(lyrics_result.get("message") or "Partial lyrics could not be reconstructed from this clip."))

    target_language = st.selectbox("Translate lyrics to", ["en", "hi"], format_func=lambda item: "English" if item == "en" else "Hindi")
    if translate_col.button("Translate Lyrics", width="stretch", disabled=not bool(lyrics_text)):
        with st.spinner("Translating lyrics..."):
            st.session_state.translated_lyrics = translate_text(
                lyrics_text,
                target_language=target_language,
                source_language=lyrics_source_language_code if lyrics_source_language_code != "unknown" else "auto",
            )
    if st.session_state.translated_lyrics:
        translated = st.session_state.translated_lyrics
        if translated.get("error"):
            st.warning(f"Translation unavailable: {translated['error']}")
        elif translated.get("text"):
            st.markdown("Translated lyrics")
            st.markdown(f"<div class='lyrics-box'>{escape(str(translated['text']))}</div>", unsafe_allow_html=True)

    st.subheader("Hybrid Music Intelligence")
    hybrid = fuse_multimodal_emotion(
        lyrics_sentiment=str(lyrics_sentiment["sentiment"]),
        music_mood=str(mood_result["mood"]),
        genre=str(result["genre"]),
        energy_score=float(mood_result["energy_score"]),
        valence_score=float(mood_result["valence_score"]),
    )
    hybrid_cols = st.columns(3)
    hybrid_cols[0].metric("Interpretation", str(hybrid["final_interpretation"]))
    hybrid_cols[1].metric("Dissonance", f"{hybrid['dissonance_score']:.2f}")
    hybrid_cols[2].metric("Confidence", f"{hybrid['confidence']:.0%}")
    st.caption(str(hybrid["explanation"]))

    with st.expander("Music Galaxy", expanded=False):
        st.plotly_chart(
            music_galaxy_figure(
                [
                    {
                        "title": metadata["title"] or uploaded_music.name,
                        "energy": float(mood_result["energy_score"]),
                        "valence": float(mood_result["valence_score"]),
                        "tempo": float(mood_result["tempo_bpm"]),
                        "cluster": str(mood_result["mood"]),
                    }
                ]
            ),
            width="stretch",
        )

    with st.expander("Optional Stem Separation", expanded=False):
        st.caption("Demucs is optional and can be slow. Use it only for short demo clips.")
        if st.button("Separate Vocals / Karaoke", width="stretch"):
            with st.spinner("Running optional stem separation..."):
                stems = separate_stems(music_path)
            if not stems["ok"]:
                st.warning(stems["error"])
            elif stems["stems"]:
                for name, path in stems["stems"].items():
                    st.markdown(name.title())
                    st.audio(path)
            else:
                st.info("Stem process finished, but no playable stems were found.")

    with st.expander("Audio visualizations", expanded=True):
        col1, col2 = st.columns(2)
        col1.pyplot(music_waveform_figure(music_path))
        col2.pyplot(music_spectrogram_figure(music_path))
        st.pyplot(mel_spectrogram_figure(music_path))


def _render_analytics() -> None:
    st.header("Analytics / History")
    history = fetch_history()
    feedback = fetch_feedback()
    chart_tab, profile_tab, raw_tab = st.tabs(["Trends", "User Profile", "Raw History"])
    with chart_tab:
        col1, col2 = st.columns(2)
        col1.plotly_chart(mood_analytics_chart(history), width="stretch")
        col2.plotly_chart(language_preference_chart(history), width="stretch")
        st.plotly_chart(daily_mood_timeline(history), width="stretch")
        st.plotly_chart(mood_heatmap(history), width="stretch")
        st.plotly_chart(recommendation_feedback_chart(feedback), width="stretch")
    with profile_tab:
        st.json(build_user_profile(), expanded=True)
    with raw_tab:
        if history:
            st.dataframe(history, width="stretch", hide_index=True)
        else:
            st.info("No analysis history yet.")


def render_dashboard() -> None:
    """Render the full Streamlit app."""
    st.set_page_config(page_title="Emotion Music AI", page_icon="music", layout="wide")
    _apply_lightweight_css()
    st.title("Emotion-Aware Music Recommendation System")
    st.caption("Separated speech emotion, sentiment, music genre, and recommendation pipelines.")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Section",
            ["Voice Emotion Analysis", "Music Genre Analyzer", "Analytics / History"],
            label_visibility="collapsed",
        )

    if page == "Voice Emotion Analysis":
        _render_voice_analysis()
    elif page == "Music Genre Analyzer":
        _render_music_genre_analyzer()
    else:
        _render_analytics()
