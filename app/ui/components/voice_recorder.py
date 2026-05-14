"""Nonblocking microphone recorder for the voice assistant UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from scipy.io import wavfile

from app.config.settings import CACHE_DIR, DEFAULT_SAMPLE_RATE


def init_voice_recorder_state() -> None:
    """Initialize Streamlit state keys used by the microphone recorder."""
    defaults = {
        "recorder_active": False,
        "recorder_stream": None,
        "recorder_chunks": [],
        "recorder_status_messages": [],
        "recorder_started_at": None,
        "recorder_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Recorded audio is empty.")

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


def start_recording(sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Start microphone capture and keep chunks in session state."""
    init_voice_recorder_state()
    if st.session_state.recorder_active:
        return

    chunks = []
    status_messages = []
    st.session_state.recorder_chunks = chunks
    st.session_state.recorder_error = ""

    try:
        import sounddevice as sd

        def _callback(indata, frames, time, status) -> None:  # noqa: ANN001
            if status:
                status_messages.append(str(status))
            chunks.append(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        stream.start()
    except Exception as exc:  # pragma: no cover - depends on local microphone.
        st.session_state.recorder_active = False
        st.session_state.recorder_stream = None
        raise RuntimeError(
            "Could not start recording. Check microphone permissions and input device."
        ) from exc

    st.session_state.recorder_stream = stream
    st.session_state.recorder_status_messages = status_messages
    st.session_state.recorder_started_at = datetime.now()
    st.session_state.recorder_active = True


def stop_recording(sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[Path]:
    """Stop microphone capture, save the WAV file, and clear recorder state."""
    init_voice_recorder_state()
    stream = st.session_state.recorder_stream

    try:
        if stream is not None:
            stream.stop()
            stream.close()
    finally:
        st.session_state.recorder_stream = None
        st.session_state.recorder_active = False
        st.session_state.recorder_started_at = None

    chunks = st.session_state.recorder_chunks or []
    st.session_state.recorder_chunks = []
    status_messages = st.session_state.get("recorder_status_messages") or []
    st.session_state.recorder_status_messages = []
    if status_messages:
        st.session_state.recorder_error = "; ".join(status_messages[-3:])

    if not chunks:
        return None

    audio = np.concatenate(chunks, axis=0).squeeze()
    normalized = _normalize_audio(audio)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = CACHE_DIR / f"recorded_voice_{stamp}.wav"
    wavfile.write(output_path, sample_rate, normalized)
    return output_path


def recording_seconds() -> int:
    """Return elapsed recording seconds for status copy."""
    started_at = st.session_state.get("recorder_started_at")
    if not started_at:
        return 0
    return max(0, int((datetime.now() - started_at).total_seconds()))


def render_recording_feedback(max_seconds: int) -> None:
    """Render animated feedback while the microphone stream is active."""
    elapsed = recording_seconds()
    st.markdown(
        f"""
        <div class="recording-status">
            <div class="recording-copy">
                <strong>Recording... Speak now</strong>
                <span>Click the active mic again to stop. {elapsed}s / {max_seconds}s suggested.</span>
            </div>
            <div class="waveform" aria-label="Recording waveform">
                <span></span><span></span><span></span><span></span>
                <span></span><span></span><span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
