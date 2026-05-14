"""Microphone recording helpers.

This module uses only free, local Python libraries:
- sounddevice captures microphone audio.
- scipy.io.wavfile writes standard WAV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile

from app.config.settings import CACHE_DIR, DEFAULT_RECORD_SECONDS, DEFAULT_SAMPLE_RATE


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio safely and return int16 data for WAV writing."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Recorded audio is empty.")

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


def record_audio(
    duration: int = DEFAULT_RECORD_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
) -> np.ndarray:
    """Record voice from the default microphone.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Samples per second.
        channels: Number of input channels. Mono is recommended for speech.

    Returns:
        Normalized int16 audio samples.
    """
    if duration <= 0:
        raise ValueError("Duration must be greater than zero seconds.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    try:
        import sounddevice as sd

        frames = int(duration * sample_rate)
        audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
        sd.wait()
    except Exception as exc:  # pragma: no cover - depends on local microphone.
        raise RuntimeError(
            "Could not record audio. Check microphone permissions and input device."
        ) from exc

    audio = np.squeeze(audio)
    return _normalize_audio(audio)


def save_audio(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    file_path: Optional[str | Path] = None,
) -> Path:
    """Save audio samples as a WAV file and return the saved path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(file_path) if file_path else CACHE_DIR / "recorded_voice.wav"

    try:
        wavfile.write(output_path, sample_rate, np.asarray(audio, dtype=np.int16))
    except Exception as exc:
        raise RuntimeError(f"Failed to save audio to {output_path}") from exc

    return output_path

