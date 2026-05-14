"""Shared audio preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from app.config.settings import DEFAULT_SAMPLE_RATE


def load_mono_audio(audio_path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as mono and trim leading/trailing silence."""
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("Audio file is empty or could not be decoded.")
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError("Audio contains only silence after trimming.")
    return y, sr

