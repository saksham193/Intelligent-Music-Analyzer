"""Tempo and rhythm analysis."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def analyze_tempo(audio_path: str | Path) -> dict[str, float]:
    """Estimate BPM and rhythm density from onset activity."""
    y, sr = librosa.load(str(audio_path), duration=30, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr) or 1.0
    return {
        "tempo_bpm": float(np.asarray(tempo).item()),
        "beat_count": float(len(beats)),
        "rhythm_density": float(len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / duration),
    }

