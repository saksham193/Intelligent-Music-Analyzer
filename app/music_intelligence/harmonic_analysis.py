"""Harmonic and brightness analysis."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def analyze_harmony(audio_path: str | Path) -> dict[str, float]:
    """Estimate harmonic intensity and brightness."""
    y, sr = librosa.load(str(audio_path), duration=30, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    harmonic_power = float(np.mean(np.abs(harmonic)))
    percussive_power = float(np.mean(np.abs(percussive)))
    total = harmonic_power + percussive_power + 1e-8
    brightness = float(np.clip(np.mean(centroid) / 5000, 0.0, 1.0))

    return {
        "harmonic_intensity": float(harmonic_power / total),
        "percussive_intensity": float(percussive_power / total),
        "brightness": brightness,
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_rolloff": float(np.mean(rolloff)),
    }

