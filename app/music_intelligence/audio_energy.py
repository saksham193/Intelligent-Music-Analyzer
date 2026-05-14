"""Energy and intensity estimates from audio features."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def analyze_energy(audio_path: str | Path) -> dict[str, float]:
    """Estimate energy from RMS loudness and spectral movement."""
    y, sr = librosa.load(str(audio_path), duration=30, mono=True)
    rms = librosa.feature.rms(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    raw_energy = float(np.mean(rms) * 10)
    energy_score = float(np.clip(raw_energy, 0.0, 1.0))
    intensity = float(np.clip((np.std(rms) * 20) + (np.mean(zcr) * 3), 0.0, 1.0))
    return {
        "energy_score": energy_score,
        "emotional_intensity": intensity,
        "rms_energy": float(np.mean(rms)),
        "spectral_contrast": float(np.mean(contrast)),
        "zero_crossing_rate": float(np.mean(zcr)),
    }

