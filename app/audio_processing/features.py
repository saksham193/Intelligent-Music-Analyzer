"""Audio feature extraction for voice emotion detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import librosa
import numpy as np

from app.config.settings import DEFAULT_SAMPLE_RATE


def _summary_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Convert a feature matrix into stable mean/std/min/max summary values."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    stats: Dict[str, float] = {}
    for idx, row in enumerate(values):
        stats[f"{prefix}_{idx}_mean"] = float(np.mean(row))
        stats[f"{prefix}_{idx}_std"] = float(np.std(row))
    return stats


def extract_pitch(y: np.ndarray, sr: int) -> float:
    """Estimate average pitch using librosa's probabilistic YIN algorithm."""
    try:
        pitches, voiced_flags, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_pitches = pitches[voiced_flags]
        if voiced_pitches.size == 0:
            return 0.0
        return float(np.nanmean(voiced_pitches))
    except Exception:
        return 0.0


def extract_tempo(y: np.ndarray, sr: int) -> float:
    """Estimate tempo/BPM from the audio signal."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(np.asarray(tempo).item())
    except Exception:
        return 0.0


def extract_features(audio_path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Dict[str, float]:
    """Extract normalized speech/audio features from a WAV/MP3 file.

    The return value is a flat dictionary so it works naturally with pandas,
    scikit-learn, and Streamlit display code.
    """
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("Audio file is empty or could not be decoded.")

    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError("Audio contains only silence after trimming.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features: Dict[str, float] = {}
    features.update(_summary_stats(mfcc, "mfcc"))
    features.update(_summary_stats(chroma, "chroma"))
    features.update(_summary_stats(rms, "rms"))
    features.update(_summary_stats(spectral_contrast, "spectral_contrast"))
    features.update(_summary_stats(zcr, "zcr"))
    features["pitch_mean"] = extract_pitch(y, sr)
    features["tempo_bpm"] = extract_tempo(y, sr)

    return features

