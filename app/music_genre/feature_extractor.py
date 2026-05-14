"""Feature extraction for uploaded music files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import librosa
import numpy as np


def extract_genre_features(audio_path: str | Path) -> np.ndarray:
    """Extract the same feature vector used by the previous genre model."""
    y, sr = librosa.load(str(audio_path), duration=30, mono=True)
    if y.size == 0:
        raise ValueError("Music file is empty or could not be decoded.")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = np.hstack(
        [
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
        ]
    )
    return features.reshape(1, -1)


def extract_music_summary(audio_path: str | Path) -> Dict[str, float]:
    """Extract display-friendly music features for the UI."""
    y, sr = librosa.load(str(audio_path), duration=30, mono=True)
    if y.size == 0:
        raise ValueError("Music file is empty or could not be decoded.")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return {
        "tempo_bpm": float(np.asarray(tempo).item()),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "zero_crossing_rate": float(np.mean(zcr)),
        "rms_energy": float(np.mean(rms)),
    }

