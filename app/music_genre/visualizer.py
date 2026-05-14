"""Music visualization wrappers."""

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def waveform_figure(audio_path: str | Path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def spectrogram_figure(audio_path: str | Path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    stft = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def mel_spectrogram_figure(audio_path: str | Path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

