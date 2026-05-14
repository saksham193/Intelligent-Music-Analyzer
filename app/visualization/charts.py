"""Charts for explainable emotion and audio analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def emotion_confidence_chart(confidence_scores: Dict[str, float]) -> go.Figure:
    """Create a bar chart for emotion confidence scores."""
    labels = list(confidence_scores.keys())
    values = list(confidence_scores.values())
    fig = px.bar(x=labels, y=values, labels={"x": "Emotion", "y": "Confidence"})
    fig.update_layout(yaxis_range=[0, 1], margin=dict(l=10, r=10, t=25, b=10))
    return fig


def bpm_chart(bpm: float) -> go.Figure:
    """Create a compact BPM gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(bpm or 0),
            title={"text": "Tempo / BPM"},
            gauge={"axis": {"range": [0, 220]}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def mood_analytics_chart(history: List[Dict[str, object]]) -> go.Figure:
    """Create a count chart of previously detected final emotions."""
    moods = [str(item.get("final_emotion", "neutral")) for item in history]
    if not moods:
        moods = ["neutral"]
    fig = px.histogram(x=moods, labels={"x": "Mood", "y": "Sessions"})
    fig.update_layout(margin=dict(l=10, r=10, t=25, b=10))
    return fig


def waveform_figure(audio_path: str | Path):
    """Create a Matplotlib waveform figure for Streamlit."""
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def spectrogram_figure(audio_path: str | Path):
    """Create a Matplotlib spectrogram figure for Streamlit."""
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    stft = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

