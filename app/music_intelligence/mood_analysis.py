"""Explainable music mood analysis."""

from __future__ import annotations

from pathlib import Path

from app.music_intelligence.audio_energy import analyze_energy
from app.music_intelligence.harmonic_analysis import analyze_harmony
from app.music_intelligence.tempo_analysis import analyze_tempo
from app.music_intelligence.valence_estimator import estimate_valence


def _label_mood(tempo: float, energy: float, valence: float, brightness: float) -> str:
    if energy >= 0.68 and valence >= 0.58:
        return "Energetic upbeat"
    if energy >= 0.65 and valence < 0.45:
        return "Dark aggressive"
    if energy < 0.35 and valence >= 0.5:
        return "Relaxed ambient"
    if energy < 0.5 and valence < 0.45:
        return "Emotional melancholic"
    if tempo < 95 and brightness < 0.45:
        return "Calm acoustic"
    return "Balanced expressive"


def analyze_music_mood(audio_path: str | Path) -> dict[str, object]:
    """Return an explainable musical mood profile."""
    tempo = analyze_tempo(audio_path)
    energy = analyze_energy(audio_path)
    harmony = analyze_harmony(audio_path)
    valence = estimate_valence(
        tempo_bpm=tempo["tempo_bpm"],
        energy_score=energy["energy_score"],
        brightness=harmony["brightness"],
        harmonic_intensity=harmony["harmonic_intensity"],
    )
    mood = _label_mood(tempo["tempo_bpm"], energy["energy_score"], valence, harmony["brightness"])

    return {
        "mood": mood,
        "valence_score": valence,
        **tempo,
        **energy,
        **harmony,
    }

