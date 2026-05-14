"""Lightweight musical valence estimation."""

from __future__ import annotations


def estimate_valence(tempo_bpm: float, energy_score: float, brightness: float, harmonic_intensity: float) -> float:
    """Estimate positivity/valence from normalized audio cues.

    This is an explainable heuristic, not a clinical emotion model. Brighter,
    faster, more energetic tracks usually feel more positive; slower and darker
    tracks often feel more melancholic.
    """
    tempo_norm = min(max(tempo_bpm / 180.0, 0.0), 1.0)
    score = (0.32 * tempo_norm) + (0.28 * energy_score) + (0.28 * brightness) + (0.12 * harmonic_intensity)
    return float(min(max(score, 0.0), 1.0))

