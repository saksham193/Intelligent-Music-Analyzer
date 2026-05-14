"""Music genre prediction using the previous Random Forest genre model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from app.config.settings import GENRE_LABELS_PATH, GENRE_MODEL_PATH
from app.music_genre.feature_extractor import extract_genre_features, extract_music_summary


def load_genre_model(
    model_path: str | Path = GENRE_MODEL_PATH,
    labels_path: str | Path = GENRE_LABELS_PATH,
) -> tuple[object, np.ndarray]:
    """Load genre model and labels from trained_models."""
    model_file = Path(model_path)
    labels_file = Path(labels_path)
    if not model_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            "Genre model files are missing. Expected "
            f"{model_file.name} and {labels_file.name} in trained_models/."
        )
    return joblib.load(model_file), np.load(labels_file, allow_pickle=True)


def predict_genre(audio_path: str | Path) -> Dict[str, object]:
    """Predict genre, confidence, probabilities, and summary features."""
    model, labels = load_genre_model()
    features = extract_genre_features(audio_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label_index = int(prediction)
    genre = str(labels[label_index]) if label_index < len(labels) else str(prediction)
    confidence_scores = {
        str(label): float(prob)
        for label, prob in zip(labels, probabilities)
    }

    return {
        "genre": genre,
        "confidence": float(np.max(probabilities)),
        "confidence_scores": confidence_scores,
        "features": extract_music_summary(audio_path),
    }

