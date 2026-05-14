"""Voice emotion prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from app.audio_processing.features import extract_features
from app.config.settings import DEFAULT_MODEL_PATH


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict[str, object]:
    """Load the saved Random Forest model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it with app/emotion_detection/train_model.py."
        )
    return joblib.load(path)


def predict_emotion(audio_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict[str, object]:
    """Predict voice emotion and class confidence scores for one audio file."""
    bundle = load_model(model_path)
    features = extract_features(audio_path)
    columns = bundle["feature_columns"]
    X = pd.DataFrame([features]).reindex(columns=columns, fill_value=0.0)

    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    encoded_prediction = pipeline.predict(X)[0]
    emotion = str(label_encoder.inverse_transform([encoded_prediction])[0])

    probabilities = pipeline.predict_proba(X)[0]
    confidence_scores = {
        str(label): float(prob)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }

    return {
        "emotion": emotion,
        "confidence": float(max(confidence_scores.values())),
        "confidence_scores": confidence_scores,
        "features": features,
    }

