"""Train the music genre model from datasets/music_dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.config.settings import GENRE_LABELS_PATH, GENRE_MODEL_PATH, MUSIC_DATASET_DIR
from app.music_genre.feature_extractor import extract_genre_features


def train_genre_model(dataset_dir: str | Path = MUSIC_DATASET_DIR) -> dict[str, object]:
    """Train a Random Forest genre classifier.

    Expected dataset layout:
    datasets/music_dataset/<genre_name>/<audio files>
    """
    dataset_path = Path(dataset_dir)
    rows: List[np.ndarray] = []
    labels: List[str] = []

    for genre_dir in sorted(dataset_path.iterdir()) if dataset_path.exists() else []:
        if not genre_dir.is_dir():
            continue
        for audio_file in genre_dir.rglob("*"):
            if audio_file.suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
                continue
            try:
                rows.append(extract_genre_features(audio_file).ravel())
                labels.append(genre_dir.name)
            except Exception as exc:
                print(f"Skipping {audio_file}: {exc}")

    if not rows:
        raise FileNotFoundError(
            f"No music files found under {dataset_path}. Use genre subfolders."
        )

    X = np.array(rows)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), target_names=encoder.classes_)

    GENRE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, GENRE_MODEL_PATH)
    np.save(GENRE_LABELS_PATH, encoder.classes_)
    return {"model_path": GENRE_MODEL_PATH, "labels_path": GENRE_LABELS_PATH, "report": report}


if __name__ == "__main__":
    metrics = train_genre_model()
    print(metrics["report"])
    print(f"Saved model: {metrics['model_path']}")
