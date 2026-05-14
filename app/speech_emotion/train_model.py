"""Train a Random Forest voice emotion classifier on RAVDESS audio."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.audio_processing.features import extract_features
from app.config.settings import (
    DEFAULT_MODEL_PATH,
    RAVDESS_DIR,
    SUPPORTED_EMOTIONS,
)


RAVDESS_EMOTION_CODES: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_emotion(file_path: str | Path) -> Optional[str]:
    """Parse emotion label from a RAVDESS filename."""

    name = Path(file_path).stem
    parts = name.split("-")

    if len(parts) < 3:
        return None

    emotion_code = parts[2]

    return RAVDESS_EMOTION_CODES.get(emotion_code)


def load_dataset(
    dataset_dir: str | Path = RAVDESS_DIR,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load RAVDESS files, extract features, and return X plus labels."""

    dataset_path = Path(dataset_dir)
    audio_files: Iterable[Path] = dataset_path.rglob("*.wav")

    rows: List[Dict] = []
    labels: List[str] = []

    for audio_file in audio_files:

        print(f"\nProcessing: {audio_file}")

        label = parse_ravdess_emotion(audio_file)

        if label not in SUPPORTED_EMOTIONS:
            print(f"Skipping unsupported emotion: {label}")
            continue

        try:
            features = extract_features(audio_file)

            rows.append(features)
            labels.append(label)

            print(f"SUCCESS: {label}")

        except Exception as exc:
            print(f"\nERROR processing {audio_file}")
            print(exc)

    print(f"\nTotal successful samples: {len(rows)}")

    if not rows:
        raise FileNotFoundError(
            f"No usable RAVDESS WAV files found in {dataset_path}. "
            "Place the dataset under datasets/ravdess first."
        )

    return pd.DataFrame(rows).fillna(0.0), labels


def train_model(
    dataset_dir: str | Path = RAVDESS_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train and save the emotion classifier."""

    print("\nLoading dataset...")

    X, labels = load_dataset(dataset_dir)

    print("\nEncoding labels...")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"Dataset size: {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nTraining Random Forest model...")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("\nEvaluating model...")

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    model_bundle = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_columns": list(X.columns),
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving trained model...")

    joblib.dump(model_bundle, output_path)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "model_path": output_path,
    }


if __name__ == "__main__":

    print("\nStarting emotion model training...\n")

    metrics = train_model()

    print("\n==============================")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("==============================\n")

    print(metrics["classification_report"])
    print(f"\nSaved model: {metrics['model_path']}")