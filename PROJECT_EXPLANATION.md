# Project Explanation

## Architecture

The project is split into small Python modules so each responsibility can grow independently. Streamlit handles the frontend, while the backend logic is organized under `app/`.

## Folder Purpose

- `app/ui`: Renders the Streamlit dashboard, input controls, cards, charts, and session state.
- `app/audio_processing`: Records microphone audio and extracts acoustic features from files.
- `app/emotion_detection`: Trains the voice emotion classifier, predicts emotion, analyzes text sentiment, and fuses both signals.
- `app/recommendation`: Maps final moods to music tags and fetches Last.fm recommendations.
- `app/speech_to_text`: Uses a local Whisper model for Hindi/English speech transcription.
- `app/visualization`: Builds waveform, spectrogram, BPM, confidence, and mood analytics charts.
- `app/database`: Stores user history in SQLite.
- `app/models`: Placeholder for future model wrappers or metadata.
- `app/utils`: Placeholder for shared helper functions.
- `app/config`: Central paths, constants, and environment-based settings.
- `datasets/ravdess`: RAVDESS voice emotion dataset location.
- `datasets/music_dataset`: Future local music metadata or CSV files.
- `trained_models`: Saved machine learning model artifacts.
- `cache`: Temporary recordings, uploaded audio, and the SQLite database.
- `assets`: Images, icons, and static UI assets.
- `tests`: Future unit and integration tests.

## File Purpose

- `main.py`: Starts the Streamlit dashboard.
- `requirements.txt`: Lists free/open-source dependencies.
- `README.md`: Installation, training, running steps, and common fixes.
- `app/config/settings.py`: Defines reusable project paths, defaults, supported emotions, and `LASTFM_API_KEY`.
- `app/audio_processing/recorder.py`: Records voice with `sounddevice`, normalizes safely, and saves WAV files using `scipy.io.wavfile`.
- `app/audio_processing/features.py`: Extracts MFCC, chroma, RMS, spectral contrast, zero crossing rate, pitch, and BPM using Librosa.
- `app/emotion_detection/train_model.py`: Loads RAVDESS, extracts features, encodes labels, trains Random Forest, evaluates accuracy, and saves the model with joblib.
- `app/emotion_detection/predict.py`: Loads the saved model and predicts voice emotion with confidence scores.
- `app/emotion_detection/text_sentiment.py`: Uses HuggingFace sentiment analysis and combines voice emotion with text sentiment.
- `app/speech_to_text/transcriber.py`: Runs local Whisper transcription with automatic or forced Hindi/English language handling.
- `app/recommendation/recommender.py`: Calls Last.fm when an API key exists and returns offline fallback recommendations otherwise.
- `app/database/db.py`: Creates SQLite tables, inserts analysis history, and fetches recent sessions.
- `app/visualization/charts.py`: Builds explainability charts with Plotly, Matplotlib, and Librosa display.
- `app/ui/dashboard.py`: Connects all modules into the user-facing workflow.
- `__init__.py` files: Mark folders as Python packages for clean imports.

## Why These Tools

- `sounddevice` and `scipy`: Free local microphone recording and WAV saving.
- `librosa`: Reliable open-source audio feature extraction for speech/music signals.
- `scikit-learn RandomForestClassifier`: Stable baseline model, works well with tabular acoustic features, and avoids deep-learning complexity at the initial stage.
- `joblib`: Standard model persistence for scikit-learn pipelines.
- `Whisper local model`: Strong free speech recognition with Hindi and English support after local model download.
- `HuggingFace transformers`: Gives pretrained multilingual sentiment analysis without building a text model from scratch.
- `Last.fm`: Free recommendation API suitable for academic projects. The system still works with fallback songs when the key is missing.
- `SQLite`: Local, zero-server database that is ideal for demos and can later be migrated to PostgreSQL/MySQL.
- `Streamlit`: Fast Python frontend for ML projects, with simple state management and interactive widgets.
- `Plotly`, `Matplotlib`, `Librosa display`: Explain predictions through confidence, BPM, waveform, and spectrogram views.

## Audio Feature Meaning

- MFCC: Captures speech tone and vocal texture, useful for emotion recognition.
- Chroma: Captures pitch-class energy and helps with tonal mood cues.
- RMS energy: Measures loudness, often higher in angry or excited speech.
- Spectral contrast: Measures brightness and frequency distribution.
- Zero crossing rate: Indicates noisiness or sharpness in the signal.
- Pitch: Tracks speaking pitch, which often changes with emotion.
- Tempo/BPM: Useful for uploaded music/audio and high-level rhythm analytics.

## Emotion Model Notes

Random Forest is used first because it trains quickly, is explainable enough for a student project, handles mixed feature scales well with a pipeline, and needs less data than CNNs. Deep learning is intentionally avoided initially because it needs more data, more compute, longer tuning, and stronger overfitting controls.

Overfitting is reduced with train/test splitting, class balancing, limited tree depth, minimum leaf samples, and evaluation on held-out data. Predictions include confidence scores using class probabilities.

## Offline Behavior

Voice recording, feature extraction, trained voice prediction, SQLite, charts, and local Whisper transcription run offline. Whisper and HuggingFace models may need a first-time download, then can run locally from cache. Last.fm recommendations require internet and an API key, but fallback recommendations keep the app usable without them.

