"""Central settings for the Emotion Music AI project.

All paths are derived from the project root so modules can be called from
Streamlit, tests, or standalone scripts without hard-coded absolute paths.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAVDESS_DIR = DATASETS_DIR / "ravdess"
MUSIC_DATASET_DIR = DATASETS_DIR / "music_dataset"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
CACHE_DIR = PROJECT_ROOT / "cache"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATABASE_PATH = PROJECT_ROOT / "cache" / "emotion_music_ai.sqlite3"

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_RECORD_SECONDS = 5
DEFAULT_MODEL_PATH = TRAINED_MODELS_DIR / "voice_emotion_random_forest.joblib"
VOICE_EMOTION_MODEL_PATH = DEFAULT_MODEL_PATH
GENRE_MODEL_PATH = TRAINED_MODELS_DIR / "genre_model.pkl"
GENRE_LABELS_PATH = TRAINED_MODELS_DIR / "genre_label_classes.npy"
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")
LYRICS_CACHE_DIR = APP_DIR / "lyrics_engine" / "cache"
STEM_CACHE_DIR = CACHE_DIR / "stems"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "75"))
CACHE_RETENTION_HOURS = int(os.getenv("CACHE_RETENTION_HOURS", "24"))

SUPPORTED_EMOTIONS = ("happy", "sad", "calm", "angry", "fearful", "neutral")
SUPPORTED_AUDIO_TYPES = ("wav", "mp3", "m4a", "ogg", "flac")
SUPPORTED_LYRIC_LANGUAGES = ("en", "hi")

HYBRID_FUSION_WEIGHTS = {
    "speech_emotion": 0.35,
    "speech_sentiment": 0.2,
    "lyrics_sentiment": 0.25,
    "music_mood": 0.2,
}
