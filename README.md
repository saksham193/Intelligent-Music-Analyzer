# INTELLIGENT MUSIC ANALYZER

An AI-powered Streamlit application for emotion-aware music analysis, multilingual lyrics intelligence, and mood-based music recommendations.

Intelligent Music Analyzer combines voice emotion detection, text sentiment analysis, audio feature extraction, lyrics processing, and a modern chatbot-style interface to recommend playable music from the user's emotional context.

## Features

- Voice emotion analysis from microphone recordings and uploaded voice notes
- Text sentiment analysis for typed mood prompts
- Hybrid mood fusion across voice, text, lyrics, genre, tempo, energy, and valence
- Music genre prediction for uploaded songs
- Lyrics intelligence with metadata lookup, audio transcription, cleaning, language detection, and translation
- Hindi, English, and Hinglish-aware language routing
- Spotify/YouTube Music-style recommendation cards
- YouTube playable redirects for recommendations
- Like and skip feedback stored for personalization analytics
- Dark Streamlit dashboard with chatbot/WhatsApp-style prompt input
- Analytics dashboard for mood history, language trends, and recommendation feedback
- Optional stem separation support through Demucs

## Tech Stack

- Python
- Streamlit
- scikit-learn
- librosa
- faster-whisper / Whisper
- Last.fm API
- lyricsgenius
- deep-translator
- Plotly
- SQLite
- sounddevice / soundfile
- optional Demucs

## Screenshots

Add screenshots here after deployment:

- Voice Emotion Analysis dashboard
- Recommendation card panel
- Lyrics Intelligence workflow
- Music Genre Analyzer
- Analytics / History dashboard

## Installation

```powershell
git clone https://github.com/saksham193/Intelligent-Music-Analyzer.git
cd Intelligent-Music-Analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install FFmpeg and make sure `ffmpeg` is available in your system `PATH`. Whisper-based transcription and audio preprocessing depend on it.

Optional dependencies:

```powershell
pip install -r requirements-optional.txt
```

## Environment Variables

Create a local `.env` file when using external services:

```env
LASTFM_API_KEY=your_lastfm_api_key
GENIUS_ACCESS_TOKEN=your_genius_token
```

The app still works without these keys by using curated offline recommendation fallbacks and audio transcription fallbacks where possible.

## Run

```powershell
streamlit run main.py
```

The app starts with three main sections:

- Voice Emotion Analysis
- Music Genre Analyzer
- Analytics / History

## Folder Structure

```text
.
├── app/
│   ├── audio_processing/       Audio loading, recording, preprocessing, features
│   ├── config/                 Central paths and settings
│   ├── database/               SQLite history storage
│   ├── lyrics_engine/          Lyrics extraction, transcription, language routing
│   ├── music_genre/            Genre model, feature extraction, visualizers
│   ├── music_intelligence/     Mood, energy, valence, tempo, stems
│   ├── recommendation/         Last.fm recommendations, fallbacks, personalization
│   ├── speech_emotion/         Voice emotion, text sentiment, transcription, fusion
│   ├── ui/                     Streamlit dashboard, components, styles
│   ├── utils/                  Cache, logging, retry, validation helpers
│   └── visualization/          Charts and analytics views
├── assets/                     Static assets
├── datasets/                   Training datasets or dataset placeholders
├── documentation/              Architecture and project notes
├── trained_models/             Saved ML models used by the app
├── main.py                     Streamlit entry point
├── requirements.txt
├── requirements-optional.txt
└── Procfile
```

## AI Capabilities

### Voice Emotion Analysis

The app can analyze spoken prompts or uploaded voice notes. It extracts acoustic features, predicts a voice emotion, transcribes speech, detects language, and fuses the result with text sentiment.

### Text Emotion Analysis

Typed prompts are analyzed with multilingual sentiment logic and lightweight Hindi/Hinglish cues to produce a recommendation mood.

### Lyrics Intelligence

The lyrics engine combines metadata lookup, Genius fetching, audio transcription, language detection, cleaning, quality scoring, and translation.

### Music Intelligence

Uploaded music files can be analyzed for genre, tempo, energy, valence, emotional intensity, waveform, spectrograms, and optional stem separation.

## Supported Languages

- English
- Hindi
- Hindi-English / Hinglish

The app routes Whisper transcription and recommendation context using language detection and Hinglish hint matching.

## Recommendation System

The recommendation engine combines:

- detected mood and emotion
- text sentiment
- prompt language
- BPM / tempo context
- Last.fm tag recommendations
- similar-track lookup
- Hindi/Hinglish curated fallbacks
- static mood backups
- user Like / Skip feedback

Each recommendation card includes:

- cover artwork or a styled fallback icon
- song title and artist
- mood and language tags
- match score
- Like button
- Skip button
- Play button with a YouTube redirect

## Training

Voice emotion model:

```powershell
python -m app.speech_emotion.train_model
```

Music genre model:

```powershell
python -m app.music_genre.train_genre_model
```

## Deployment

This project is Streamlit-ready. For hosted deployment, configure environment variables and include the required model files or train them before launch.

The included `Procfile` supports platforms that run:

```text
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
```

## Future Scope

- Spotify Web API track playback and album artwork enrichment
- User accounts and persistent cloud personalization
- Larger multilingual mood datasets
- Better on-device speech emotion models
- Playlist generation and export
- Real-time voice mood tracking
- Richer recommendation explanations

## Notes

Generated caches, uploaded audio, logs, virtual environments, and local secrets are intentionally ignored by git.
