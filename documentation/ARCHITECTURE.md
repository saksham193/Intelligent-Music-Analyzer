# Architecture Notes

The application is split into independent AI modules so model purpose remains clear.

## Speech Emotion + Sentiment

Human speech inputs live in `app/speech_emotion`.

- Text prompts go directly to sentiment analysis.
- Microphone recordings and uploaded voice notes can be transcribed with Whisper.
- Voice emotion prediction uses the RAVDESS-trained speech model.
- Fusion combines speech emotion and text sentiment into the final mood.

This module should not analyze full songs or mixed music tracks.

## Music Genre Analysis

Song and music uploads live in `app/music_genre`.

- Genre prediction reuses the previous Random Forest model and label classes.
- Feature extraction follows the old chroma/MFCC mean and standard deviation layout.
- Visualizers show waveform, spectrogram, and mel spectrogram views.

This module should not feed music files into the speech emotion model.

## Lyrics Intelligence

Lyrics intelligence lives in `app/lyrics_engine` and only runs for uploaded music files. It extracts metadata with Mutagen, prepares Genius title/artist queries, fetches lyrics through LyricsGenius, caches lyrics locally, detects language with `langdetect`, and translates with `deep-translator`.

The Genius API is free to use with an access token, but it can miss songs when metadata is wrong or when lyrics are not available publicly. Caching reduces repeated API calls and makes demos faster.

## Music Mood Intelligence

Music mood analysis lives in `app/music_intelligence`. It estimates musical feeling from audio features:

- tempo and rhythm density
- RMS energy and emotional intensity
- brightness from spectral centroid
- harmonic/percussive balance
- valence as an explainable positivity score

This is separate from genre prediction because two songs in the same genre can feel very different.

## Hybrid Recommendation Layer

`app/recommendation/hybrid_recommender.py` is the final intelligence layer. It combines speech emotion, speech sentiment, lyrics sentiment, music mood, genre, tempo, and energy using weighted scoring. It also reports emotional dissonance when channels disagree.

## Optional Stem Separation

`app/music_intelligence/stem_separator.py` wraps Demucs. It is optional because neural source separation is slow and resource-heavy. When available, it can generate vocal/no-vocal stems for karaoke-style playback.

## Analytics And Personalization

Advanced analytics live in `app/visualization/advanced_dashboard.py` and `app/visualization/music_galaxy.py`. Personalization lives under `app/recommendation/personalization` and stores local Like/Skip feedback in SQLite.
