"""Last.fm-based music recommendation engine with offline fallbacks."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus
from typing import Dict, List

import requests

from app.config.settings import LASTFM_API_KEY


LOGGER = logging.getLogger(__name__)

EMOTION_GENRE_MAP: Dict[str, str] = {
    "sad": "lo-fi chill",
    "emotionally tired": "lo-fi chill",
    "happy": "pop",
    "energetic": "edm workout",
    "calm": "instrumental",
    "angry": "rock metal",
    "fearful": "ambient",
    "anxious": "calming piano",
    "neutral": "acoustic",
}

LANGUAGE_TAG_MAP: Dict[str, Dict[str, str]] = {
    "hindi": {
        "sad": "hindi sad",
        "emotionally tired": "bollywood sad",
        "happy": "bollywood dance",
        "energetic": "bollywood dance",
        "calm": "hindi acoustic",
        "angry": "bollywood rock",
        "fearful": "hindi chill",
        "anxious": "hindi chill",
        "neutral": "bollywood",
    },
    "hinglish": {
        "sad": "bollywood sad",
        "emotionally tired": "bollywood sad",
        "happy": "indian pop",
        "energetic": "punjabi pop",
        "calm": "bollywood acoustic",
        "angry": "indian rock",
        "fearful": "bollywood chill",
        "anxious": "bollywood chill",
        "neutral": "indian pop",
    },
    "english": EMOTION_GENRE_MAP,
}

PLACEHOLDER_COVER_URL = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='500' height='500' viewBox='0 0 500 500'%3E"
    "%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E"
    "%3Cstop offset='0' stop-color='%231ed760'/%3E%3Cstop offset='1' stop-color='%235865f2'/%3E"
    "%3C/linearGradient%3E%3C/defs%3E%3Crect width='500' height='500' rx='64' fill='%23141620'/%3E"
    "%3Ccircle cx='250' cy='250' r='132' fill='url(%23g)' opacity='.9'/%3E"
    "%3Ccircle cx='250' cy='250' r='54' fill='%23141620'/%3E"
    "%3Cpath d='M320 214v88c0 25-22 45-50 45s-50-20-50-45 22-45 50-45c9 0 17 2 24 5v-86h26z' fill='white' opacity='.95'/%3E"
    "%3C/svg%3E"
)

FALLBACK_RECOMMENDATIONS: Dict[str, List[Dict[str, str]]] = {
    "sad": [
        {"title": "The Night We Met", "artist": "Lord Huron"},
        {"title": "Let Her Go", "artist": "Passenger"},
        {"title": "Someone Like You", "artist": "Adele"},
        {"title": "Fix You", "artist": "Coldplay"},
        {"title": "Skinny Love", "artist": "Bon Iver"},
        {"title": "All I Want", "artist": "Kodaline"},
        {"title": "Breathe Me", "artist": "Sia"},
        {"title": "Weightless", "artist": "Marconi Union"},
    ],
    "happy": [
        {"title": "Good Life", "artist": "OneRepublic"},
        {"title": "Walking on Sunshine", "artist": "Katrina and the Waves"},
        {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake"},
        {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars"},
        {"title": "Happy", "artist": "Pharrell Williams"},
        {"title": "Good as Hell", "artist": "Lizzo"},
        {"title": "Shut Up and Dance", "artist": "WALK THE MOON"},
        {"title": "On Top of the World", "artist": "Imagine Dragons"},
    ],
    "energetic": [
        {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars"},
        {"title": "Don't Start Now", "artist": "Dua Lipa"},
        {"title": "Blinding Lights", "artist": "The Weeknd"},
        {"title": "Titanium", "artist": "David Guetta ft. Sia"},
        {"title": "Stronger", "artist": "Kanye West"},
        {"title": "Shut Up and Dance", "artist": "WALK THE MOON"},
        {"title": "Can't Hold Us", "artist": "Macklemore and Ryan Lewis"},
        {"title": "Levels", "artist": "Avicii"},
    ],
    "calm": [
        {"title": "River Flows in You", "artist": "Yiruma"},
        {"title": "Nuvole Bianche", "artist": "Ludovico Einaudi"},
        {"title": "Bloom", "artist": "The Paper Kites"},
        {"title": "Holocene", "artist": "Bon Iver"},
        {"title": "Sunset Lover", "artist": "Petit Biscuit"},
        {"title": "Experience", "artist": "Ludovico Einaudi"},
        {"title": "Comptine d'un autre ete", "artist": "Yann Tiersen"},
        {"title": "Weightless", "artist": "Marconi Union"},
    ],
    "angry": [
        {"title": "Numb", "artist": "Linkin Park"},
        {"title": "Believer", "artist": "Imagine Dragons"},
        {"title": "In The End", "artist": "Linkin Park"},
        {"title": "Killing In The Name", "artist": "Rage Against The Machine"},
        {"title": "Seven Nation Army", "artist": "The White Stripes"},
        {"title": "Stronger", "artist": "Kanye West"},
        {"title": "Till I Collapse", "artist": "Eminem"},
        {"title": "Radioactive", "artist": "Imagine Dragons"},
    ],
    "anxious": [
        {"title": "Weightless", "artist": "Marconi Union"},
        {"title": "Clair de Lune", "artist": "Claude Debussy"},
        {"title": "Sunset Lover", "artist": "Petit Biscuit"},
        {"title": "Night Owl", "artist": "Galimatias"},
        {"title": "A Walk", "artist": "Tycho"},
        {"title": "Intro", "artist": "The xx"},
        {"title": "Gymnopedie No. 1", "artist": "Erik Satie"},
        {"title": "Spiegel im Spiegel", "artist": "Arvo Part"},
    ],
    "fearful": [
        {"title": "Weightless", "artist": "Marconi Union"},
        {"title": "Clair de Lune", "artist": "Claude Debussy"},
        {"title": "A Walk", "artist": "Tycho"},
        {"title": "Holocene", "artist": "Bon Iver"},
        {"title": "Nuvole Bianche", "artist": "Ludovico Einaudi"},
        {"title": "River Flows in You", "artist": "Yiruma"},
        {"title": "Bloom", "artist": "The Paper Kites"},
        {"title": "Experience", "artist": "Ludovico Einaudi"},
    ],
    "neutral": [
        {"title": "Yellow", "artist": "Coldplay"},
        {"title": "Photograph", "artist": "Ed Sheeran"},
        {"title": "Counting Stars", "artist": "OneRepublic"},
        {"title": "The Scientist", "artist": "Coldplay"},
        {"title": "Viva La Vida", "artist": "Coldplay"},
        {"title": "Perfect", "artist": "Ed Sheeran"},
        {"title": "Riptide", "artist": "Vance Joy"},
        {"title": "Ho Hey", "artist": "The Lumineers"},
    ],
}

HINDI_FALLBACK_RECOMMENDATIONS: Dict[str, List[Dict[str, str]]] = {
    "sad": [
        {"title": "Channa Mereya", "artist": "Arijit Singh"},
        {"title": "Agar Tum Saath Ho", "artist": "Alka Yagnik and Arijit Singh"},
        {"title": "Tujhe Kitna Chahne Lage", "artist": "Arijit Singh"},
        {"title": "Phir Bhi Tumko Chaahunga", "artist": "Arijit Singh"},
        {"title": "Bekhayali", "artist": "Sachet Tandon"},
        {"title": "Kaise Hua", "artist": "Vishal Mishra"},
        {"title": "Humnava Mere", "artist": "Jubin Nautiyal"},
        {"title": "Tera Yaar Hoon Main", "artist": "Arijit Singh"},
    ],
    "happy": [
        {"title": "Nashe Si Chadh Gayi", "artist": "Arijit Singh"},
        {"title": "Gallan Goodiyaan", "artist": "Yashita Sharma and Manish Kumar Tipu"},
        {"title": "Badtameez Dil", "artist": "Benny Dayal"},
        {"title": "Kar Gayi Chull", "artist": "Badshah and Amaal Mallik"},
        {"title": "London Thumakda", "artist": "Labh Janjua and Neha Kakkar"},
        {"title": "Kala Chashma", "artist": "Amar Arshi and Badshah"},
        {"title": "Ilahi", "artist": "Arijit Singh"},
        {"title": "Ude Dil Befikre", "artist": "Benny Dayal"},
    ],
    "energetic": [
        {"title": "Malhari", "artist": "Vishal Dadlani"},
        {"title": "Apna Time Aayega", "artist": "Ranveer Singh"},
        {"title": "Zinda", "artist": "Siddharth Mahadevan"},
        {"title": "Kar Gayi Chull", "artist": "Badshah and Amaal Mallik"},
        {"title": "Kala Chashma", "artist": "Amar Arshi and Badshah"},
        {"title": "Badtameez Dil", "artist": "Benny Dayal"},
        {"title": "Nashe Si Chadh Gayi", "artist": "Arijit Singh"},
        {"title": "Jai Jai Shivshankar", "artist": "Vishal Dadlani and Benny Dayal"},
    ],
    "calm": [
        {"title": "Raabta", "artist": "Arijit Singh"},
        {"title": "Shayad", "artist": "Arijit Singh"},
        {"title": "Tera Ban Jaunga", "artist": "Akhil Sachdeva and Tulsi Kumar"},
        {"title": "Ranjha", "artist": "B Praak and Jasleen Royal"},
        {"title": "Aabaad Barbaad", "artist": "Arijit Singh"},
        {"title": "Pehla Pyaar", "artist": "Armaan Malik"},
        {"title": "Mann Bharryaa", "artist": "B Praak"},
        {"title": "Tum Hi Ho", "artist": "Arijit Singh"},
    ],
    "angry": [
        {"title": "Zinda", "artist": "Siddharth Mahadevan"},
        {"title": "Sultan Title Track", "artist": "Sukhwinder Singh"},
        {"title": "Malhari", "artist": "Vishal Dadlani"},
        {"title": "Apna Time Aayega", "artist": "Ranveer Singh"},
        {"title": "Brothers Anthem", "artist": "Vishal Dadlani"},
        {"title": "Kar Har Maidaan Fateh", "artist": "Sukhwinder Singh and Shreya Ghoshal"},
        {"title": "Jee Karda", "artist": "Divya Kumar"},
        {"title": "Dangal", "artist": "Daler Mehndi"},
    ],
    "anxious": [
        {"title": "Kun Faya Kun", "artist": "A.R. Rahman, Javed Ali and Mohit Chauhan"},
        {"title": "Iktara", "artist": "Kavita Seth"},
        {"title": "Safarnama", "artist": "Lucky Ali"},
        {"title": "Aao Milo Chalo", "artist": "Shaan and Ustad Sultan Khan"},
        {"title": "Khaabon Ke Parinday", "artist": "Alyssa Mendonsa and Mohit Chauhan"},
        {"title": "Phir Se Ud Chala", "artist": "Mohit Chauhan"},
        {"title": "Ilahi", "artist": "Arijit Singh"},
        {"title": "Agar Tum Saath Ho", "artist": "Alka Yagnik and Arijit Singh"},
    ],
    "fearful": [
        {"title": "Kun Faya Kun", "artist": "A.R. Rahman, Javed Ali and Mohit Chauhan"},
        {"title": "Iktara", "artist": "Kavita Seth"},
        {"title": "Safarnama", "artist": "Lucky Ali"},
        {"title": "Aao Milo Chalo", "artist": "Shaan and Ustad Sultan Khan"},
        {"title": "Phir Se Ud Chala", "artist": "Mohit Chauhan"},
        {"title": "Ranjha", "artist": "B Praak and Jasleen Royal"},
        {"title": "Shayad", "artist": "Arijit Singh"},
        {"title": "Raabta", "artist": "Arijit Singh"},
    ],
    "neutral": [
        {"title": "Kesariya", "artist": "Arijit Singh"},
        {"title": "Tum Se Hi", "artist": "Mohit Chauhan"},
        {"title": "Shayad", "artist": "Arijit Singh"},
        {"title": "Raabta", "artist": "Arijit Singh"},
        {"title": "Kabira", "artist": "Tochi Raina and Rekha Bhardwaj"},
        {"title": "Tera Ban Jaunga", "artist": "Akhil Sachdeva and Tulsi Kumar"},
        {"title": "Pee Loon", "artist": "Mohit Chauhan"},
        {"title": "Hawayein", "artist": "Arijit Singh"},
    ],
}


def get_genre_for_emotion(emotion: str) -> str:
    """Return a Last.fm tag/search genre for the detected emotion."""
    return EMOTION_GENRE_MAP.get((emotion or "neutral").lower(), "acoustic")


def _normalized_mood(emotion: str, sentiment: str = "", bpm: float = 0.0, valence: float | None = None) -> str:
    mood = (emotion or "neutral").strip().lower()
    sentiment = (sentiment or "").strip().lower()

    if mood in {"fear", "afraid", "scared"}:
        mood = "fearful"
    if mood in {"stress", "stressed", "anxiety"}:
        mood = "anxious"
    if mood not in FALLBACK_RECOMMENDATIONS:
        mood = "sad" if sentiment == "negative" else "happy" if sentiment == "positive" else "neutral"

    if mood == "neutral" and valence is not None:
        mood = "happy" if valence >= 0.66 else "sad" if valence <= 0.34 else mood
    if mood in {"fearful", "anxious"}:
        return "anxious"
    if bpm >= 135 and mood in {"happy", "neutral"}:
        return "energetic"
    return mood


def _normalized_language(language: str) -> str:
    label = (language or "english").strip().lower()
    if any(token in label for token in ("hinglish", "mixed", "hindi-english")):
        return "hinglish"
    if label in {"hi", "hin"} or "hindi" in label:
        return "hindi"
    return "english"


def _youtube_url(title: str, artist: str) -> str:
    query = f"{title} {artist} official audio"
    return "https://www.youtube.com/results?search_query=" + quote_plus(query)


def _best_image(images: list[dict[str, str]]) -> str:
    for image in reversed(images or []):
        image_url = image.get("#text", "").strip()
        if image_url:
            return image_url
    LOGGER.debug("Missing Last.fm cover image; using local placeholder cover.")
    return PLACEHOLDER_COVER_URL


def _format_recommendation(
    title: str,
    artist: str,
    mood: str,
    language: str,
    cover_url: str = "",
    source: str = "lastfm",
) -> Dict[str, str]:
    cover = cover_url or PLACEHOLDER_COVER_URL
    return {
        "title": title or "Unknown title",
        "artist": artist or "Unknown artist",
        "url": _youtube_url(title or "Unknown title", artist or "Unknown artist"),
        "cover_url": cover,
        "image": cover,
        "mood": mood,
        "language": language,
        "source": source,
    }


def _fallback_for(emotion: str, language: str = "english", limit: int = 8) -> List[Dict[str, str]]:
    mood = _normalized_mood(emotion)
    language_key = _normalized_language(language)
    catalog = HINDI_FALLBACK_RECOMMENDATIONS if language_key in {"hindi", "hinglish"} else FALLBACK_RECOMMENDATIONS
    tracks = catalog.get(mood, catalog["neutral"])
    LOGGER.debug(
        "Using mood fallback recommendations: mood=%s language=%s requested=%s available=%s",
        mood,
        language_key,
        limit,
        len(tracks),
    )
    recommendations = [
        _format_recommendation(
            str(track.get("title", "")),
            str(track.get("artist", "")),
            mood,
            language_key,
            str(track.get("cover_url") or track.get("image") or ""),
        )
        for track in tracks[: max(limit, 5)]
    ]
    for song in recommendations:
        song["source_detail"] = "mood_fallback"
    return recommendations


def _tag_for_context(mood: str, language: str) -> str:
    language_key = _normalized_language(language)
    language_tags = LANGUAGE_TAG_MAP.get(language_key, EMOTION_GENRE_MAP)
    return language_tags.get(mood, language_tags.get("neutral", get_genre_for_emotion(mood)))


def _lastfm_tracks_from_tag(api_key: str, tag: str, limit: int) -> list[dict[str, object]]:
    params = {
        "method": "tag.gettoptracks",
        "tag": tag,
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    response = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=8)
    response.raise_for_status()
    return response.json().get("tracks", {}).get("track", [])


def _lastfm_similar_tracks(api_key: str, seeds: list[dict[str, str]], limit: int) -> list[dict[str, object]]:
    similar: list[dict[str, object]] = []
    for seed in seeds[:2]:
        params = {
            "method": "track.getsimilar",
            "track": seed.get("title", ""),
            "artist": seed.get("artist", ""),
            "api_key": api_key,
            "format": "json",
            "limit": limit,
        }
        try:
            response = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=8)
            response.raise_for_status()
            similar.extend(response.json().get("similartracks", {}).get("track", []))
        except Exception as exc:
            LOGGER.debug("Last.fm similar-track lookup failed for %s: %s", seed.get("title"), exc)
    return similar


def _append_lastfm_tracks(
    recommendations: list[dict[str, str]],
    tracks: list[dict[str, object]],
    seen: set[tuple[str, str]],
    mood: str,
    language_key: str,
    source_detail: str,
) -> None:
    for track in tracks:
        artist_data = track.get("artist", {})
        artist = artist_data.get("name", "") if isinstance(artist_data, dict) else str(artist_data)
        title = str(track.get("name", "Unknown title"))
        key = (title.lower(), artist.lower())
        if key in seen:
            continue
        seen.add(key)
        song = _format_recommendation(
            title,
            artist,
            mood,
            language_key,
            _best_image(track.get("image", [])),
        )
        song["source_detail"] = source_detail
        recommendations.append(song)


def recommend_songs(
    emotion: str,
    limit: int = 8,
    api_key: str = LASTFM_API_KEY,
    language: str = "English",
    sentiment: str = "",
    bpm: float = 0.0,
    valence: float | None = None,
) -> List[Dict[str, str]]:
    """Fetch Last.fm tracks for an emotion, falling back when API access is absent."""
    limit = max(limit, 5)
    mood = _normalized_mood(emotion, sentiment=sentiment, bpm=bpm, valence=valence)
    language_key = _normalized_language(language)
    LOGGER.debug(
        "Recommendation request: mood=%s language=%s sentiment=%s bpm=%s limit=%s api=%s",
        mood,
        language_key,
        sentiment,
        bpm,
        limit,
        bool(api_key),
    )

    if not api_key:
        fallback = _fallback_for(mood, language_key, limit)
        LOGGER.debug("Recommendation result: count=%s source=mood_fallback reason=no_api_key", len(fallback))
        return fallback

    tag = _tag_for_context(mood, language_key)
    tracks: list[dict[str, object]] = []
    try:
        tracks = _lastfm_tracks_from_tag(api_key, tag, limit)
        LOGGER.debug("Last.fm tag recommendations received: tag=%s count=%s", tag, len(tracks))
    except Exception as exc:
        LOGGER.debug("Last.fm tag recommendation failed: tag=%s error=%s", tag, exc)

    recommendations: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    _append_lastfm_tracks(recommendations, tracks, seen, mood, language_key, "lastfm_tag")

    if len(recommendations) < limit:
        similar_tracks = _lastfm_similar_tracks(api_key, _fallback_for(mood, language_key, 2), limit)
        LOGGER.debug("Last.fm similar recommendations received: count=%s", len(similar_tracks))
        _append_lastfm_tracks(recommendations, similar_tracks, seen, mood, language_key, "lastfm_similar")

    if len(recommendations) < limit:
        LOGGER.debug(
            "Filling recommendation pool with fallback: current=%s needed=%s",
            len(recommendations),
            limit,
        )
        existing = {(song["title"].lower(), song["artist"].lower()) for song in recommendations}
        for song in _fallback_for(mood, language_key, limit):
            key = (song["title"].lower(), song["artist"].lower())
            if key not in existing:
                recommendations.append(song)
            if len(recommendations) >= limit:
                break

    result = recommendations[:limit]
    LOGGER.debug(
        "Recommendation result: count=%s sources=%s fallback_used=%s missing_cover=%s",
        len(result),
        sorted({song.get("source_detail", song.get("source", "unknown")) for song in result}),
        len(tracks) < limit or len(result) < limit,
        sum(1 for song in result if not song.get("cover_url")),
    )
    return result
