import requests
import streamlit as st  # 👈 ADD THIS

API_KEY = "a9e85a9519d88ad847dbb325aeade14e"

# ----------------------------
# 🔁 FALLBACK (for safety)
# ----------------------------
fallback = {
    "classical": [
        "Fur Elise - Beethoven",
        "Moonlight Sonata - Beethoven",
        "The Four Seasons - Vivaldi"
    ],
    "rock": [
        "Bohemian Rhapsody - Queen",
        "Smells Like Teen Spirit - Nirvana",
        "Hotel California - Eagles"
    ],
    "pop": [
        "Blinding Lights - The Weeknd",
        "Shape of You - Ed Sheeran",
        "Levitating - Dua Lipa"
    ],
    "jazz": [
        "Take Five - Dave Brubeck",
        "So What - Miles Davis",
        "Autumn Leaves - Bill Evans"
    ]
}

# ----------------------------
# 🌐 API CALL
# ----------------------------
def recommend_songs(genre):
    url = "http://ws.audioscrobbler.com/2.0/"

    params = {
        "method": "tag.gettoptracks",
        "tag": genre,
        "api_key": API_KEY,
        "format": "json",
        "limit": 5
    }

    try:
        # 🔍 DEBUG: show API call
        print(f"[DEBUG] Calling API with genre: {genre}")

        response = requests.get(url, params=params, timeout=5)

        # 🔍 DEBUG: status code
        print(f"[DEBUG] Status Code: {response.status_code}")

        data = response.json()

        # 🔍 DEBUG: show part of response
        print("[DEBUG] API response received")

        # 👇 SHOW INSIDE STREAMLIT UI ALSO
        # st.write("🔍 Debug API Response:", data)

        songs = []

        # 🔥 SAFE ACCESS (avoid crash)
        tracks = data.get("tracks", {}).get("track", [])

        for track in tracks:
            name = track.get("name", "Unknown")
            artist = track.get("artist", {}).get("name", "Unknown Artist")

            songs.append({
                "name": name,
                "artist": artist,
                "url": track.get("url", "#")
            })

        # If API returns empty
        if not songs:
            print("[DEBUG] No songs from API → using fallback")
            st.warning("⚠️ API returned no songs, using fallback")

            return fallback.get(genre.lower(), ["No recommendations available"])

        return songs

    except Exception as e:
        print("[ERROR] API Error:", e)

        # 👇 SHOW ERROR IN UI
        st.error(f"API Error: {e}")

        return fallback.get(genre.lower(), ["No recommendations available"])