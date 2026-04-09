import random

# Dummy dataset (you can improve later)
music_db = {
    "classical": ["Moonlight Sonata", "Four Seasons", "Canon in D"],
    "rock": ["Bohemian Rhapsody", "Stairway to Heaven", "Hotel California"],
    "pop": ["Blinding Lights", "Shape of You", "Levitating"],
    "jazz": ["Take Five", "So What", "Blue in Green"],
    "hiphop": ["Lose Yourself", "Sicko Mode", "God's Plan"]
}

def recommend_songs(genre):
    if genre in music_db:
        return random.sample(music_db[genre], 2)
    else:
        return ["No recommendations available"]