# """ import librosa
# import numpy as np

# def detect_mood(file_path):
#     try:
#         print(f"[DEBUG] Processing file: {file_path}")

#         # Load audio
#         y, sr = librosa.load(file_path, duration=30)

#         if y is None or len(y) == 0:
#             print("[ERROR] Empty audio signal")
#             return "Unknown", 0.0

#         # ----------------------------
#         # 🎵 TEMPO
#         # ----------------------------
#         tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

#         # ----------------------------
#         # 🔊 ENERGY
#         # ----------------------------
#         energy = np.mean(librosa.feature.rms(y=y))

#         # ----------------------------
#         # 🎼 SPECTRAL
#         # ----------------------------
#         spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

#         print(f"[DEBUG] Tempo: {tempo}, Energy: {energy}, Centroid: {spectral_centroid}")

#         # ----------------------------
#         # 🧠 MOOD LOGIC
#         # ----------------------------
#         if tempo > 120 and energy > 0.02:
#             mood = "Energetic ⚡"
#         elif tempo < 80 and energy < 0.015:
#             mood = "Calm 🌿"
#         elif spectral_centroid < 2000:
#             mood = "Sad 😢"
#         else:
#             mood = "Happy 😊"

#         return mood, float(tempo)

#     except Exception as e:
#         print(f"[ERROR] Mood detection failed: {e}")
#         return "Unknown", 0.0 """

# import librosa
# import numpy as np

# def detect_mood(file_path):
#     try:
#         y, sr = librosa.load(file_path, duration=30)

#         if y is None or len(y) == 0:
#             return "Unknown", 0.0

#         # ----------------------------
#         # 🎧 FEATURES
#         # ----------------------------
#         rms = np.mean(librosa.feature.rms(y=y))  # energy
#         zcr = np.mean(librosa.feature.zero_crossing_rate(y))  # activity

#         # ----------------------------
#         # 🎯 TEMPO
#         # ----------------------------
#         try:
#             tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#         except:
#             tempo = 0

#         if tempo == 0 or np.isnan(tempo):
#             tempo = 70  # fallback

#         # ----------------------------
#         # 😊 IMPROVED MOOD LOGIC
#         # ----------------------------
#         if tempo > 120 and rms > 0.02:
#             mood = "Energetic 🔥"

#         elif tempo > 100:
#             mood = "Happy 😊"

#         elif tempo > 80:
#             mood = "Chill 😎"

#         elif rms < 0.01 and zcr < 0.05:
#             mood = "Sad 😢"

#         else:
#             mood = "Calm 😌"

#         return mood, float(tempo)

#     except Exception as e:
#         print(f"Mood detection error: {e}")
#         return "Unknown", 0.0

import librosa
import numpy as np

def detect_mood(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, duration=30)

        # ----------------------------
        # 🎵 TEMPO DETECTION (FIXED)
        # ----------------------------
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Fallback if tempo fails
        if tempo == 0 or np.isnan(tempo):
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

        # Ensure tempo is a scalar
        tempo = float(np.asarray(tempo).item())

        # ----------------------------
        # ⚡ ENERGY CALCULATION
        # ----------------------------
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)

        # ----------------------------
        # 😊 MOOD LOGIC (IMPROVED)
        # ----------------------------
        if tempo < 90 and energy < 0.05:
            mood = "Calm 😌"

        elif tempo < 120 and energy < 0.08:
            mood = "Happy 😊"

        elif tempo >= 120 and energy >= 0.08:
            mood = "Energetic 🔥"

        elif energy > 0.1:
            mood = "Aggressive ⚡"

        else:
            mood = "Relaxed 🎵"

        return mood, float(tempo)

    except Exception as e:
        print(f"Error in mood detection: {e}")
        return "Unknown", 0.0