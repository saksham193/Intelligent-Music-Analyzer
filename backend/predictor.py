import numpy as np
import librosa
import joblib

# ----------------------------
# 📦 LOAD MODEL + LABELS
# ----------------------------
model = joblib.load("model/genre_model.pkl")
labels = np.load("model/label_classes.npy", allow_pickle=True)


# ----------------------------
# 🎧 FEATURE EXTRACTION
# ----------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        features = np.hstack([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ])

        return features.reshape(1, -1)

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


# ----------------------------
# 🎯 PREDICTION
# ----------------------------
def predict_genre(file_path):
    features = extract_features(file_path)

    if features is None:
        return "Error", 0.0, []

    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        confidence = np.max(probabilities)

        # 🔥 RETURN FULL PROBABILITY ARRAY (NEW)
        return labels[prediction], float(confidence), probabilities

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0, []