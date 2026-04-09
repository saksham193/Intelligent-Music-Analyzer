import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data"
MODEL_PATH = "model/genre_model.pkl"
LABEL_PATH = "model/label_classes.npy"

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = np.hstack([
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features

# ----------------------------
# LOAD DATASET
# ----------------------------
X = []
y = []

print("📂 Loading dataset...")

for genre in os.listdir(DATA_PATH):
    genre_path = os.path.join(DATA_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
        file_path = os.path.join(genre_path, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(genre)
        except:
            continue

X = np.array(X)
y = np.array(y)

# ----------------------------
# ENCODE LABELS
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save labels
os.makedirs("model", exist_ok=True)
np.save(LABEL_PATH, le.classes_)

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ----------------------------
# MODEL TRAINING
# ----------------------------
print("\n🚀 Training model...")

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, MODEL_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")