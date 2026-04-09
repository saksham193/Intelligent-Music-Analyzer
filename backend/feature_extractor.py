""" import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

# ----------------------------
# 🎧 AUDIO → CHROMA IMAGE
# ----------------------------
def audio_to_chroma_image(file_path, save_path="temp.png"):
    try:
        # Load audio
        y, sr = librosa.load(file_path, duration=30)

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Normalize (important)
        chroma = (chroma - np.min(chroma)) / (np.max(chroma) - np.min(chroma) + 1e-6)

        # Create heatmap image (same style as training)
        plt.figure(figsize=(4, 3))
        sns.heatmap(chroma, cmap="viridis", cbar=False)
        plt.axis("off")

        # Save image
        os.makedirs("temp", exist_ok=True)
        full_path = os.path.join("temp", save_path)
        plt.savefig(full_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return full_path

    except Exception as e:
        print(f"Error in audio_to_chroma_image: {e}")
        return None


# ----------------------------
# 🧠 IMAGE → MODEL INPUT
# ----------------------------
def prepare_image_for_model(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))   # match training
        img = img / 255.0                  # normalize
        img = np.expand_dims(img, axis=0)  # shape (1, 128, 128, 3)

        return img

    except Exception as e:
        print(f"Error in prepare_image_for_model: {e}")
        return None


# ----------------------------
# 🔁 FULL PIPELINE
# ----------------------------
def extract_features(file_path):
    image_path = audio_to_chroma_image(file_path)

    if image_path is None:
        return None

    return prepare_image_for_model(image_path) """
import librosa
import numpy as np

def extract_features(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, duration=30)

        # ----------------------------
        # 🎵 FEATURE EXTRACTION
        # ----------------------------

        # MFCC (40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral Contrast (7)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_mean = np.mean(spec_contrast, axis=1)

        # Combine → total = 40 + 12 + 7 = 59
        features = np.hstack((mfcc_mean, chroma_mean, spec_mean))

        # Pad to 130 (VERY IMPORTANT)
        if len(features) < 130:
            features = np.pad(features, (0, 130 - len(features)))
        else:
            features = features[:130]

        return features.reshape(1, -1)

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None