import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 🎧 Waveform
# ----------------------------
def plot_waveform(file_path):
    y, sr = librosa.load(file_path)

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")

    return fig


# ----------------------------
# 🎵 Spectrogram
# ----------------------------
def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path)

    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax)

    return fig


# ----------------------------
# 🔥 MEL SPECTROGRAM (NEW)
# ----------------------------
def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax)

    return fig


# ----------------------------
# 🔥 CHROMA FEATURES (NEW)
# ----------------------------
def plot_chroma(file_path):
    y, sr = librosa.load(file_path)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set_title("Chroma Features")
    fig.colorbar(img, ax=ax)

    return fig


# ----------------------------
# 🎯 CONFIDENCE GRAPH (NEW)
# ----------------------------
def plot_confidence(predictions, labels):
    fig, ax = plt.subplots()

    ax.bar(labels, predictions)
    ax.set_title("Genre Confidence")
    ax.set_xlabel("Genres")
    ax.set_ylabel("Probability")

    plt.xticks(rotation=45)

    return fig