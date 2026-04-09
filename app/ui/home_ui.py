import streamlit as st
import os
import numpy as np

from backend.visualizer import (
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_chroma,
    plot_confidence
)

from backend.mood_detector import detect_mood
from backend.predictor import predict_genre
from backend.recommender import recommend_songs


def show_home():
    st.title("🎵 Music Genre Classifier")

    st.markdown("Upload an audio file and detect its genre + mood")

    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3"],
        label_visibility="collapsed"
    )

    if st.button("Predict"):
        if uploaded_file is not None:

            # Save temp file
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Play audio
            st.audio(uploaded_file)

            st.info("Processing audio... 🎧")

            try:
                # 🎵 GENRE PREDICTION (UPDATED)
                genre, confidence, predictions = predict_genre(temp_path)

                # 😊 MOOD DETECTION
                mood, tempo = detect_mood(temp_path)

                # ----------------------------
                # 🎯 OUTPUT UI
                # ----------------------------
                st.success(f"🎵 Predicted Genre: {genre}")
                st.info(f"🔥 Confidence: {confidence:.2f}")

                st.markdown("---")

                st.success(f"😊 Mood: {mood}")
                st.info(f"⚡ Tempo: {tempo:.2f} BPM")

                st.markdown("---")

                # ----------------------------
                # 📊 CONFIDENCE GRAPH
                # ----------------------------
                st.subheader("📊 Genre Confidence Distribution")

                labels = np.load("model/label_classes.npy", allow_pickle=True)
                conf_fig = plot_confidence(predictions, labels)
                st.pyplot(conf_fig)

                st.markdown("---")

                # ----------------------------
                # 📊 AUDIO VISUALIZATION
                # ----------------------------
                st.subheader("📊 Audio Visualization")

                # Waveform
                st.write("🎧 Waveform")
                waveform_fig = plot_waveform(temp_path)
                st.pyplot(waveform_fig)

                # Spectrogram
                st.write("🎵 Spectrogram")
                spec_fig = plot_spectrogram(temp_path)
                st.pyplot(spec_fig)

                # 🔥 Mel Spectrogram (NEW)
                st.write("🔥 Mel Spectrogram")
                mel_fig = plot_mel_spectrogram(temp_path)
                st.pyplot(mel_fig)

                # 🔥 Chroma Features (NEW)
                st.write("🎼 Chroma Features")
                chroma_fig = plot_chroma(temp_path)
                st.pyplot(chroma_fig)

                # ------------------------------
                # 🔍 SONG RECOMMENDATION
                # ------------------------------
                recommendations = recommend_songs(genre)

                st.markdown("---")
                st.subheader("🎧 Recommended Songs")

                for song in recommendations:
                    st.write(f"👉 {song}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        else:
            st.warning("⚠️ Please upload an audio file first.")