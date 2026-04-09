import streamlit as st
import os
import numpy as np
import tempfile

from backend.visualizer import (
    plot_waveform, plot_spectrogram, plot_mel_spectrogram,
    plot_chroma, plot_confidence
)
from backend.mood_detector import detect_mood
from backend.predictor import predict_genre
from backend.recommender import recommend_songs


# ----------------------------
# 🎨 CUSTOM UI STYLES
# ----------------------------
def apply_custom_styles():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 100%);
            color: #e0e0e0;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }

        .stFileUploader {
            border: 1px dashed #6366f1;
            border-radius: 12px;
            padding: 10px;
            background: rgba(99, 102, 241, 0.05);
        }

        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 10px;
        }

        .result-card h3 {
            color: #9ca3af;
            font-size: 0.9rem;
            margin: 0;
            text-transform: uppercase;
        }

        .result-card h2 {
            color: #6366f1;
            font-size: 1.8rem;
            margin: 5px 0 0 0;
        }

        .chart-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)


# ----------------------------
# 📊 METRICS DISPLAY
# ----------------------------
def display_metrics(genre, mood, tempo, confidence):
    cols = st.columns(4)

    metrics = [
        ("🎵 Genre", f"{genre}"),
        ("😊 Mood", f"{mood}"),
        ("⚡ Tempo", f"{tempo:.0f} BPM"),
        ("📊 Confidence", f"{confidence*100:.1f}%")
    ]

    for col, (label, val) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="result-card"><h3>{label}</h3><h2>{val}</h2></div>',
                unsafe_allow_html=True
            )


# ----------------------------
# 🏠 MAIN UI
# ----------------------------
def show_home():
    st.set_page_config(page_title="Music AI", page_icon="🎵", layout="wide")
    apply_custom_styles()

    st.title("🎵 Intelligent Music Analyzer")
    st.markdown("Upload a track to analyze genre, mood, and acoustic signatures.")

    uploaded_file = st.file_uploader("", type=["wav", "mp3"])

    if uploaded_file is not None:

        # Save temp file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        st.audio(uploaded_file)

        if st.button("🔍 Run Full Analysis", use_container_width=True):
            try:
                with st.spinner("Processing audio signals..."):

                    # 🎵 GENRE
                    genre, confidence, predictions = predict_genre(temp_path)

                    # 😊 MOOD
                    mood, tempo = detect_mood(temp_path)

                    # 🎧 RECOMMENDATIONS
                    recommendations = recommend_songs(genre)

                    labels = np.load("model/label_classes.npy", allow_pickle=True)

                    # ----------------------------
                    # 🎯 RESULTS
                    # ----------------------------
                    display_metrics(genre, mood, tempo, confidence)

                    # ----------------------------
                    # 🎧 RECOMMENDATION SECTION (FIXED)
                    # ----------------------------
                    st.markdown("### 🎧 Similar Recommendations")

                    if recommendations:
                        for song in recommendations:

                            # ✅ Handle both formats safely
                            if isinstance(song, dict):
                                name = song.get("name", "Unknown")
                                artist = song.get("artist", "Unknown Artist")
                                url = song.get("url", "#")
                            else:
                                name = song
                                artist = "Unknown Artist"
                                url = "#"

                            st.markdown(f"""
                            <div style="
                                background: rgba(255,255,255,0.05);
                                padding: 12px;
                                border-radius: 10px;
                                margin-bottom: 10px;
                                border: 1px solid rgba(99,102,241,0.2);
                            ">
                                <b>🎵 {name}</b><br>
                                👤 {artist}<br>
                                <a href="{url}" target="_blank" style="color:#6366f1;">
                                    🔗 Listen / View
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No recommendations found.")

                    st.divider()

                    # ----------------------------
                    # 📊 VISUALIZATION
                    # ----------------------------
                    st.subheader("📊 Audio Visualizations")

                    viz_tasks = [
                        ("Genre Confidence", lambda: plot_confidence(predictions, labels)),
                        ("Waveform", lambda: plot_waveform(temp_path)),
                        ("Spectrogram", lambda: plot_spectrogram(temp_path)),
                        ("Mel Spectrogram", lambda: plot_mel_spectrogram(temp_path)),
                        ("Chroma Features", lambda: plot_chroma(temp_path))
                    ]

                    for i in range(0, len(viz_tasks), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(viz_tasks):
                                title, func = viz_tasks[i + j]
                                with cols[j]:
                                    st.markdown(f'<div class="chart-box"><b>{title}</b>', unsafe_allow_html=True)
                                    fig = func()
                                    st.pyplot(fig, use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis Error: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    else:
        st.info("Directly drag an audio file above to begin.")


# ----------------------------
# 🚀 ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    show_home()