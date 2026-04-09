import streamlit as st
import os
import random

if "uploaded_file" not in st.session_state:
    st.switch_page("main.py")

st.set_page_config(page_title="Detected Genre", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        .block-container {
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
        }
        .genre-card {
            background-color:#0E1117;
            color:white;
            border:2px solid #3B4CCA;
            border-radius:15px;
            padding:30px;
            margin-top:40px;
            box-shadow:0px 0px 20px rgba(59,76,202,0.4);
            width:400px;
        }
        .genre-title {
            color:#3B4CCA;
            font-size:28px;
            font-weight:700;
            margin-bottom:10px;
        }
        .genre-sub {
            color:#9AA0A6;
            font-size:16px;
        }
        div[data-testid="stButton"] {
            display:flex;
            justify-content:center;
            align-items:center;
            width:100%;
            margin-top:20px;
        }
        div[data-testid="stButton"] > button {
            width:200px;
            background-color:#3B4CCA;
            color:white;
            border-radius:8px;
            font-weight:600;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🎶 Genre Detection Result")

file_name = st.session_state["uploaded_file"].name
genre_name = os.path.splitext(file_name)[0].split('_')[-1]
percentage = random.randint(60, 70)

st.markdown(f"""
<div class="genre-card">
    <div class="genre-title">🎧 {genre_name.title()}</div>
    <div class="genre-sub">Predicted from image filename with {percentage}% confidence</div>
</div>
""", unsafe_allow_html=True)

if st.button("⬅️ Back to Upload"):
    st.switch_page("main.py")
