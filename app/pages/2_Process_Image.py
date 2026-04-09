import streamlit as st
import time

if "uploaded_file" not in st.session_state:
    st.switch_page("main.py")

st.set_page_config(page_title="Processing Image", layout="centered")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .loader {
            border: 6px solid #0E1117;
            border-top: 6px solid #3B4CCA;
            border-radius: 50%;
            width: 80px;
            height: 80px;
            animation: spin 1s linear infinite;
            margin-top: 30px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loader-text {
            color: #9AA0A6;
            margin-top: 20px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🔄 Processing your uploaded image...")

st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
st.markdown('<div class="loader-text">Analyzing chroma patterns...</div>', unsafe_allow_html=True)

time.sleep(4)

st.switch_page("pages/3_Show_Genre.py")

