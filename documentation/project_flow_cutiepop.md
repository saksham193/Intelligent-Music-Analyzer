# 🎀 Music Genre Classification — Project Flow

Hey bestie 💕 here’s your step-by-step guide to slay this project like a pro 🎧✨  

---

## 💅 Step 1: UI / Frontend Setup
- Start with a cute **Streamlit** or **Tkinter** app 💻  
- Add:
  - 🎵 File upload button (`.wav` / `.mp3`)
  - 💖 “Predict Genre” button (placeholder first)
- Show:
  - File name
  - Chroma visualization
  - Predicted genre (after model hookup)

---

## 🎧 Step 2: Data Preparation
- Grab your dataset (GTZAN works great!)  
- Folder glow-up:
  ```bash
  dataset/
    ├── rock/
    ├── jazz/
    ├── classical/
    └── pop/
  ```
- Normalize the vibes:
  - Mono sound only 🦋  
  - 22,050 Hz sample rate  
  - 30 seconds long (trim/pad)

---

## 💽 Step 3: Feature Extraction
- Use **Librosa** to get that **chroma energy** 🎶  
- Convert to grayscale images (128×128).  
- Save them like:
  ```bash
  images/<genre>/<filename>.png
  ```

---

## 🪄 Step 4: Model Creation
- Craft a **CNN** (cutest brain ever 🧠💋)  
- Input: chroma images  
- Output: genre labels  
- Split: 80% train / 10% val / 10% test  
- Save your model:
  ```bash
  models/best_model.h5
  ```

---

## 💫 Step 5: Integration
- Connect UI ➡️ Model ➡️ Prediction flow 💞  
- Flow:
  - Upload → Extract chroma → Generate image → Predict genre  
- Add progress bar + prediction message ✨

---

## 🧁 Step 6: Testing & Optimization
- Test with unseen songs 🎤  
- Check confusion matrix 📊  
- Tune your model till it hits high accuracy 💅  

---

## 🌸 Step 7: Final Packaging
- Bundle your magic:
  - `requirements.txt`
  - `README.md`
  - `models/best_model.h5`
  - Screenshots or demo video  
- Bonus points:
  - Add **Dockerfile** 💅  
  - Deploy on Streamlit Cloud ☁️  

---

Go rock that genre classification queen 👑🎶💋
