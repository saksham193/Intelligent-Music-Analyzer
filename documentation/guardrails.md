# ⚙️ Project Guardrails

## Code
•⁠  ⁠Keep code modular (⁠ extract_features.py ⁠, ⁠ train_model.py ⁠, ⁠ predict.py ⁠).
•⁠  ⁠Use relative paths.
•⁠  ⁠Save visuals in ⁠ /images/ ⁠ folder.
•⁠  ⁠Log errors instead of printing.

## Data
•⁠  ⁠Ensure all audio files are 30s or trimmed to same length.
•⁠  ⁠Use ⁠ .wav ⁠ or ⁠ .mp3 ⁠ formats only.
•⁠  ⁠Maintain train/test split consistency.

## Model
•⁠  ⁠Use chroma image input size = 128x128 (standardize).
•⁠  ⁠Save best model only.
•⁠  ⁠Validate with unseen test data.

## Repo Hygiene
•⁠  ⁠No raw datasets uploaded to GitHub.
•⁠  ⁠Use ⁠ .gitignore ⁠ for large files (⁠ *.wav ⁠, ⁠ *.h5 ⁠, ⁠ *.pkl ⁠).
•⁠  ⁠Include ⁠ requirements.txt ⁠ and ⁠ README.md ⁠.

## Collaboration
•⁠  ⁠Follow concise commits.
•⁠  ⁠Document function purpose briefly in comments.