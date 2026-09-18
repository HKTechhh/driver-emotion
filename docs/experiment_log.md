# Experiment log

Add a line after every run or decision. This becomes the Methods and Results sections of the paper.

| Date | Phase | Run name | What changed / why | Key result | Notes |
|---|---|---|---|---|---|
| 2026-09-18 | 0 | — | Environment set up: uv venv (Python 3.11.15), deps installed from requirements.txt, src/ and tests/ scaffolded | TensorFlow 2.20.0, CPU only (no GPU detected) | mediapipe installed and imported fine; git repo initialised scoped to this folder |
| 2026-09-18 | 1 | — | FER2013 downloaded to `data/raw/fer2013/{train,test}`, EDA run (`notebooks/01_eda.ipynb`), `src/data.py` pipeline built | Train 28,709 / test 7,178 images, 48x48 grayscale (mode `L`) | Strong "disgust" imbalance: 436 train / 111 test vs. 7,215 train / 1,774 test for "happy" (~16.5x fewer) — class weights in `src/data.py` handle this. FER2013 is noisy (some stock-photo watermarks, at least one blank/black image spotted in the sample grid) |
| | 2 | cnn_v1 | Baseline custom CNN, config defaults | val_acc = | |
| | 3 | vgg16_v1 | VGG16, 2-stage fine-tuning from block5 | val_acc = | |

## Decisions
- (e.g. "Kept all 7 classes and used class weights instead of merging disgust into angry, because …")

## Problems and fixes
- (e.g. "Colab OOM at batch 64 → reduced VGG16 batch to 32")
