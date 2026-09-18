# Experiment log

Add a line after every run or decision. This becomes the Methods and Results sections of the paper.

| Date | Phase | Run name | What changed / why | Key result | Notes |
|---|---|---|---|---|---|
| 2026-09-18 | 0 | — | Environment set up: uv venv (Python 3.11.15), deps installed from requirements.txt, src/ and tests/ scaffolded | TensorFlow 2.20.0, CPU only (no GPU detected) | mediapipe installed and imported fine; git repo initialised scoped to this folder |
| | 1 | — | FER2013 downloaded, class counts recorded | | Note the "disgust" imbalance |
| | 2 | cnn_v1 | Baseline custom CNN, config defaults | val_acc = | |
| | 3 | vgg16_v1 | VGG16, 2-stage fine-tuning from block5 | val_acc = | |

## Decisions
- (e.g. "Kept all 7 classes and used class weights instead of merging disgust into angry, because …")

## Problems and fixes
- (e.g. "Colab OOM at batch 64 → reduced VGG16 batch to 32")
