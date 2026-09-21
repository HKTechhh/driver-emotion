# Driver Emotion Recognition — Build Plan (Cursor edition)

**Project:** Analysing driver emotions under urban traffic conditions using deep learning
**Comparison:** Custom CNN with VGG-style kernels/filters vs VGG16 transfer learning
**Order:** Build and freeze the models first (Phases 0–9), then write the paper (Phase 10).

---

## How to use this plan in Cursor

1. Open the `driver-emotion/` folder in Cursor. The rules in `.cursor/rules/project.mdc` load automatically, so the agent always knows the project goals, the folder layout and the coding standards.
2. **One phase = one new Agent chat.** Start each chat by pasting that phase's prompt (below). Always `@PLAN.md` and `@config.py` so the agent has the context.
3. Let the agent write the code, then **run the "Done when" checks yourself** in the terminal. Don't move on until they pass.
4. `git commit` at the end of every phase (`git commit -m "phase 3: vgg16 training"`). If Cursor breaks something later, you can roll back.
5. Tick the box in the progress tracker at the bottom and add a line to `docs/experiment_log.md`. That log becomes your Methods and Results sections later.

**Where things run**

| Task | Where |
|---|---|
| Writing code, smoke tests, EDA, real-time webcam app | Your laptop (Kali, CPU is fine) |
| Full training runs (especially VGG16) | Google Colab or Kaggle notebook with a free GPU (T4) |
| Syncing code between the two | GitHub: push from Cursor, `git clone` / `git pull` in Colab |

Colab cell to train (Runtime → Change runtime type → T4 GPU):

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/<you>/driver-emotion.git && cd driver-emotion  # later: !git pull
%cd driver-emotion
!pip install -q kaggle visualkeras
# upload kaggle.json, then:
!mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013 --unzip -q
!python -m src.train --model custom_cnn --run-name cnn_v1
!cp models/*.keras results/*.csv /content/drive/MyDrive/driver-emotion/   # keep results safe
```

Every training script takes a `--subset 0.02` flag so you can test it on 2% of the data on your CPU in about a minute before paying for GPU time.

---

## Phase 0 — Environment and repo setup (Day 1)

**Goal:** a clean Python 3.11 environment and a Git repo.

Kali ships a newer Python than TensorFlow/MediaPipe like, so use `uv` to get 3.11:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd driver-emotion
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import tensorflow as tf, cv2, mediapipe; print(tf.__version__)"
git init && git add . && git commit -m "phase 0: scaffold"
```

In Cursor, pick the `.venv` interpreter (Ctrl+Shift+P → "Python: Select Interpreter").

**Cursor prompt:**
> Read @PLAN.md and @config.py. Create empty `__init__.py` files for `src/` and `src/models/`, a `src/utils.py` with `set_seed(seed)` (python, numpy, tensorflow), `ensure_dirs()` that creates every directory defined in config, and `count_params(model)`. Add a `tests/test_setup.py` that imports config and calls `ensure_dirs()`. Don't change config.py.

**Done when:** `pytest -q` passes and `python -c "from src.utils import ensure_dirs; ensure_dirs()"` runs clean.

---

## Phase 1 — Data (Days 2–4)

**Goal:** FER2013 downloaded, explored, and loadable as a `tf.data` pipeline for both models.

Download (needs a Kaggle account and API token in `~/.kaggle/kaggle.json`):

```bash
uv pip install kaggle
kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013 --unzip
# expected: data/raw/fer2013/train/<7 class folders>, data/raw/fer2013/test/<7 class folders>
```

Also request **KMU-FED** (real in-car driver faces) now; approval can take a while. It's used in Phase 5 as an in-car test set.

**Cursor prompt 1 (EDA):**
> Create `notebooks/01_eda.ipynb` that uses paths from @config.py to: count images per class for train and test, plot a bar chart of the class distribution, show a 7×5 grid of sample faces (5 per class), and print image size and mode. Save the bar chart and grid to `results/figures/` as PNG at 200 dpi.

**Cursor prompt 2 (data pipeline):**
> Create `src/data.py` with a function `get_datasets(model_name, subset=1.0)` that returns `(train_ds, val_ds, test_ds, class_weights)`.
> - Use `tf.keras.utils.image_dataset_from_directory` on FER_DIR/train with `validation_split=VAL_SPLIT` and `seed=SEED`, and FER_DIR/test for the test set. Class order must equal CLASS_NAMES.
> - For `custom_cnn`: grayscale, resize to CNN["img_size"], scale to [0,1].
> - For `vgg16`: load grayscale, convert to 3 channels, resize to VGG16["img_size"], apply `keras.applications.vgg16.preprocess_input`.
> - Augmentation on the training set only, as a Keras `Sequential` of layers: RandomFlip("horizontal"), RandomRotation(0.05), RandomZoom(0.1), RandomBrightness(0.2), RandomContrast(0.2).
> - `subset` < 1.0 takes that fraction of each split for quick tests.
> - Compute balanced class weights from training labels with sklearn.
> - Use `.cache()` (for custom_cnn only) and `.prefetch(tf.data.AUTOTUNE)`.
> Add `tests/test_data.py` that checks batch shapes and label range for both models with subset=0.01.

**Done when:** `pytest tests/test_data.py -q` passes and the EDA figures exist. Write the class counts and the "disgust" imbalance in the experiment log.

---

## Phase 2 — Model A: custom CNN with VGG-style kernels (Days 5–8)

**Goal:** a from-scratch CNN built from VGG-style blocks (stacked 3×3 convolutions → 2×2 max-pooling), trained and saved.

**Cursor prompt 1 (model):**
> Create `src/models/custom_cnn.py` with `build_custom_cnn(cfg=CNN, num_classes=NUM_CLASSES)`. Architecture: for each value in `cfg["filters"]` (64,128,256,512) add a VGG block of two Conv2D(3×3, padding="same") → BatchNorm → ReLU, then MaxPool 2×2 and Dropout(0.25). Then GlobalAveragePooling2D → Dense(256) → BatchNorm → ReLU → Dropout(cfg["dropout"]) → Dense(num_classes, softmax). Use L2(1e-4) on the conv layers. Name each layer clearly (e.g. `block1_conv1`) so it reads well in Netron and in the paper. Add a `__main__` that prints `model.summary()`.

**Cursor prompt 2 (training script, shared by both models):**
> Create `src/train.py` with an argparse CLI: `--model {custom_cnn,vgg16}`, `--subset` (float, default 1.0), `--epochs` (optional override), `--img-size` (optional override, passed to both `get_datasets` and the model builder — add an `img_size=None` parameter to `get_datasets` for this), `--run-name` (optional, default `{model}_{timestamp}`). It should:
> - call `set_seed`, `ensure_dirs`, `get_datasets`
> - build the chosen model; compile with Adam, sparse categorical cross-entropy, and accuracy metric
> - callbacks: ModelCheckpoint (best val_accuracy → `models/{run_name}.keras`), EarlyStopping (patience 10, restore best), ReduceLROnPlateau (factor 0.5, patience 4), TensorBoard (`logs/{run_name}`), CSVLogger (`results/{run_name}_history.csv`)
> - pass class_weights to `fit`
> - after training, save loss/accuracy curves to `results/figures/{run_name}_curves.png`
> - append a row to `results/runs.csv`: run_name, model, date, epochs run, best val_acc, params, training time in minutes, img_size, batch_size, lr
> - the vgg16 path should call a `train_vgg16()` function that Phase 3 will fill in; for now raise NotImplementedError.

**Run it:**

```bash
python -m src.train --model custom_cnn --subset 0.02 --epochs 2   # laptop smoke test (~1 min)
# then in Colab (GPU):
python -m src.train --model custom_cnn --run-name cnn_v1
tensorboard --logdir logs    # watch the curves
```

**Done when:** `models/cnn_v1.keras` exists, validation accuracy is about 60–68% (normal for FER2013), and the curves show no runaway overfitting. If train accuracy is far above val, increase dropout or augmentation, try again as `cnn_v2`, and note what changed in the log.

---

## Phase 3 — Model B: VGG16 transfer learning (Days 9–12)

**Goal:** VGG16 pretrained on ImageNet, trained in two stages.

**Cursor prompt:**
> Create `src/models/vgg16_tl.py` with `build_vgg16(cfg=VGG16, num_classes=NUM_CLASSES)` that loads `keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(img, img, 3))`, freezes the base, and adds GlobalAveragePooling2D → Dense(512, relu) → Dropout(cfg["dropout"]) → Dense(num_classes, softmax). Add `unfreeze_top(model, from_layer=cfg["unfreeze_from"])` that makes layers from that name onward trainable and keeps BatchNorm layers (if any) frozen.
> Then implement `train_vgg16()` in @src/train.py as two stages:
> Stage 1: frozen base, lr=cfg["head_lr"], cfg["head_epochs"] epochs.
> Stage 2: unfreeze_top, recompile with lr=cfg["finetune_lr"], continue for cfg["finetune_epochs"] epochs using `initial_epoch`.
> Combine both histories into one curves plot with a vertical line where fine-tuning starts. Log it to runs.csv like the CNN.

**Run it:**

```bash
python -m src.train --model vgg16 --subset 0.02 --epochs 1   # smoke test (use --img-size 96 if too slow on CPU)
# Colab GPU:
python -m src.train --model vgg16 --run-name vgg16_v1
```

**Done when:** `models/vgg16_v1.keras` exists and validation accuracy is in the high 60s or better. Save both `.keras` files to Google Drive; they're large.

---

## Phase 4 — Evaluation and comparison (Days 13–15)

**Goal:** a fair side-by-side comparison of both models on the same test set.

**Cursor prompt:**
> Create `src/evaluate.py` with a CLI `--model-path` and `--model {custom_cnn,vgg16}`. On the test set, compute: accuracy, macro-F1, weighted-F1, per-class precision/recall/F1 (sklearn classification_report), and the confusion matrix (raw and normalised, saved as PNG heatmaps with CLASS_NAMES labels). Also measure: parameter count, `.keras` file size in MB, and inference latency — the mean and p95 milliseconds for a single image over 200 runs after 20 warm-up runs, plus the implied FPS. Save everything to `results/{run_name}_metrics.json`.
> Then create `src/compare.py` that reads every `*_metrics.json` and writes `results/comparison.csv` plus a grouped bar chart (accuracy, macro-F1, FPS) to `results/figures/comparison.png`.

**Done when:** `results/comparison.csv` has one row per model. This table is the core of your Results section.

---

## Phase 5 — Robustness under "urban traffic" conditions (Days 16–18)

**Goal:** show how each model holds up under conditions a driver actually faces. This is what connects the project to *urban traffic*.

**Cursor prompt:**
> Create `src/robustness.py`. For each condition in `ROBUSTNESS_CONDITIONS` in @config.py, apply the corruption to every test image (deterministic, seeded) and re-evaluate both models: accuracy and macro-F1. Conditions: `clean`, `low_light` (brightness ×0.4 + gaussian noise σ=10 to mimic night/tunnels), `glare` (brightness ×1.6 with a random bright elliptical blob), `motion_blur` (horizontal kernel size 7, vibration), `occlusion` (random black rectangle covering 20% of the face — hand/sunglasses), and `head_pose` (rotation ±20°). Output `results/robustness.csv` (rows = conditions, columns = model metrics) and a line chart `results/figures/robustness.png`. Also save one example image per condition to `results/figures/conditions_grid.png`.

If KMU-FED arrives, add a `--dataset kmu` option to `evaluate.py` and report cross-dataset accuracy. It's a strong result for the paper even if the numbers drop.

**Done when:** `robustness.csv` exists and you can say, in one sentence, which model degrades less under each condition.

---

## Phase 6 — Explainability with Grad-CAM (Days 19–20)

**Cursor prompt:**
> Create `src/gradcam.py` that, for a given model and a list of test images, computes Grad-CAM heatmaps from the last convolutional layer (`block4_conv2` for custom_cnn, `block5_conv3` for vgg16), overlays them on the face, and saves a grid: rows = 7 emotions (one correct example each), columns = original | CNN Grad-CAM | VGG16 Grad-CAM. Save to `results/figures/gradcam_grid.png`. Also save a grid of 6 misclassified examples per model.

**Done when:** the heatmaps focus on the eyes, brows and mouth. If they highlight the background, note that as a finding.

---

## Phase 7 — Real-time driver app (Days 21–25)

**Goal:** a live webcam app that detects the driver's face and shows their emotion.

**Cursor prompt 1 (core loop):**
> Create `src/realtime.py` with CLI `--model-path`, `--model {custom_cnn,vgg16}`, `--camera` (default REALTIME["camera_index"]), `--video` (optional file path instead of the camera), and `--log` (CSV path). Pipeline per frame: OpenCV capture → MediaPipe Face Detection (min confidence from config) → crop the largest face with 15% padding → preprocess exactly like `src/data.py` for that model → predict. Smooth predictions with a moving average over `REALTIME["smoothing_window"]` frames. Draw the bounding box, the top emotion with its confidence, a small bar chart of all 7 probabilities, and FPS. If an emotion in `alert_emotions` stays on top for more than `alert_seconds`, show a red "Stay calm — take a breath" banner. Log timestamp, emotion, confidence and FPS to CSV. Press `q` to quit. Move the preprocessing into a shared `src/preprocess.py` used by both data.py and realtime.py so training and live input always match.

**Cursor prompt 2 (optional demo UI):**
> Create `app.py`, a Streamlit app with a sidebar to choose the model, a live webcam feed via `streamlit-webrtc`, and a live line chart of emotion confidence over time. Reuse `src/preprocess.py` and the model loading from `src/realtime.py`.

**Run it:**

```bash
python -m src.realtime --model custom_cnn --model-path models/cnn_v1.keras --log results/live_log.csv
```

**Done when:** it runs at a usable FPS on your laptop. Test with a few consenting volunteers in daylight, at night, and wearing sunglasses. Record FPS for each model (use a short screen recording for your presentation).

---

## Phase 8 — Model viewing and paper figures (Days 26–27)

- **Netron:** `uv pip install netron` then `netron models/cnn_v1.keras` (or drag the file into netron.app). Screenshot both architectures.
- **visualkeras:**
  > Cursor prompt: Create `src/plot_architecture.py` that loads both models and saves layered architecture diagrams with visualkeras (with legend) to `paper/figures/`.
- **TensorBoard:** screenshot the training curves (or reuse the PNGs from Phase 2/3).
- Copy the final figures into `paper/figures/`: EDA, curves, confusion matrices, comparison, robustness, Grad-CAM, architecture diagrams, a real-time app screenshot.

---

## Phase 9 — Freeze results (Day 28)

- Final models chosen and named (e.g. `cnn_final.keras`, `vgg16_final.keras`), stored on Google Drive.
- `results/comparison.csv`, `robustness.csv` and `runs.csv` final.
- `docs/experiment_log.md` complete: every run, what changed, why.
- `README.md` with setup and run commands (Cursor prompt: *"Write a README from @PLAN.md and the actual scripts in src/"*).
- Tag the repo: `git tag v1.0-results && git push --tags`.

**Rule:** once results are frozen, don't retrain while writing. If you do, re-run Phases 4–6 so every number in the paper matches.

---

## Phase 10 — Research paper (after the code, ~2 weeks)

Write from `docs/experiment_log.md`, the CSVs and `paper/figures/`. Target 15 pages:

| Section | Pages | Source material |
|---|---|---|
| Abstract | 0.5 | Write last |
| 1. Introduction (road safety, emotion and driving, research questions) | 1.5 | Literature |
| 2. Literature review (FER, CNNs, VGG, driver monitoring systems) | 3 | Literature |
| 3. Methodology (datasets, preprocessing, both architectures, training, metrics, robustness protocol, real-time pipeline) | 3.5 | Phases 1–3, 5, 7, config.py |
| 4. Results (comparison table, curves, confusion matrices, robustness, FPS, Grad-CAM) | 2.5 | Phases 4–6 |
| 5. Discussion (accuracy vs speed, which fits a car, failure cases, limitations) | 2 | Your notes |
| 6. Ethics and privacy | 0.5 | Consent, on-device processing, dataset bias |
| 7. Conclusion and future work | 1 | |
| References | — | APA or IEEE, as your department requires |

---

## Progress tracker

- [x] Phase 0 — Environment and repo
- [x] Phase 1 — Data pipeline and EDA
- [x] Phase 2 — Custom CNN trained (`cnn_v1`)
- [x] Phase 3 — VGG16 trained (`vgg16_v1`)
  - [x] Model + two-stage training script ready (`src/models/vgg16_tl.py`, `train_vgg16`); CPU smoke test passed
- [x] Phase 4 — Evaluation and comparison table
- [x] Phase 5 — Robustness tests
- [x] Phase 6 — Grad-CAM
- [ ] Phase 7 — Real-time app
- [ ] Phase 8 — Figures and model viewing
- [ ] Phase 9 — Results frozen (`v1.0-results`)
- [ ] Phase 10 — Paper

## When Cursor gets stuck

- **Shape errors:** paste the full traceback and add "print the shapes at each step before fixing".
- **Out of memory on Colab:** lower `batch_size` in config, or VGG16 `img_size` to 160.
- **Accuracy stuck around 25%:** the model is predicting only "happy". Check labels, check class weights, and lower the learning rate.
- **Agent rewriting files you didn't ask about:** say "only edit the files I name" and reject the other diffs.
- **Live app accuracy far worse than the test set:** preprocessing mismatch. Compare `src/preprocess.py` output against a training batch.
