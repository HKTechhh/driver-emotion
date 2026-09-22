# Driver Emotion Recognition

Analysing driver emotions under urban traffic conditions using deep learning. A
comparative study of two architectures on FER2013 (7 emotion classes):

- **Model A — `custom_cnn`**: a from-scratch CNN built from VGG-style blocks (stacked
  3×3 convolutions → 2×2 max-pooling), 48×48 grayscale input.
- **Model B — `vgg16`**: Keras `VGG16` pretrained on ImageNet, adapted with two-stage
  transfer learning, 224×224×3 input.

The full build plan, phase by phase, is in [`PLAN.md`](PLAN.md). Project rules and
coding conventions are in [`.cursor/rules/project.mdc`](.cursor/rules/project.mdc).

**Status:** models trained, evaluated, stress-tested and explained; a live webcam spot
check confirmed the real-time app; the Netron / TensorBoard / app screenshots are in the
paper; all 9 build phases are complete. What's left needs a person, not a computer:
verifying the paper's references, choosing the final format/page limit, and (optionally)
a longer live session with drivers. See `PLAN.md` (progress tracker) and
`docs/experiment_log.md` (every run, decision and bug).

## Results (FER2013, full 7,178-image test set)

| | Custom CNN (`cnn_v1`) | VGG16 (`vgg16_v1`) |
|---|---|---|
| Test accuracy | **0.6753** | 0.6581 |
| Macro-F1 | **0.6595** | 0.6397 |
| Parameters / file size | 4.83 M / 55.4 MiB | 14.98 M / 113.3 MiB |
| Latency, laptop CPU, 1 image | 39.9 ms (25 FPS) | 216.2 ms (4.6 FPS) |

Accuracy under simulated driving conditions (full test set):

| Condition | CNN | VGG16 |
|---|---|---|
| clean | 0.676 | 0.655 |
| low light | 0.286 | 0.218 |
| glare | 0.515 | 0.478 |
| motion blur | 0.441 | 0.286 |
| occlusion | 0.444 | 0.396 |
| head rotation (±20°) | 0.668 | 0.634 |

Read these with the caveats in mind: FER2013 is web images, not in-cabin footage; the
degradations are synthetic; each model was trained once; and low light collapses both
models (the CNN's 28.6% is only ~4 points above always answering *happy*). The
real-time app has been verified on simulated frames but not yet live. Details, per-class
results and the failure analysis are in `docs/experiment_log.md` and `paper/paper.md`.

## Setup

Requires Python 3.11 (TensorFlow/MediaPipe don't yet support newer Pythons). This
project uses [`uv`](https://github.com/astral-sh/uv) to manage that regardless of your
system Python version.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already installed
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Verify the install:

```bash
python -c "import tensorflow as tf, cv2, mediapipe, sklearn; print('TF', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))"
```

(If MediaPipe fails to install or import, that's fine — the real-time app in
`src/realtime.py` falls back to OpenCV's Haar cascade automatically.)

Run the tests:

```bash
python -m pytest -q
```

## Data

Download [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (needs a Kaggle
account and API token at `~/.kaggle/kaggle.json`):

```bash
uv pip install --python .venv/bin/python kaggle
kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013 --unzip
```

This should produce:

```
data/raw/fer2013/train/<angry,disgust,fear,happy,neutral,sad,surprise>/
data/raw/fer2013/test/<same 7 classes>/
```

`data/raw/` is gitignored — everyone who clones this repo needs to download the
dataset themselves.

## Everything is a module, run from the repo root

Every script has an argparse CLI and a `main()`, and is run as `python -m src.<name>`
so relative imports (`config`, `src.*`) resolve correctly. Add `--subset 0.02` (or
similar) to any training/evaluation script to smoke-test it on a couple of percent of
the data in under a minute on a CPU before spending GPU time on a full run.

### Exploration

```bash
jupyter notebook notebooks/01_eda.ipynb   # class counts, sample grid, image size/mode
```

### Training (`src/train.py`)

```bash
# CPU smoke test (~1 min)
python -m src.train --model custom_cnn --subset 0.02 --epochs 2
python -m src.train --model vgg16 --subset 0.02 --img-size 96 --head-epochs 1 --finetune-epochs 1

# Full run — do this on a GPU (Colab/Kaggle), not a laptop CPU
python -m src.train --model custom_cnn --run-name cnn_v1
python -m src.train --model vgg16 --run-name vgg16_v1
```

| Flag | Meaning |
|---|---|
| `--model {custom_cnn,vgg16}` | which architecture to train (required) |
| `--subset FLOAT` | fraction of each data split to use (default `1.0`) |
| `--epochs INT` | override the config's epoch count (`custom_cnn` only) |
| `--head-epochs INT` / `--finetune-epochs INT` | override the two `vgg16` stage lengths independently (frozen-base stage / fine-tune stage) |
| `--img-size INT` | override the config's input resolution (e.g. `--img-size 96` to smoke-test VGG16 on a CPU) |
| `--run-name NAME` | defaults to `{model}_{timestamp}`; controls every output filename |

`vgg16` trains in two stages automatically: a frozen ImageNet base at `head_lr` for
`head_epochs`, then `unfreeze_top` and fine-tune at `finetune_lr` for
`finetune_epochs`. Every run produces:

- `models/{run_name}.keras` — best checkpoint by `val_accuracy`
- `results/{run_name}_history.csv` — per-epoch metrics (both stages, for `vgg16`)
- `results/figures/{run_name}_curves.png` — loss/accuracy curves
- a row appended to `results/runs.csv`

### Evaluation (`src/evaluate.py`)

```bash
python -m src.evaluate --model custom_cnn --model-path models/cnn_v1.keras
python -m src.evaluate --model vgg16 --model-path models/vgg16_v1.keras
```

Writes `results/{run_name}_metrics.json` (accuracy, macro/weighted F1, full per-class
`classification_report`, param count, file size, and single-image inference
latency/FPS) plus confusion matrix heatmaps in `results/figures/`.

### Comparison (`src/compare.py`)

```bash
python -m src.compare
```

Aggregates every `results/*_metrics.json` into `results/comparison.csv` and
`results/figures/comparison.png`.

### Robustness under simulated driving conditions (`src/robustness.py`)

```bash
python -m src.robustness --cnn-model-path models/cnn_v1.keras --vgg16-model-path models/vgg16_v1.keras
```

Re-evaluates both models under six corruptions (`clean`, `low_light`, `glare`,
`motion_blur`, `occlusion`, `head_pose`) meant to mimic real driving conditions.
Writes `results/robustness.csv`, `results/figures/robustness.png`, and
`results/figures/conditions_grid.png`.

### Grad-CAM explainability (`src/gradcam.py`)

```bash
python -m src.gradcam --cnn-model-path models/cnn_v1.keras --vgg16-model-path models/vgg16_v1.keras
```

Saves `results/figures/gradcam_grid.png` (one correctly classified example per
emotion, original vs. each model's Grad-CAM) and a misclassified-examples grid per
model.

### Is the CNN-vs-VGG16 gap real? (`src/significance.py`)

```bash
python -m src.significance --cnn-model-path models/cnn_v1.keras --vgg16-model-path models/vgg16_v1.keras
```

Saves per-image predictions (`results/predictions_*.csv`) and writes
`results/significance.json`: bootstrap 95% confidence intervals, a paired bootstrap for the
accuracy difference, and an exact McNemar test.

### Architecture diagrams (`src/plot_architecture.py`)

```bash
python -m src.plot_architecture
```

Saves layered architecture diagrams (via `visualkeras`) to `paper/figures/`.

### Real-time driver app (`src/realtime.py`)

```bash
python -m src.realtime --model custom_cnn --model-path models/cnn_v1.keras --log results/live_log.csv
```

Captures from a webcam (or `--video path/to/file.mp4`), detects the largest face,
predicts an emotion, smooths it over a moving window, and shows a live overlay
(bounding box, top emotion, per-class probability bars, FPS, and a "stay calm" alert
banner if `angry`/`fear` leads for too long). Press `q` to quit. Every frame's
prediction is logged to the `--log` CSV. Face detection prefers MediaPipe (downloading
a small model on first use) and falls back to OpenCV's Haar cascade automatically if
that's unavailable.

`src/preprocess.py` holds the per-model pixel preprocessing shared by `src/data.py`
(training) and `src/realtime.py` (live inference), so a camera frame is always treated
exactly like a training image.

## Project layout

```
config.py              # the only place for paths and hyperparameters
src/
  data.py               # tf.data pipelines for FER2013
  preprocess.py          # per-model preprocessing shared by training and the live app
  utils.py               # set_seed, ensure_dirs, count_params
  train.py                # shared training CLI for both models
  evaluate.py             # per-model test-set metrics
  compare.py               # aggregates metrics into one comparison table/chart
  robustness.py            # accuracy under simulated driving conditions
  gradcam.py                # Grad-CAM explainability grids
  significance.py             # bootstrap CIs + McNemar test for the CNN-vs-VGG16 gap
  plot_architecture.py       # visualkeras architecture diagrams
  realtime.py                # live webcam driver-emotion app
  models/
    custom_cnn.py             # Model A builder
    vgg16_tl.py                # Model B builder + two-stage fine-tuning helper
notebooks/01_eda.ipynb    # class distribution, sample grid, image properties
tests/                     # pytest
docs/experiment_log.md      # dated log of every run and decision
docs/car_deployment_guide.md # hardware/wiring/mounting/safety guide for running this in a real car
results/                     # CSVs, JSON metrics, figures (mostly gitignored outputs)
paper/figures/                 # curated final figures for the write-up
paper/paper.md                  # first draft of the research paper (TODOs marked inside)
paper/make_figures.py            # pipeline diagram, per-class chart, retained-accuracy chart, combined Grad-CAM
paper/build_docx.py              # any paper.md-style file -> .docx (python3 paper/build_docx.py [SRC.md OUT.docx]; docx is git-ignored)
```

## Notes

- `config.py` is the single source of truth for paths and hyperparameters — nothing is
  hard-coded elsewhere. Don't edit it without a reason; the Methods section of the
  paper is meant to be written straight from it.
- Full training runs (especially VGG16) should happen on a GPU (Google Colab or
  Kaggle), not a laptop CPU; see the Colab cell in `PLAN.md` for a ready-to-paste
  workflow, including syncing code via this GitHub repo.
- See `docs/experiment_log.md` for what's actually been run so far, what changed
  between runs, and why.
