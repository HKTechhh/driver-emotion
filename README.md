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
paper; all 9 build phases are complete. Both models have also been fine-tuned on a real
in-cabin dataset (KMU-FED, see below) and both fine-tuned checkpoints are wired into
`app_collect.py`. **The paper has not been updated with the KMU-FED work yet** — that
happens once the fine-tuned model has been validated on real driver photos/video, not
before. What's left otherwise needs a person, not a computer: verifying the paper's
references, choosing the final format/page limit, and (optionally) a longer live session
with drivers. See `PLAN.md` (progress tracker) and `docs/experiment_log.md` (every run,
decision and bug).

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

## In-car fine-tuning (KMU-FED)

Both models were also evaluated and fine-tuned on [KMU-FED](https://www.kaggle.com/datasets/anandpanajkar/kmu-fed),
a real in-cabin driver dataset (1,106 NIR/night-vision-style photos, 12 subjects, 6
emotions — no "neutral"), to measure and close the gap between FER2013's web photos and
an actual driving cabin. Full methodology (subject-level 8/2/2 split, why it's
seed-constrained on disgust coverage, three imbalance-handling attempts) is in
`docs/experiment_log.md`; only the headline numbers are here.

**Step 1 — domain-gap baseline** (both FER2013 models, unchanged, evaluated on KMU-FED's
held-out test subjects):

| | Custom CNN | VGG16 |
|---|---|---|
| FER2013 test accuracy | 67.5% | 65.8% |
| KMU-FED accuracy (no fine-tuning) | 25.6% | 36.3% |
| Accuracy drop | -41.9 pts | -29.6 pts |

**Step 2 — after fine-tuning** (fresh 6-unit head, last conv block + head unfrozen,
`--balance oversample` to counter disgust's scarcity — 60 of 700 train images vs.
120-160 for every other class):

| | `cnn_kmufed_ft_v1` | `vgg16_kmufed_ft_v1` |
|---|---|---|
| KMU-FED test accuracy | **63.1%** | **76.9%** |
| Macro-F1 | 0.560 | 0.725 |

Both recover most of the domain-gap loss from well under 900 fine-tuning images.
**Caveats that matter before trusting these numbers:** only 160 test images from 2
held-out subjects; a single training run per model (no seeds averaged); *disgust* is
essentially unlearned — the CNN gets it right once (1 of 20 images) after oversampling,
VGG16 still gets 0/0/0 precision/recall/F1 under every imbalance-handling strategy
tried; and KMU-FED, while a real in-cabin dataset, is still 12 people in one room, not a
moving vehicle across lighting conditions. Two imbalance-handling attempts before
oversampling (none, then class-weighting) are also logged in `docs/experiment_log.md`
for the full before/after comparison.

Both fine-tuned models are already selectable in `app_collect.py`'s model dropdown
alongside the original two — `model_class_names()` and `model_accuracy_info()` in that
file detect a model's output size and read its own result file, so the app's class
labels, sidebar accuracy badge and About tab automatically show the right numbers for
whichever model is selected, with no per-model special-casing needed elsewhere in the
app.

To re-run this yourself on Kaggle (needs a GPU and takes over an hour for VGG16):
`notebooks/kmu_fed_step1_step2_kaggle_cell.py` (see the docstring at the top of that
file for the two Kaggle inputs it needs — the KMU-FED dataset and a private dataset with
`cnn_v1.keras`/`vgg16_v1.keras`).

## Setup

Requires Python 3.11 (TensorFlow/MediaPipe don't yet support newer Pythons). This
project uses [`uv`](https://github.com/astral-sh/uv) to manage that regardless of your
system Python version.

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already installed
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # if uv isn't already installed
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\Activate.ps1
```

Verify the install (same command on any OS, once the venv is active):

```bash
python -c "import tensorflow as tf, cv2, mediapipe, sklearn; print('TF', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))"
```

(If MediaPipe fails to install or import, that's fine — the real-time app in
`src/realtime.py` falls back to OpenCV's Haar cascade automatically.)

Run the tests:

```bash
python -m pytest -q
```

**Model files (`models/*.keras`) are gitignored** — a plain `git clone` will not include
them. If you got this project as a handoff zip, they're already in `models/`; if you
cloned from GitHub instead, you'll need to either receive them separately or retrain (see
`## Training` below and, for the KMU-FED fine-tuned models, the Kaggle notebook
referenced in the section above).

**Windows-specific notes for the real-time/browser apps:**
- `src/realtime.py`'s webcam backend defaults to `auto`, which already tries `dshow` and
  `msmf` after the platform default fails — pass `--backend dshow` or `--backend msmf`
  explicitly if the camera window never opens.
- Long paths: if `pip`/`uv` complains about path length while installing TensorFlow,
  enable Windows' long-path support (`Settings > System > For developers > Enable long
  paths`, or `git config --system core.longpaths true`).
- Use PowerShell or the Windows Terminal, not `cmd.exe` — the activation script above
  (`Activate.ps1`) needs PowerShell; `cmd.exe` would need `.venv\Scripts\activate.bat`
  instead.

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

`--dataset kmu_fed` instead runs the Step 1 domain-gap baseline described above (both
`models/cnn_v1.keras`/`models/vgg16_v1.keras`, unchanged, on KMU-FED's held-out test
subjects) — needs `data/raw/kmu_fed/` populated (see the KMU-FED section above).

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
that's unavailable. If the camera window never opens (common on Windows, where the default
backend fails for some webcam drivers), try `--backend dshow` or `--backend msmf`; `auto`
(the default) already tries both after the platform default fails.

By default **no image is ever saved**, only the CSV log. Pass `--save-frames` to also
write one face crop in every `--save-every` (default 5) to `<log>_frames/`, e.g. to grow
the training set with in-vehicle data — get informed consent from whoever is on camera
first (`docs/car_deployment_guide.md`, Section 8). Filenames carry the *predicted*
emotion, not a verified label, so anyone using these for training still needs to check
or correct them by eye before adding them to a dataset.

`src/preprocess.py` holds the per-model pixel preprocessing shared by `src/data.py`
(training) and `src/realtime.py` (live inference), so a camera frame is always treated
exactly like a training image.

### Browser demo / data-collection app (`app_collect.py`)

```bash
streamlit run app_collect.py
```

A Streamlit dashboard with five tabs: **Image upload** and **Video upload** (run inference,
optionally correct the label, and save the face to `data/collected/` once the sidebar consent
checkbox is ticked), **Real-time video** (live webcam analysis in the browser via
`streamlit-webrtc` - view only, nothing is ever saved from this tab), **Analytics dashboard**
(what's been collected so far vs. FER2013's own class balance), and **About**. The sidebar's
six **traffic scenarios** (Night Driving, Highway/Freeway, etc.) each reapply one condition
from `src/robustness.py`'s `CONDITION_FUNCS` - the same functions used to measure the paper's
robustness numbers - to preview how the model's own accuracy changes under that condition,
plus a simple, clearly-labelled risk/safety heuristic (not a validated safety claim). Reuses
`src/realtime.py`'s `load_model_for_inference`/`predict_face` rather than duplicating them.

## Project layout

```text
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
  kmu_fed_data.py             # KMU-FED subject-level split (Step 1)
  train_kmu_fed_finetune.py    # fine-tune cnn_v1/vgg16_v1 on KMU-FED (Step 2)
  models/
    custom_cnn.py             # Model A builder
    vgg16_tl.py                # Model B builder + two-stage fine-tuning helper
notebooks/01_eda.ipynb    # class distribution, sample grid, image properties
notebooks/kmu_fed_*.py       # Kaggle cells: KMU-FED inspection, then Steps 1+2 end to end
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
