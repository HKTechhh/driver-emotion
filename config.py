"""Central project settings.

Every script imports paths and hyperparameters from here. Nothing is hard-coded
elsewhere, so the Methods section of the paper can be written straight from this file.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FER_DIR = RAW_DIR / "fer2013"      # expects train/ and test/, one sub-folder per class
KMU_DIR = RAW_DIR / "kmu_fed"      # optional in-car test set (Phase 5)
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"
LOGS_DIR = ROOT / "logs"           # TensorBoard

ALL_DIRS = [DATA_DIR, RAW_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, PAPER_FIG_DIR, LOGS_DIR]

# ---------------------------------------------------------------- data
# Alphabetical, so it matches the folder order image_dataset_from_directory uses.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
VAL_SPLIT = 0.15                   # carved from FER2013 train; FER2013 test stays untouched

# "balanced" (sklearn) alone let disgust's weight (~9.5x) destabilize training -- see
# docs/experiment_log.md. "sqrt" square-roots the balanced weights and caps them at
# CLASS_WEIGHT_MAX, keeping the imbalance correction without the instability.
CLASS_WEIGHT_MODE = "sqrt"         # one of "none", "balanced", "sqrt"
CLASS_WEIGHT_MAX = 3.0

# ---------------------------------------------------------------- Model A: custom CNN (VGG-style kernels)
CNN = {
    "img_size": 48,
    "channels": 1,
    "filters": (64, 128, 256, 512),  # one VGG block (2x conv3x3 + maxpool) per value
    "kernel_size": 3,
    "l2": 1e-4,
    "dropout": 0.4,
    "batch_size": 64,
    "epochs": 60,
    "lr": 1e-3,
}

# ---------------------------------------------------------------- Model B: VGG16 transfer learning
VGG16 = {
    "img_size": 224,                 # drop to 96/160 for CPU smoke tests or Colab OOM
    "channels": 3,
    "dropout": 0.5,
    "batch_size": 32,
    "head_epochs": 15,               # stage 1: frozen base
    "finetune_epochs": 25,           # stage 2: unfreeze top block
    "head_lr": 1e-3,
    "finetune_lr": 1e-5,
    "unfreeze_from": "block5_conv1",
}

# ---------------------------------------------------------------- training callbacks
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 4
LR_FACTOR = 0.5

# ---------------------------------------------------------------- robustness ("urban traffic" conditions)
ROBUSTNESS_CONDITIONS = ["clean", "low_light", "glare", "motion_blur", "occlusion", "head_pose"]

# ---------------------------------------------------------------- real-time app
REALTIME = {
    "camera_index": 0,
    "min_face_confidence": 0.6,
    "face_padding": 0.15,
    "smoothing_window": 10,          # frames
    "alert_emotions": ("angry", "fear"),
    "alert_seconds": 3.0,
}

# ---------------------------------------------------------------- Grad-CAM target layers
GRADCAM_LAYER = {"custom_cnn": "block4_conv2", "vgg16": "block5_conv3"}
