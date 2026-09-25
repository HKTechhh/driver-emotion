"""KMU-FED Steps 1 + 2, end to end - paste this whole file into one Kaggle notebook cell and
run it (Save & Run All / Commit, so it survives a disconnect - this takes a few minutes).

Before running:
  1) Add "anandpanajkar/kmu-fed" as an input (Datasets tab) - already done if you ran
     notebooks/kmu_fed_inspect_cell.py for Step 0.
  2) Create a private Kaggle Dataset from your local models/cnn_v1.keras and
     models/vgg16_v1.keras (kaggle.com -> Datasets -> New Dataset -> upload both files, any
     dataset name is fine), and Add that as an input too.

What this does:
  - Clones/pulls the repo, locates KMU-FED and the two base models under /kaggle/input
    (wherever Kaggle actually mounted them - the same nested-path issue Step 0 hit), and
    puts them where config.py expects (a symlink for the dataset, a copy for the two small
    model files).
  - Step 1: `python -m src.evaluate --dataset kmu_fed` - the pre-fine-tune domain-gap
    baseline for both models, unchanged, on the held-out test subjects.
  - Step 2: `python -m src.train_kmu_fed_finetune` for each model - fine-tunes a fresh 6-unit
    head (+ the last conv block) on the 8 train subjects, validates on the 2 val subjects,
    evaluates on the same 2 held-out test subjects Step 1 used.
  - Prints a before/after summary and zips everything you need to bring back.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/HKTechhh/driver-emotion.git"


def find_dir_containing(root: Path, name_fragment: str, max_depth: int = 6):
    """Walk `root` looking for a directory whose own name contains `name_fragment`."""
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if name_fragment in Path(dirpath).name.lower():
            return Path(dirpath)
    return None


def find_file(root: Path, filename: str, max_depth: int = 6):
    """Walk `root` looking for an exact filename."""
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if filename in filenames:
            return Path(dirpath) / filename
    return None


# ---- 1. Get the repo
subprocess.run(f"cd driver-emotion 2>/dev/null && git pull || git clone {REPO_URL}", shell=True, check=True)
os.chdir("driver-emotion")

# ---- 2. Dependencies (Kaggle's base image usually has TensorFlow/pandas/sklearn already;
#         opencv/mediapipe/seaborn are the ones actually new here)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)

# ---- 3. Locate KMU-FED and link it where config.py expects it (data/raw/kmu_fed)
input_root = Path("/kaggle/input")
kmu_root = find_dir_containing(input_root, "kmu")
if kmu_root is None:
    raise SystemExit(
        "Could not find a 'kmu'-named folder under /kaggle/input. Add the KMU-FED dataset as "
        "an input first (see the docstring at the top of this file)."
    )
print(f"KMU-FED found at {kmu_root} ({len(os.listdir(kmu_root))} files)")

os.makedirs("data/raw", exist_ok=True)
kmu_link = Path("data/raw/kmu_fed")
if kmu_link.is_symlink() or kmu_link.exists():
    (os.remove if kmu_link.is_symlink() else shutil.rmtree)(kmu_link)
os.symlink(kmu_root.resolve(), kmu_link.resolve())

# ---- 4. Locate cnn_v1.keras / vgg16_v1.keras and copy them into models/
cnn_src = find_file(input_root, "cnn_v1.keras")
vgg_src = find_file(input_root, "vgg16_v1.keras")
if cnn_src is None or vgg_src is None:
    raise SystemExit(
        "Could not find cnn_v1.keras and/or vgg16_v1.keras under /kaggle/input. Upload them "
        "as a private Kaggle Dataset and add it as an input first (see the docstring above)."
    )
os.makedirs("models", exist_ok=True)
shutil.copy(cnn_src, "models/cnn_v1.keras")
shutil.copy(vgg_src, "models/vgg16_v1.keras")
print(f"Models ready: {cnn_src} -> models/cnn_v1.keras, {vgg_src} -> models/vgg16_v1.keras")

# ---- 5. Step 1: domain-gap baseline (both models, unchanged)
print("\n" + "=" * 70)
print("STEP 1: domain-gap evaluation (cnn_v1/vgg16_v1, unchanged, on KMU-FED test subjects)")
print("=" * 70)
subprocess.run([sys.executable, "-m", "src.evaluate", "--dataset", "kmu_fed"], check=True)

# ---- 6. Step 2: fine-tune both models
print("\n" + "=" * 70)
print("STEP 2a: fine-tuning custom_cnn on KMU-FED")
print("=" * 70)
subprocess.run(
    [sys.executable, "-m", "src.train_kmu_fed_finetune", "--model", "custom_cnn",
     "--base-model-path", "models/cnn_v1.keras"],
    check=True,
)

print("\n" + "=" * 70)
print("STEP 2b: fine-tuning vgg16 on KMU-FED")
print("=" * 70)
subprocess.run(
    [sys.executable, "-m", "src.train_kmu_fed_finetune", "--model", "vgg16",
     "--base-model-path", "models/vgg16_v1.keras"],
    check=True,
)

# ---- 7. Before/after summary
import pandas as pd  # noqa: E402 - after the pip install above

print("\n" + "=" * 70)
print("SUMMARY: FER2013 test accuracy -> KMU-FED domain gap -> after fine-tuning")
print("=" * 70)
gap = pd.read_csv("results/kmu_fed_domain_gap.csv")
print(gap[["model", "fer2013_test_accuracy", "kmu_fed_accuracy", "accuracy_drop"]].to_string(index=False))
for name in ("cnn_kmufed_ft_v1", "vgg16_kmufed_ft_v1"):
    with open(f"results/{name}_kmu_fed_test_metrics.json") as f:
        m = json.load(f)
    print(f"{name}: fine-tuned KMU-FED test accuracy = {m['accuracy']:.4f} "
          f"(macro_f1={m['macro_f1']:.4f}, n={m['n_test_evaluated']}, "
          f"{m['n_test_no_face']} images had no face detected)")

# ---- 8. Zip everything worth bringing back
subprocess.run(
    "cd .. && zip -r kmu_fed_results.zip "
    "driver-emotion/models/cnn_kmufed_ft_v1.keras driver-emotion/models/vgg16_kmufed_ft_v1.keras "
    "driver-emotion/results/kmu_fed_domain_gap.csv "
    "driver-emotion/results/cnn_kmufed_ft_v1_kmu_fed_test_metrics.json "
    "driver-emotion/results/vgg16_kmufed_ft_v1_kmu_fed_test_metrics.json "
    "driver-emotion/results/cnn_kmufed_ft_v1_history.csv "
    "driver-emotion/results/vgg16_kmufed_ft_v1_history.csv "
    "driver-emotion/results/runs.csv "
    "driver-emotion/results/figures/kmu_fed_confusion_*.png "
    "driver-emotion/results/figures/cnn_kmufed_ft_v1_test_confusion_matrix*.png "
    "driver-emotion/results/figures/vgg16_kmufed_ft_v1_test_confusion_matrix*.png",
    shell=True,
)
print("\nWrote /kaggle/working/kmu_fed_results.zip - download it from the notebook's Output pane.")
