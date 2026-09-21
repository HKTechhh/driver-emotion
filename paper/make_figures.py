"""Extra figures for the paper, generated from the result files (nothing is typed in by hand).

    .venv/bin/python paper/make_figures.py

Writes pipeline.png, per_class_f1_recall.png, robustness_retained.png and gradcam_combined.png to
results/figures/ and paper/figures/.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CNN_C, VGG_C = "#1f77b4", "#ff7f0e"


def save(fig, name: str) -> None:
    for folder in ("results/figures", "paper/figures"):
        fig.savefig(ROOT / folder / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def box(ax, xy, w, h, text, colour, size=8.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=colour, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, wrap=True)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#333"))


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.9); ax.axis("off")
    ax.text(0.0, 4.65, "A. Training and evaluation (offline, GPU for training, laptop CPU for evaluation)", fontsize=10, weight="bold")
    train = [("FER2013\n28,709 train\n7,178 test", "#e8e8e8"), ("Split\n24,403 train\n4,306 val", "#e8e8e8"),
             ("Augment\n(raw pixels) +\nshared\npreprocessing", "#dbe9f6"), ("Model A: CNN\nModel B: VGG16\n(2-stage\ntransfer)", "#fde6cf"),
             ("Evaluation:\nacc, macro-F1,\nlatency,\nMcNemar", "#dff0d8"), ("Robustness:\n6 driving\nconditions", "#dff0d8"), ("Grad-CAM:\ncorrect vs.\nwrong", "#dff0d8")]
    w, gap, y = 1.32, 0.28, 3.0
    for i, (t, c) in enumerate(train):
        x = 0.05 + i * (w + gap)
        box(ax, (x, y), w, 1.35, t, c, size=8)
        if i:
            arrow(ax, (x - gap + 0.02, y + 0.67), (x - 0.02, y + 0.67))
    ax.text(0.0, 2.5, "B. Real-time driver application (src/realtime.py, on-device, no images stored)", fontsize=10, weight="bold")
    live = [("Camera\nframe", "#e8e8e8"), ("Face detection\n(BlazeFace;\nHaar fallback)", "#dbe9f6"), ("Crop largest\nface, +15%\npadding", "#dbe9f6"),
            ("Same\npreprocessing\nas training", "#dbe9f6"), ("Trained model\n(from A; CNN)", "#fde6cf"), ("10-frame\nmoving\naverage", "#dff0d8"), ("Overlay; alert if\nangry/fear > 3 s;\nCSV log", "#f6d8d8")]
    y2 = 0.85
    for i, (t, c) in enumerate(live):
        x = 0.05 + i * (w + gap)
        box(ax, (x, y2), w, 1.35, t, c, size=8)
        if i:
            arrow(ax, (x - gap + 0.02, y2 + 0.67), (x - 0.02, y2 + 0.67))
    ax.text(5.5, 0.35, "The preprocessing module (src/preprocess.py) is shared by both rows, so a live frame is transformed exactly like a training image.",
            ha="center", fontsize=8.5, style="italic")
    save(fig, "pipeline.png")


def per_class() -> None:
    c = json.load(open(ROOT / "results/cnn_v1_metrics.json"))["classification_report"]
    v = json.load(open(ROOT / "results/vgg16_v1_metrics.json"))["classification_report"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x, w = np.arange(len(CLASSES)), 0.38
    for ax, key, title in zip(axes, ("f1-score", "recall"), ("F1 per class", "Recall per class")):
        a = [c[k][key] for k in CLASSES]; b = [v[k][key] for k in CLASSES]
        ax.bar(x - w / 2, a, w, label="Custom CNN", color=CNN_C); ax.bar(x + w / 2, b, w, label="VGG16", color=VGG_C)
        for xi, (ai, bi) in enumerate(zip(a, b)):
            ax.text(xi - w / 2, ai + 0.01, f"{ai:.2f}", ha="center", fontsize=7); ax.text(xi + w / 2, bi + 0.01, f"{bi:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=25); ax.set_title(title); ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score (test set, n = 7,178)"); axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "per_class_f1_recall.png")


def retained() -> None:
    d = pd.read_csv(ROOT / "results/robustness.csv").set_index("condition")
    conds = [k for k in d.index if k != "clean"]
    a = [100 * d.loc[k, "custom_cnn_accuracy"] / d.loc["clean", "custom_cnn_accuracy"] for k in conds]
    b = [100 * d.loc[k, "vgg16_accuracy"] / d.loc["clean", "vgg16_accuracy"] for k in conds]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x, w = np.arange(len(conds)), 0.38
    ax.bar(x - w / 2, a, w, label="Custom CNN", color=CNN_C); ax.bar(x + w / 2, b, w, label="VGG16", color=VGG_C)
    for xi, (ai, bi) in enumerate(zip(a, b)):
        ax.text(xi - w / 2, ai + 1, f"{ai:.0f}%", ha="center", fontsize=8); ax.text(xi + w / 2, bi + 1, f"{bi:.0f}%", ha="center", fontsize=8)
    ax.axhline(100, color="#555", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([k.replace("_", " ") for k in conds]); ax.set_ylim(0, 122)
    ax.set_ylabel("Accuracy retained (% of own clean accuracy)"); ax.set_title("How much of its clean accuracy each model keeps"); ax.legend(loc="upper center", ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "robustness_retained.png")


def gradcam_combined() -> None:
    """Correct-example grid (tall, left) beside the CNN and VGG16 error grids (stacked, right), from the saved figures."""
    from PIL import Image
    fig_dir = ROOT / "results/figures"
    grid = Image.open(fig_dir / "gradcam_grid.png").convert("RGB")
    errs = [Image.open(fig_dir / f"gradcam_misclassified_{m}.png").convert("RGB") for m in ("custom_cnn", "vgg16")]
    right_w = 1770
    errs = [e.resize((right_w, round(e.height * right_w / e.width))) for e in errs]
    total_h = sum(e.height for e in errs)
    grid = grid.resize((round(grid.width * total_h / grid.height), total_h))
    canvas = Image.new("RGB", (grid.width + 30 + right_w, total_h), "white")
    canvas.paste(grid, (0, 0))
    y = 0
    for e in errs:
        canvas.paste(e, (grid.width + 30, y)); y += e.height
    for folder in ("results/figures", "paper/figures"):
        canvas.save(ROOT / folder / "gradcam_combined.png")
    print("wrote gradcam_combined.png", canvas.size)


if __name__ == "__main__":
    pipeline(); per_class(); retained(); gradcam_combined()
