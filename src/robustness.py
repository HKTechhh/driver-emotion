"""Evaluate both trained models under simulated "urban traffic" driving conditions.

Run as a module from the repo root, e.g.:
    python -m src.robustness --cnn-model-path models/cnn_v1.keras --vgg16-model-path models/vgg16_v1.keras
"""
import argparse
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report

from config import CLASS_NAMES, FER_DIR, FIGURES_DIR, NUM_CLASSES, RESULTS_DIR, ROBUSTNESS_CONDITIONS, SEED
from src.utils import ensure_dirs, set_seed


def _load_raw_test_images(subset: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load FER2013 test images as (N, 48, 48) uint8 grayscale, labels matching CLASS_NAMES order.

    `subset` < 1.0 keeps the first `subset` fraction of *each class* (not the flat list),
    so every class stays represented even in a quick CPU smoke test.
    """
    images, labels = [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        paths = sorted((FER_DIR / "test" / cls).glob("*"))
        if subset < 1.0:
            paths = paths[: max(1, round(len(paths) * subset))]
        for path in paths:
            images.append(np.array(Image.open(path).convert("L"), dtype=np.uint8))
            labels.append(label_idx)
    return np.stack(images), np.array(labels)


def _apply_clean(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """No corruption; the baseline condition."""
    return img.copy()


def _apply_low_light(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dim as if in a night drive or tunnel: brightness x0.4 plus gaussian noise (sigma=10)."""
    out = img.astype(np.float32) * 0.4 + rng.normal(0, 10, size=img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sun glare through a windshield: brightness x1.6 plus a random bright elliptical blob."""
    h, w = img.shape
    out = np.clip(img.astype(np.float32) * 1.6, 0, 255)

    mask = np.zeros((h, w), dtype=np.float32)
    center = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
    axes = (int(rng.integers(w // 6, w // 3)), int(rng.integers(h // 6, h // 3)))
    angle = float(rng.integers(0, 180))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    if mask.max() > 0:
        mask = mask / mask.max()

    return np.clip(out + mask * 120, 0, 255).astype(np.uint8)


def _apply_motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vibration/vehicle motion: horizontal motion blur, kernel size 7. Deterministic, no rng needed."""
    kernel = np.zeros((7, 7), dtype=np.float32)
    kernel[3, :] = 1.0 / 7
    return cv2.filter2D(img, -1, kernel)


def _apply_occlusion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A hand or sunglasses covering part of the face: a random black rectangle over ~20% of it."""
    h, w = img.shape
    out = img.copy()
    rect_w, rect_h = int(w * np.sqrt(0.2)), int(h * np.sqrt(0.2))
    x = int(rng.integers(0, max(1, w - rect_w)))
    y = int(rng.integers(0, max(1, h - rect_h)))
    out[y : y + rect_h, x : x + rect_w] = 0
    return out


def _apply_head_pose(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A driver glancing away: random rotation within +/-20 degrees."""
    h, w = img.shape
    angle = float(rng.uniform(-20, 20))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


CONDITION_FUNCS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "clean": _apply_clean,
    "low_light": _apply_low_light,
    "glare": _apply_glare,
    "motion_blur": _apply_motion_blur,
    "occlusion": _apply_occlusion,
    "head_pose": _apply_head_pose,
}


def _corrupt_batch(images: np.ndarray, condition: str) -> np.ndarray:
    """Apply `condition`'s corruption to every image, deterministically (seeded)."""
    rng = np.random.default_rng(SEED)
    func = CONDITION_FUNCS[condition]
    return np.stack([func(img, rng) for img in images])


def _preprocess_for_model(images: np.ndarray, model_name: str, img_size: int) -> np.ndarray:
    """Resize/convert corrupted uint8 grayscale images to one model's expected input, matching src/data.py.

    `img_size` is read from the loaded model's own input shape (see `_evaluate_condition`)
    rather than from config, so this stays correct for a model trained with `--img-size`.
    """
    resized = np.stack(
        [cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA) for img in images]
    )

    if model_name == "custom_cnn":
        return (resized.astype(np.float32) / 255.0)[..., np.newaxis]

    rgb = np.repeat(resized[..., np.newaxis], 3, axis=-1).astype(np.float32)
    return np.asarray(tf.keras.applications.vgg16.preprocess_input(rgb))


def _evaluate_condition(
    model: tf.keras.Model, model_name: str, images: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """Return (accuracy, macro_f1) for `model` on an already-corrupted image batch."""
    img_size = model.input_shape[1]
    x = _preprocess_for_model(images, model_name, img_size)
    preds = np.argmax(model.predict(x, batch_size=64, verbose=0), axis=1)
    report = classification_report(labels, preds, labels=range(NUM_CLASSES), output_dict=True, zero_division=0)
    return report["accuracy"], report["macro avg"]["f1-score"]


def _save_line_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Line chart of accuracy and macro-F1 vs. condition, for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        for model_name in ("custom_cnn", "vgg16"):
            ax.plot(df["condition"], df[f"{model_name}_{metric}"], marker="o", label=model_name)
        ax.set_xlabel("Condition")
        ax.set_ylabel(title)
        ax.set_title(f"Robustness — {title}")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_conditions_grid(example_images: Dict[str, np.ndarray], out_path: Path) -> None:
    """One example face per condition, so the corruptions can be sanity-checked visually."""
    conditions = list(example_images)
    fig, axes = plt.subplots(1, len(conditions), figsize=(3 * len(conditions), 3.5))
    for ax, condition in zip(axes, conditions):
        ax.imshow(example_images[condition], cmap="gray", vmin=0, vmax=255)
        ax.set_title(condition)
        ax.axis("off")
    fig.suptitle("Robustness conditions (urban traffic scenarios)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the robustness sweep."""
    parser = argparse.ArgumentParser(description="Evaluate both models under simulated driving conditions.")
    parser.add_argument("--cnn-model-path", dest="cnn_model_path", type=str, required=True)
    parser.add_argument("--vgg16-model-path", dest="vgg16_model_path", type=str, required=True)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of each class to use.")
    return parser


def main() -> None:
    """Run every condition against both models and write robustness.csv + figures."""
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    images, labels = _load_raw_test_images(subset=args.subset)
    models = {
        "custom_cnn": tf.keras.models.load_model(args.cnn_model_path),
        "vgg16": tf.keras.models.load_model(args.vgg16_model_path),
    }

    rows = []
    example_images = {}
    for condition in ROBUSTNESS_CONDITIONS:
        corrupted = _corrupt_batch(images, condition)
        example_images[condition] = corrupted[0]

        row = {"condition": condition}
        for model_name, model in models.items():
            accuracy, macro_f1 = _evaluate_condition(model, model_name, corrupted, labels)
            row[f"{model_name}_accuracy"] = accuracy
            row[f"{model_name}_macro_f1"] = macro_f1
        rows.append(row)
        print(f"{condition}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "condition"))

    df = pd.DataFrame(rows)
    robustness_path = RESULTS_DIR / "robustness.csv"
    df.to_csv(robustness_path, index=False)

    _save_line_chart(df, FIGURES_DIR / "robustness.png")
    _save_conditions_grid(example_images, FIGURES_DIR / "conditions_grid.png")

    print(f"Wrote {robustness_path}")


if __name__ == "__main__":
    main()
