"""Evaluate a trained model on the FER2013 test set and save metrics for comparison.

Run as a module from the repo root, e.g.:
    python -m src.evaluate --model custom_cnn --model-path models/cnn_v1.keras
"""
import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, FIGURES_DIR, NUM_CLASSES, RESULTS_DIR, SEED
from src.data import get_datasets
from src.utils import count_params, ensure_dirs, set_seed


def _predict(model: tf.keras.Model, test_ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Run `model` over `test_ds` and return (y_true, y_pred) as flat label arrays."""
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, run_name: str) -> None:
    """Save raw-count and row-normalised confusion matrix heatmaps as PNGs."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    for matrix, suffix, fmt, title in (
        (cm, "confusion_matrix", "d", "Confusion matrix (counts)"),
        (cm_norm, "confusion_matrix_normalized", ".2f", "Confusion matrix (row-normalised)"),
    ):
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{run_name} — {title}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{run_name}_{suffix}.png", dpi=200)
        plt.close(fig)


def _measure_latency(model: tf.keras.Model, sample: np.ndarray, warmup: int = 20, runs: int = 200) -> dict:
    """Mean/p95 single-image inference latency in ms, plus implied FPS.

    Calls the model directly (not `.predict()`) to avoid the batch-API overhead that
    would misrepresent frame-by-frame latency in the real-time app (Phase 7).
    """
    batch = sample[np.newaxis, ...]
    for _ in range(warmup):
        model(batch, training=False)

    times_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        model(batch, training=False)
        times_ms.append((time.perf_counter() - start) * 1000)

    times_ms = np.array(times_ms)
    mean_ms = float(times_ms.mean())
    return {
        "latency_mean_ms": mean_ms,
        "latency_p95_ms": float(np.percentile(times_ms, 95)),
        "fps": 1000.0 / mean_ms,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for evaluating a single trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the FER2013 test set.")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16"], required=True)
    parser.add_argument("--model-path", dest="model_path", type=str, required=True)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of the test set to use.")
    parser.add_argument(
        "--img-size", dest="img_size", type=int, default=None, help="Override the model's img_size."
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        default=None,
        help="Defaults to the model file's stem, e.g. cnn_v1.",
    )
    return parser


def main() -> None:
    """Evaluate one model on the test set and write results/{run_name}_metrics.json."""
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    model_path = Path(args.model_path)
    run_name = args.run_name or model_path.stem

    _, _, test_ds, _ = get_datasets(args.model, subset=args.subset, img_size=args.img_size)
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred = _predict(model, test_ds)
    report = classification_report(
        y_true, y_pred, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    _save_confusion_matrices(y_true, y_pred, run_name)

    sample_image, _ = next(iter(test_ds.unbatch().take(1)))
    latency = _measure_latency(model, sample_image.numpy())

    metrics = {
        "run_name": run_name,
        "model": args.model,
        "model_path": str(model_path),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "params": count_params(model),
        "model_size_mb": model_path.stat().st_size / (1024 ** 2),
        **latency,
    }

    metrics_path = RESULTS_DIR / f"{run_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {metrics_path}")
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} fps={metrics['fps']:.1f}")


if __name__ == "__main__":
    main()
