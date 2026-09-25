"""Evaluate a trained model on the FER2013 test set and save metrics for comparison.

Run as a module from the repo root, e.g.:
    python -m src.evaluate --model custom_cnn --model-path models/cnn_v1.keras

`--dataset kmu_fed` (no --model/--model-path needed) instead runs the KMU-FED domain-gap
check: loads models/cnn_v1.keras and models/vgg16_v1.keras unchanged and evaluates both on
KMU-FED's held-out test subjects (config.KMU_FED). See _run_kmu_fed_domain_gap below.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, FIGURES_DIR, KMU_FED, MODELS_DIR, NUM_CLASSES, REALTIME, RESULTS_DIR, SEED
from src.data import get_datasets
from src.kmu_fed_data import parse_filename
from src.realtime import build_face_detector, load_model_for_inference, predict_face
from src.utils import count_params, ensure_dirs, set_seed

# KMU-FED has no "neutral" class; its 6 codes correspond 1:1, in the same alphabetical order,
# to CLASS_NAMES minus "neutral" - AN=angry, DI=disgust, FE=fear, HA=happy, SA=sad, SU=surprise.
KMU_CODE_TO_FER_NAME = {"AN": "angry", "DI": "disgust", "FE": "fear", "HA": "happy", "SA": "sad", "SU": "surprise"}
SIX_CLASS_NAMES = [c for c in CLASS_NAMES if c != "neutral"]
NEUTRAL_INDEX = CLASS_NAMES.index("neutral")


def _predict(model: tf.keras.Model, test_ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Run `model` over `test_ds` and return (y_true, y_pred) as flat label arrays."""
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _save_confusion_matrices(
    y_true: np.ndarray, y_pred: np.ndarray, run_name: str,
    class_names: List[str] = CLASS_NAMES, num_classes: int = NUM_CLASSES,
) -> None:
    """Save raw-count and row-normalised confusion matrix heatmaps as PNGs.

    `class_names`/`num_classes` default to the FER2013 globals (every existing caller's
    behaviour is unchanged); `_run_kmu_fed_domain_gap` below passes the 6-class KMU-FED subset.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
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
            xticklabels=class_names,
            yticklabels=class_names,
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


def _renormalize_drop_neutral(probs7: np.ndarray) -> np.ndarray:
    """Drop the "neutral" logit (KMU-FED has no such class) and renormalize the remaining 6
    class probabilities over their own sum, so a model isn't penalised for a class the test
    set can't contain - it's simply excluded from consideration, not scored as always wrong.
    """
    six = np.delete(probs7, NEUTRAL_INDEX)
    total = six.sum()
    return six / total if total > 0 else six


def _evaluate_kmu_fed_model(model_name: str, model_path: Path, test_subjects: List[int], detect_fn) -> Dict:
    """Run one model over KMU-FED's held-out test-subject images.

    KMU-FED ships full driver-cabin photos, not pre-cropped faces like FER2013, so this runs
    the same detect->crop->preprocess->predict path as the live app (`src.realtime.predict_face`)
    on each one, rather than a tf.data pipeline of pre-cropped images.
    """
    model, img_size = load_model_for_inference(str(model_path))
    kmu_dir = Path(KMU_FED["dir"])
    files = sorted(f for f in kmu_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))

    y_true, y_pred = [], []
    n_total = n_no_face = 0
    for f in files:
        subject_id, cls_code, _initials, _index = parse_filename(f.name)
        if subject_id not in test_subjects:
            continue
        n_total += 1

        image_bgr = cv2.imread(str(f))
        if image_bgr is None:
            n_no_face += 1  # unreadable file counts the same way as "no face found"
            continue
        result = predict_face(image_bgr, detect_fn, model, model_name, img_size, REALTIME["face_padding"])
        if result is None:
            n_no_face += 1
            continue

        six_probs = _renormalize_drop_neutral(result["probs"])
        pred_name = SIX_CLASS_NAMES[int(np.argmax(six_probs))]
        true_name = KMU_CODE_TO_FER_NAME[cls_code]
        y_true.append(SIX_CLASS_NAMES.index(true_name))
        y_pred.append(SIX_CLASS_NAMES.index(pred_name))

    y_true_arr, y_pred_arr = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    n_evaluated = len(y_true_arr)
    report = classification_report(
        y_true_arr, y_pred_arr, labels=range(len(SIX_CLASS_NAMES)), target_names=SIX_CLASS_NAMES,
        output_dict=True, zero_division=0,
    ) if n_evaluated else None

    return {
        "model": model_name,
        "model_path": str(model_path),
        "n_test_images_total": n_total,
        "n_no_face_detected": n_no_face,
        "n_evaluated": n_evaluated,
        "accuracy": report["accuracy"] if report else float("nan"),
        "macro_f1": report["macro avg"]["f1-score"] if report else float("nan"),
        "weighted_f1": report["weighted avg"]["f1-score"] if report else float("nan"),
        "classification_report": report,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    }


def _fer2013_test_accuracy(run_name: str) -> float:
    """This run's FER2013 test accuracy, for the domain-gap comparison line.

    Reads results/comparison.csv (real test-set accuracy) rather than results/runs.csv (which
    only has best *validation* accuracy) - comparing test-vs-test is the apples-to-apples number.
    Falls back to runs.csv's val accuracy, clearly labelled, if comparison.csv isn't there yet.
    """
    comparison_path = RESULTS_DIR / "comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        row = df[df["run_name"] == run_name]
        if not row.empty:
            return float(row.iloc[0]["accuracy"])
    runs_path = RESULTS_DIR / "runs.csv"
    if runs_path.exists():
        df = pd.read_csv(runs_path)
        row = df[df["run_name"] == run_name]
        if not row.empty:
            print(f"  (comparison.csv missing '{run_name}' - falling back to runs.csv's "
                  f"*validation* accuracy, not test accuracy)")
            return float(row.iloc[0]["best_val_acc"])
    return float("nan")


def _run_kmu_fed_domain_gap() -> None:
    """Evaluate models/cnn_v1.keras and models/vgg16_v1.keras, unchanged, on KMU-FED's
    held-out test subjects (config.KMU_FED["test_subjects"]). Writes
    results/kmu_fed_domain_gap.csv and results/figures/kmu_fed_confusion_{model}*.png.
    """
    detect_fn = build_face_detector(REALTIME["min_face_confidence"])
    test_subjects = KMU_FED["test_subjects"]
    models = {"custom_cnn": MODELS_DIR / "cnn_v1.keras", "vgg16": MODELS_DIR / "vgg16_v1.keras"}

    rows = []
    for model_name, model_path in models.items():
        print(f"\n=== {model_name} ({model_path}) on KMU-FED test subjects {test_subjects} ===")
        result = _evaluate_kmu_fed_model(model_name, model_path, test_subjects, detect_fn)
        print(f"  {result['n_test_images_total']} test images, {result['n_no_face_detected']} with no face "
              f"detected, {result['n_evaluated']} evaluated")
        print(f"  accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
              f"(renormalized over the 6 shared classes - no 'neutral' in KMU-FED)")

        run_name = "cnn_v1" if model_name == "custom_cnn" else "vgg16_v1"
        fer2013_acc = _fer2013_test_accuracy(run_name)
        drop = fer2013_acc - result["accuracy"] if not np.isnan(fer2013_acc) else float("nan")
        print(f"  FER2013 test accuracy was {fer2013_acc:.4f} -> KMU-FED domain-gap accuracy "
              f"{result['accuracy']:.4f} (dropped {drop:.4f}, i.e. {drop * 100:.1f} points)")

        if result["n_evaluated"]:
            _save_confusion_matrices(
                result["y_true"], result["y_pred"], f"kmu_fed_confusion_{model_name}",
                class_names=SIX_CLASS_NAMES, num_classes=len(SIX_CLASS_NAMES),
            )

        rows.append({
            "model": model_name,
            "n_test_images_total": result["n_test_images_total"],
            "n_no_face_detected": result["n_no_face_detected"],
            "n_evaluated": result["n_evaluated"],
            "kmu_fed_accuracy": result["accuracy"],
            "kmu_fed_macro_f1": result["macro_f1"],
            "kmu_fed_weighted_f1": result["weighted_f1"],
            "fer2013_test_accuracy": fer2013_acc,
            "accuracy_drop": drop,
        })

    out_path = RESULTS_DIR / "kmu_fed_domain_gap.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for evaluating a single trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the FER2013 test set.")
    parser.add_argument("--dataset", choices=["fer2013", "kmu_fed"], default="fer2013")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16"], required=False)
    parser.add_argument("--model-path", dest="model_path", type=str, required=False)
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
    """Evaluate one model on the test set and write results/{run_name}_metrics.json.

    `--dataset kmu_fed` instead runs `_run_kmu_fed_domain_gap` (no --model/--model-path
    needed - it always loads models/cnn_v1.keras and models/vgg16_v1.keras, per the task).
    """
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    if args.dataset == "kmu_fed":
        _run_kmu_fed_domain_gap()
        return

    if not args.model or not args.model_path:
        raise SystemExit("--model and --model-path are required for --dataset fer2013 (the default).")

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
