"""Is the CNN-vs-VGG16 gap real? Per-image predictions, bootstrap CIs and a McNemar test.

Predicts on the full FER2013 test set through the same tf.data pipeline as src/evaluate.py
(fixed, unshuffled order, so both models see the images in the same order), saves the
per-image predictions, then computes:
- 95% bootstrap confidence intervals for accuracy and macro-F1 of each model,
- a *paired* bootstrap CI for the difference (same resampled images for both models),
- an exact McNemar test on the images the two models disagree about.

Run as a module from the repo root:
    python -m src.significance --cnn-model-path models/cnn_v1.keras --vgg16-model-path models/vgg16_v1.keras
"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from config import NUM_CLASSES, RESULTS_DIR, SEED
from src.data import get_datasets
from src.evaluate import _predict
from src.utils import ensure_dirs, set_seed


def _predictions(model_name: str, model_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(y_true, y_pred) on the full test set; cached in results/predictions_{run}.csv."""
    run_name = Path(model_path).stem
    cache = RESULTS_DIR / f"predictions_{run_name}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        return df["true"].to_numpy(), df["pred"].to_numpy()
    _, _, test_ds, _ = get_datasets(model_name, subset=1.0)
    y_true, y_pred = _predict(tf.keras.models.load_model(model_path), test_ds)
    pd.DataFrame({"true": y_true, "pred": y_pred}).to_csv(cache, index_label="test_index")
    return y_true, y_pred


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0))


def _ci(values: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def analyse(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Paired comparison of model A (CNN) and model B (VGG16) on the same images."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    acc_a, acc_b, f1_a, f1_b = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        acc_a.append((pred_a[idx] == y_true[idx]).mean())
        acc_b.append((pred_b[idx] == y_true[idx]).mean())
        f1_a.append(_macro_f1(y_true[idx], pred_a[idx]))
        f1_b.append(_macro_f1(y_true[idx], pred_b[idx]))
    acc_a, acc_b, f1_a, f1_b = map(np.array, (acc_a, acc_b, f1_a, f1_b))

    right_a, right_b = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    p_value = float(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    return {
        "n_test": n,
        "bootstrap_resamples": n_boot,
        "cnn": {"accuracy": float(right_a.mean()), "accuracy_ci95": _ci(acc_a),
                "macro_f1": _macro_f1(y_true, pred_a), "macro_f1_ci95": _ci(f1_a)},
        "vgg16": {"accuracy": float(right_b.mean()), "accuracy_ci95": _ci(acc_b),
                  "macro_f1": _macro_f1(y_true, pred_b), "macro_f1_ci95": _ci(f1_b)},
        "accuracy_diff_cnn_minus_vgg16": float(right_a.mean() - right_b.mean()),
        "accuracy_diff_ci95": _ci(acc_a - acc_b),
        "macro_f1_diff_cnn_minus_vgg16": _macro_f1(y_true, pred_a) - _macro_f1(y_true, pred_b),
        "macro_f1_diff_ci95": _ci(f1_a - f1_b),
        "both_right": int((right_a & right_b).sum()),
        "only_cnn_right": only_a,
        "only_vgg16_right": only_b,
        "both_wrong": int((~right_a & ~right_b).sum()),
        "mcnemar_exact_p_two_sided": p_value,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap CIs and McNemar test for CNN vs VGG16.")
    parser.add_argument("--cnn-model-path", dest="cnn_model_path", type=str, required=True)
    parser.add_argument("--vgg16-model-path", dest="vgg16_model_path", type=str, required=True)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=2000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    y_true_a, pred_a = _predictions("custom_cnn", args.cnn_model_path)
    y_true_b, pred_b = _predictions("vgg16", args.vgg16_model_path)
    assert (y_true_a == y_true_b).all(), "the two models saw the test images in different orders"

    result = analyse(y_true_a, pred_a, pred_b, n_boot=args.n_boot)
    out = RESULTS_DIR / "significance.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
