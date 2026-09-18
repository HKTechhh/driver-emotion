"""Shared training entry point for both models.

Run as a module from the repo root, e.g.:
    python -m src.train --model custom_cnn --subset 0.02 --epochs 2
"""
import argparse
import time
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from config import (
    CNN,
    EARLY_STOP_PATIENCE,
    FIGURES_DIR,
    LOGS_DIR,
    LR_FACTOR,
    LR_PATIENCE,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    VGG16,
)
from src.data import get_datasets
from src.models.custom_cnn import build_custom_cnn
from src.utils import count_params, ensure_dirs, set_seed


def _build_callbacks(run_name: str, monitor: str = "val_accuracy") -> list:
    """Standard callback set shared by every training run."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / f"{run_name}.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(LOGS_DIR / run_name)),
        tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / f"{run_name}_history.csv")),
    ]


def _save_curves(history: dict, run_name: str, finetune_start: Optional[int] = None) -> None:
    """Save loss/accuracy curves to results/figures/{run_name}_curves.png."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        if finetune_start is not None:
            ax.axvline(finetune_start, color="gray", linestyle="--", label="fine-tune start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{run_name} — {title}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_curves.png", dpi=200)
    plt.close(fig)


def _log_run(
    run_name: str,
    model_name: str,
    epochs_run: int,
    best_val_acc: float,
    params: int,
    minutes: float,
    img_size: int,
    batch_size: int,
    lr: float,
) -> None:
    """Append one row describing this run to results/runs.csv."""
    row = {
        "run_name": run_name,
        "model": model_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "epochs_run": epochs_run,
        "best_val_acc": best_val_acc,
        "params": params,
        "train_minutes": round(minutes, 2),
        "img_size": img_size,
        "batch_size": batch_size,
        "lr": lr,
    }
    runs_path = RESULTS_DIR / "runs.csv"
    df_row = pd.DataFrame([row])
    df_row.to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)


def train_custom_cnn(args: argparse.Namespace) -> None:
    """Train Model A (custom CNN) and record its results."""
    cfg = dict(CNN)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    img_size = args.img_size or cfg["img_size"]

    train_ds, val_ds, _test_ds, class_weights = get_datasets(
        "custom_cnn", subset=args.subset, img_size=img_size
    )
    model = build_custom_cnn(cfg=cfg, img_size=img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        class_weight=class_weights,
        callbacks=_build_callbacks(args.run_name),
    )
    minutes = (time.time() - start) / 60

    _save_curves(history.history, args.run_name)
    _log_run(
        run_name=args.run_name,
        model_name="custom_cnn",
        epochs_run=len(history.history["loss"]),
        best_val_acc=max(history.history["val_accuracy"]),
        params=count_params(model),
        minutes=minutes,
        img_size=img_size,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
    )


def train_vgg16(args: argparse.Namespace) -> None:
    """Train Model B (VGG16 transfer learning). Implemented in Phase 3."""
    raise NotImplementedError("train_vgg16 is implemented in Phase 3.")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI shared by both models."""
    parser = argparse.ArgumentParser(description="Train custom_cnn or vgg16 on FER2013.")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16"], required=True)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of each split to use.")
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg epochs.")
    parser.add_argument(
        "--img-size", dest="img_size", type=int, default=None, help="Override cfg img_size."
    )
    parser.add_argument(
        "--run-name", dest="run_name", type=str, default=None, help="Defaults to {model}_{timestamp}."
    )
    return parser


def main() -> None:
    """Parse args, set up the environment, and dispatch to the chosen model's trainer."""
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.model}_{timestamp}"

    if args.model == "custom_cnn":
        train_custom_cnn(args)
    else:
        train_vgg16(args)


if __name__ == "__main__":
    main()
