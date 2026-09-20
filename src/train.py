"""Shared training entry point for both models.

Run as a module from the repo root, e.g.:
    python -m src.train --model custom_cnn --subset 0.02 --epochs 2
"""
import argparse
import time
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

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
from src.models.vgg16_tl import build_vgg16, unfreeze_top
from src.utils import count_params, ensure_dirs, set_seed


def _build_callbacks(run_name: str, monitor: str = "val_accuracy") -> list:
    """Standard callback set shared by every training run.

    CSVLogger uses `append=True` so a run's history survives being written across
    multiple `fit()` calls (needed for vgg16's two training stages); ModelCheckpoint's
    `best` is likewise only reset when a *new* callback instance is created, so reusing
    the same callback list across stages keeps "best across the whole run" correct.
    """
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
        tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / f"{run_name}_history.csv"), append=True),
    ]


def _val_macro_f1(model: tf.keras.Model, val_ds: tf.data.Dataset) -> float:
    """Macro-F1 on the validation set. Accuracy alone hides collapse onto majority classes."""
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(labels.numpy())
    return float(f1_score(np.concatenate(y_true), np.concatenate(y_pred), average="macro", zero_division=0))


def _combine_histories(first: dict, second: dict) -> dict:
    """Concatenate two per-epoch `History.history` dicts (stage 1 + stage 2), key-wise."""
    keys = set(first) | set(second)
    return {key: list(first.get(key, [])) + list(second.get(key, [])) for key in keys}


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
    val_macro_f1: float,
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
        "val_macro_f1": val_macro_f1,
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
        val_macro_f1=_val_macro_f1(model, val_ds),
        params=count_params(model),
        minutes=minutes,
        img_size=img_size,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
    )


def train_vgg16(args: argparse.Namespace) -> None:
    """Train Model B (VGG16) in two stages: frozen ImageNet base, then fine-tune the top block.

    `--head-epochs` / `--finetune-epochs`, if given, override `cfg["head_epochs"]` /
    `cfg["finetune_epochs"]` independently (used for quick smoke tests that exercise
    both stages without paying for a full run). The same callbacks are reused across
    both `fit()` calls so checkpointing tracks the best epoch across the whole run.
    """
    cfg = dict(VGG16)
    img_size = args.img_size or cfg["img_size"]
    head_epochs = cfg["head_epochs"] if args.head_epochs is None else args.head_epochs
    finetune_epochs = cfg["finetune_epochs"] if args.finetune_epochs is None else args.finetune_epochs

    train_ds, val_ds, _test_ds, class_weights = get_datasets(
        "vgg16", subset=args.subset, img_size=img_size
    )
    model = build_vgg16(cfg=cfg, img_size=img_size)
    callbacks = _build_callbacks(args.run_name)

    # Stage 1: frozen ImageNet base.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["head_lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    start = time.time()
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Stage 2: unfreeze the top block and fine-tune at a lower learning rate.
    unfreeze_top(model, cfg["unfreeze_from"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["finetune_lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs + finetune_epochs,
        initial_epoch=head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    minutes = (time.time() - start) / 60

    combined = _combine_histories(history_head.history, history_finetune.history)
    _save_curves(combined, args.run_name, finetune_start=head_epochs)
    _log_run(
        run_name=args.run_name,
        model_name="vgg16",
        epochs_run=len(combined["loss"]),
        best_val_acc=max(combined["val_accuracy"]),
        val_macro_f1=_val_macro_f1(model, val_ds),
        params=count_params(model),
        minutes=minutes,
        img_size=img_size,
        batch_size=cfg["batch_size"],
        lr=cfg["finetune_lr"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI shared by both models."""
    parser = argparse.ArgumentParser(description="Train custom_cnn or vgg16 on FER2013.")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16"], required=True)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of each split to use.")
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg epochs (custom_cnn).")
    parser.add_argument(
        "--head-epochs", dest="head_epochs", type=int, default=None,
        help="Override cfg['head_epochs'] (vgg16 stage 1, frozen base).",
    )
    parser.add_argument(
        "--finetune-epochs", dest="finetune_epochs", type=int, default=None,
        help="Override cfg['finetune_epochs'] (vgg16 stage 2, fine-tune).",
    )
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
