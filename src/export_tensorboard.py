"""Write TensorBoard event files from per-epoch history CSVs.

The original TensorBoard logs of the Kaggle training runs were not exported (only the model, the
history CSV and the curves PNG were), so this regenerates them from `results/{run}_history.csv`
using the same tag layout Keras' `TensorBoard` callback uses (`epoch_accuracy`, `epoch_loss`,
`epoch_learning_rate`, under `train/` and `validation/`). The numbers are exactly those in the CSV.

Run as a module from the repo root:
    python -m src.export_tensorboard --runs cnn_v1 vgg16_v1
    tensorboard --logdir logs/paper
"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf

from config import LOGS_DIR, RESULTS_DIR


def export_run(run_name: str, out_root: Path) -> int:
    """Write `train` and `validation` event files for one run; returns the number of epochs written."""
    history = pd.read_csv(RESULTS_DIR / f"{run_name}_history.csv")
    train_writer = tf.summary.create_file_writer(str(out_root / run_name / "train"))
    val_writer = tf.summary.create_file_writer(str(out_root / run_name / "validation"))
    for _, row in history.iterrows():
        step = int(row["epoch"])
        with train_writer.as_default():
            tf.summary.scalar("epoch_accuracy", row["accuracy"], step=step)
            tf.summary.scalar("epoch_loss", row["loss"], step=step)
            tf.summary.scalar("epoch_learning_rate", row["learning_rate"], step=step)
        with val_writer.as_default():
            tf.summary.scalar("epoch_accuracy", row["val_accuracy"], step=step)
            tf.summary.scalar("epoch_loss", row["val_loss"], step=step)
    train_writer.close()
    val_writer.close()
    return len(history)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate TensorBoard logs from history CSVs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run names, e.g. cnn_v1 vgg16_v1.")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=str(LOGS_DIR / "paper"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = Path(args.out_dir)
    runs: List[str] = args.runs
    for run in runs:
        print(f"{run}: {export_run(run, out_root)} epochs written to {out_root / run}")


if __name__ == "__main__":
    main()
