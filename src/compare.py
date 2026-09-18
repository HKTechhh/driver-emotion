"""Aggregate every run's metrics JSON into a single comparison table and chart.

Run as a module from the repo root:
    python -m src.compare
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, RESULTS_DIR
from src.utils import ensure_dirs

COLUMNS = [
    "run_name",
    "model",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "params",
    "model_size_mb",
    "latency_mean_ms",
    "latency_p95_ms",
    "fps",
]


def _load_metrics(results_dir: Path) -> pd.DataFrame:
    """Read every {run_name}_metrics.json in `results_dir` into one DataFrame."""
    rows = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with open(path) as f:
            data = json.load(f)
        rows.append({col: data.get(col) for col in COLUMNS})
    return pd.DataFrame(rows, columns=COLUMNS)


def _save_comparison_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: accuracy and macro-F1 on the left axis, FPS on the right."""
    labels = df["run_name"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width, df["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax1.bar(x, df["macro_f1"], width, label="Macro-F1", color="#55A868")
    ax1.set_ylabel("Accuracy / Macro-F1")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + width, df["fps"], width, label="FPS", color="#C44E52")
    ax2.set_ylabel("FPS")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Model comparison: accuracy, macro-F1 and inference speed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the comparison aggregation step."""
    parser = argparse.ArgumentParser(description="Aggregate metrics JSON files into a comparison table.")
    parser.add_argument(
        "--results-dir", dest="results_dir", type=str, default=str(RESULTS_DIR), help="Where *_metrics.json live."
    )
    return parser


def main() -> None:
    """Read every metrics JSON, write results/comparison.csv and results/figures/comparison.png."""
    args = build_arg_parser().parse_args()
    ensure_dirs()

    df = _load_metrics(Path(args.results_dir))
    if df.empty:
        raise SystemExit(f"No *_metrics.json files found in {args.results_dir}. Run src.evaluate first.")

    comparison_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(comparison_path, index=False)
    _save_comparison_chart(df, FIGURES_DIR / "comparison.png")

    print(f"Wrote {comparison_path} ({len(df)} rows)")
    print(df[["run_name", "model", "accuracy", "macro_f1", "fps"]].to_string(index=False))


if __name__ == "__main__":
    main()
