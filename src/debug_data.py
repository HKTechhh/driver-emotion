"""Debug script (not part of the pipeline): inspect raw train/val batches for custom_cnn
before touching any training code. Checks pixel stats, label distribution, and whether
labels visually match the faces.

Run as a module from the repo root:
    python -m src.debug_data
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, CNN, FER_DIR, FIGURES_DIR, SEED, VAL_SPLIT
from src.data import get_datasets
from src.preprocess import preprocess_custom_cnn
from src.utils import ensure_dirs, set_seed


def load_raw_batch(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """One fixed, preprocessed (but NOT augmented) custom_cnn batch. `split` is "training" or "validation"."""
    ds = tf.keras.utils.image_dataset_from_directory(
        FER_DIR / "train",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(CNN["img_size"], CNN["img_size"]),
        batch_size=CNN["batch_size"],
        validation_split=VAL_SPLIT,
        subset=split,
        seed=SEED,
    )
    images, labels = next(iter(ds))
    return preprocess_custom_cnn(images).numpy(), labels.numpy()


def _print_batch_stats(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    print(f"--- {name} ---")
    print(f"dtype={images.dtype} shape={images.shape}")
    print(f"min={images.min():.4f} max={images.max():.4f} mean={images.mean():.4f} std={images.std():.4f}")
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    print("label counts:", dict(zip(CLASS_NAMES, counts.tolist())))
    print()


def _save_grid(images: np.ndarray, labels: np.ndarray, out_path, n: int = 16) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n and i < len(images):
            ax.imshow(images[i, ..., 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(CLASS_NAMES[labels[i]], fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()

    train_images, train_labels = load_raw_batch("training")
    val_images, val_labels = load_raw_batch("validation")
    _print_batch_stats("train batch (augmentation OFF)", train_images, train_labels)
    _print_batch_stats("val batch", val_images, val_labels)
    _save_grid(train_images, train_labels, FIGURES_DIR / "debug_batch.png")
    print(f"Wrote {FIGURES_DIR / 'debug_batch.png'}")

    aug_train_ds, _, _, _ = get_datasets("custom_cnn", subset=1.0)
    aug_images, aug_labels = next(iter(aug_train_ds))
    aug_images, aug_labels = aug_images.numpy(), aug_labels.numpy()
    print()
    _print_batch_stats("train batch (augmentation ON)", aug_images, aug_labels)
    _save_grid(aug_images, aug_labels, FIGURES_DIR / "debug_batch_aug.png")
    print(f"Wrote {FIGURES_DIR / 'debug_batch_aug.png'}")


if __name__ == "__main__":
    main()
