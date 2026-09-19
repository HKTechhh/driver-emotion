"""tf.data pipelines for FER2013, shared by training and evaluation for both models."""
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import CLASS_NAMES, CLASS_WEIGHT_MAX, CLASS_WEIGHT_MODE, CNN, FER_DIR, NUM_CLASSES, SEED, VAL_SPLIT, VGG16
from src.preprocess import preprocess_custom_cnn, preprocess_vgg16

AUTOTUNE = tf.data.AUTOTUNE


def _build_augmentation() -> tf.keras.Sequential:
    """Light augmentation applied to the training set only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )


def _take_fraction(ds: tf.data.Dataset, subset: float) -> tf.data.Dataset:
    """Keep only the first `subset` fraction of batches in `ds`."""
    if subset >= 1.0:
        return ds
    n_batches = int(ds.cardinality().numpy())
    if n_batches <= 0:
        return ds
    return ds.take(max(1, round(n_batches * subset)))


def _safe_class_weights(
    y: np.ndarray, num_classes: int, mode: str = CLASS_WEIGHT_MODE, max_weight: float = CLASS_WEIGHT_MAX
) -> Dict[int, float]:
    """Class weights from `y`. Classes missing from `y` (e.g. a tiny --subset) get 1.0.

    `mode` is one of:
    - "none": every class weighted 1.0 (no correction for imbalance).
    - "balanced": sklearn's standard inverse-frequency weighting. On FER2013 this gives
      "disgust" a weight of ~9.5x, which destabilized training (see
      docs/experiment_log.md) -- kept here for reference/comparison, not the default.
    - "sqrt" (default): "balanced" weights, square-rooted and capped at `max_weight`.
      Keeps the imbalance correction without the instability.
    """
    if mode not in ("none", "balanced", "sqrt"):
        raise ValueError(f"Unknown CLASS_WEIGHT_MODE: {mode!r}")
    if mode == "none":
        return {c: 1.0 for c in range(num_classes)}

    present = np.unique(y)
    weights = compute_class_weight("balanced", classes=present, y=y)
    if mode == "sqrt":
        weights = np.minimum(np.sqrt(weights), max_weight)
    by_class = {int(c): float(w) for c, w in zip(present, weights)}
    return {c: by_class.get(c, 1.0) for c in range(num_classes)}


def get_datasets(
    model_name: str, subset: float = 1.0, img_size: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
    """Build (train_ds, val_ds, test_ds, class_weights) for `model_name`.

    `model_name` is one of {"custom_cnn", "vgg16"}. Images are loaded grayscale from
    `config.FER_DIR` and resized to that model's `img_size` (or `img_size` if given,
    e.g. to shrink VGG16 input for a CPU smoke test); class order matches
    `config.CLASS_NAMES`. Training data is augmented; `subset` < 1.0 keeps only that
    fraction of each split, for fast CPU smoke tests. `class_weights` is computed from
    the (post-subset) training labels per `config.CLASS_WEIGHT_MODE`.
    """
    if model_name not in ("custom_cnn", "vgg16"):
        raise ValueError(f"Unknown model_name: {model_name!r}")

    cfg = CNN if model_name == "custom_cnn" else VGG16
    img_size = img_size or cfg["img_size"]
    batch_size = cfg["batch_size"]
    common_kwargs = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        FER_DIR / "train", validation_split=VAL_SPLIT, subset="training", seed=SEED, **common_kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        FER_DIR / "train", validation_split=VAL_SPLIT, subset="validation", seed=SEED, **common_kwargs
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        FER_DIR / "test", shuffle=False, **common_kwargs
    )

    train_ds = _take_fraction(train_ds, subset)
    val_ds = _take_fraction(val_ds, subset)
    test_ds = _take_fraction(test_ds, subset)

    train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    class_weights = _safe_class_weights(train_labels, NUM_CLASSES)

    if model_name == "custom_cnn":
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    # Augment the raw grayscale pixels (still in [0, 255]) BEFORE per-model rescaling.
    # RandomBrightness/RandomContrast default to value_range=(0, 255); running them
    # after custom_cnn's /255.0 scaling (or vgg16's mean-subtracted preprocess_input)
    # would add a shift calibrated for 0-255 onto near-zero-range pixels and wipe out
    # the image (a real bug caught when cnn_v1 converged to ~24% val accuracy, i.e. the
    # model just predicting the majority class -- see docs/experiment_log.md).
    augmentation = _build_augmentation()
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE
    )

    # Shared with src/preprocess.py so live camera frames match training exactly (Phase 7).
    preprocess_fn = preprocess_custom_cnn if model_name == "custom_cnn" else preprocess_vgg16
    train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weights
