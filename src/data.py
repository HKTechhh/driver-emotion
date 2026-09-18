"""tf.data pipelines for FER2013, shared by training and evaluation for both models."""
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import CLASS_NAMES, CNN, FER_DIR, NUM_CLASSES, SEED, VAL_SPLIT, VGG16

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


def _safe_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Balanced class weights from `y`. Classes missing from `y` (e.g. a tiny --subset) get 1.0."""
    present = np.unique(y)
    weights = compute_class_weight("balanced", classes=present, y=y)
    by_class = {int(c): float(w) for c, w in zip(present, weights)}
    return {c: by_class.get(c, 1.0) for c in range(num_classes)}


def _preprocess_custom_cnn(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Scale grayscale pixels from [0, 255] to [0, 1]."""
    return tf.cast(images, tf.float32) / 255.0, labels


def _preprocess_vgg16(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convert grayscale to 3-channel and apply Keras' VGG16 preprocessing."""
    images = tf.image.grayscale_to_rgb(images)
    return tf.keras.applications.vgg16.preprocess_input(images), labels


def get_datasets(
    model_name: str, subset: float = 1.0, img_size: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
    """Build (train_ds, val_ds, test_ds, class_weights) for `model_name`.

    `model_name` is one of {"custom_cnn", "vgg16"}. Images are loaded grayscale from
    `config.FER_DIR` and resized to that model's `img_size` (or `img_size` if given,
    e.g. to shrink VGG16 input for a CPU smoke test); class order matches
    `config.CLASS_NAMES`. Training data is augmented; `subset` < 1.0 keeps only that
    fraction of each split, for fast CPU smoke tests. `class_weights` is computed from
    the (post-subset) training labels for use with balanced training.
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

    preprocess = _preprocess_custom_cnn if model_name == "custom_cnn" else _preprocess_vgg16
    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

    if model_name == "custom_cnn":
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    augmentation = _build_augmentation()
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weights
