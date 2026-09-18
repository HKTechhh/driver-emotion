"""Small shared helpers used across scripts: reproducibility, folder setup, model stats."""
import random

import numpy as np
import tensorflow as tf

from config import ALL_DIRS


def set_seed(seed: int) -> None:
    """Seed python's random, numpy and tensorflow so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs() -> None:
    """Create every directory listed in config.ALL_DIRS if it doesn't already exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def count_params(model: tf.keras.Model) -> int:
    """Return the total number of parameters (trainable + non-trainable) in a Keras model."""
    return int(sum(np.prod(w.shape) for w in model.weights))
