"""Model A: a from-scratch CNN built from VGG-style blocks (stacked 3x3 convs -> maxpool)."""
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from config import CNN, NUM_CLASSES


def build_custom_cnn(
    cfg: dict = CNN, num_classes: int = NUM_CLASSES, img_size: Optional[int] = None
) -> tf.keras.Model:
    """Build the custom CNN.

    One VGG-style block per value in `cfg["filters"]`: two Conv2D(3x3, padding="same")
    -> BatchNorm -> ReLU, then MaxPool(2x2) and Dropout(0.25). The head is
    GlobalAveragePooling2D -> Dense(256) -> BatchNorm -> ReLU -> Dropout(cfg["dropout"])
    -> Dense(num_classes, softmax). Conv layers use L2(cfg["l2"]) weight regularisation.
    `img_size` overrides `cfg["img_size"]` for the input shape (e.g. for quick CPU tests).
    """
    size = img_size or cfg["img_size"]
    reg = regularizers.l2(cfg["l2"])

    inputs = tf.keras.Input(shape=(size, size, cfg["channels"]), name="input")
    x = inputs
    for block_idx, num_filters in enumerate(cfg["filters"], start=1):
        for conv_idx in (1, 2):
            x = layers.Conv2D(
                num_filters,
                cfg["kernel_size"],
                padding="same",
                kernel_regularizer=reg,
                name=f"block{block_idx}_conv{conv_idx}",
            )(x)
            x = layers.BatchNormalization(name=f"block{block_idx}_bn{conv_idx}")(x)
            x = layers.ReLU(name=f"block{block_idx}_relu{conv_idx}")(x)
        x = layers.MaxPooling2D(2, name=f"block{block_idx}_pool")(x)
        x = layers.Dropout(0.25, name=f"block{block_idx}_dropout")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(cfg["dropout"], name="fc1_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="custom_cnn")


if __name__ == "__main__":
    build_custom_cnn().summary()
