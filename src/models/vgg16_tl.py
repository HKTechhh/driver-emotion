"""Model B: VGG16 pretrained on ImageNet, adapted via two-stage transfer learning."""
from typing import Optional

import tensorflow as tf
from tensorflow.keras import applications, layers

from config import NUM_CLASSES
from config import VGG16 as VGG16_CFG


def build_vgg16(
    cfg: dict = VGG16_CFG, num_classes: int = NUM_CLASSES, img_size: Optional[int] = None
) -> tf.keras.Model:
    """Build VGG16 with a frozen ImageNet base and a new classification head.

    Loads `keras.applications.VGG16(include_top=False, weights="imagenet")`, freezes
    it, and adds GlobalAveragePooling2D -> Dense(512, relu) -> Dropout(cfg["dropout"])
    -> Dense(num_classes, softmax). The base's own layers (`block1_conv1`, ...,
    `block5_conv3`) stay directly accessible on the returned model (e.g. for Grad-CAM).
    `img_size` overrides `cfg["img_size"]` for the input shape.
    """
    size = img_size or cfg["img_size"]

    base = applications.VGG16(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(cfg["dropout"], name="fc1_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(base.input, outputs, name="vgg16_tl")


def unfreeze_top(model: tf.keras.Model, from_layer: str = VGG16_CFG["unfreeze_from"]) -> None:
    """Make layers from `from_layer` onward trainable, in place. BatchNorm layers stay frozen."""
    unfreezing = False
    for layer in model.layers:
        if layer.name == from_layer:
            unfreezing = True
        if unfreezing:
            layer.trainable = not isinstance(layer, layers.BatchNormalization)


if __name__ == "__main__":
    build_vgg16().summary()
