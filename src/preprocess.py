"""Per-model pixel preprocessing shared by training (src/data.py) and the real-time
app (src/realtime.py), so a live camera frame is transformed exactly like a training image.

Callers are responsible for getting the image to grayscale and to the model's `img_size`
first (src/data.py's loader resizes at load time; src/realtime.py resizes the face crop
with cv2 before calling in here) -- these functions only do the model-specific channel
and value-scale conversion, which is the part most likely to silently drift out of sync
between training and live inference.
"""
import tensorflow as tf


def preprocess_custom_cnn(image: tf.Tensor) -> tf.Tensor:
    """Scale grayscale pixels from [0, 255] to [0, 1]. Works on a single image or a batch."""
    return tf.cast(image, tf.float32) / 255.0


def preprocess_vgg16(image: tf.Tensor) -> tf.Tensor:
    """Convert grayscale to 3-channel and apply Keras' VGG16 preprocessing. Single image or batch."""
    image = tf.image.grayscale_to_rgb(tf.cast(image, tf.float32))
    return tf.keras.applications.vgg16.preprocess_input(image)


def preprocess(image: tf.Tensor, model_name: str) -> tf.Tensor:
    """Dispatch to the right per-model preprocessing for an already-resized grayscale image."""
    if model_name == "custom_cnn":
        return preprocess_custom_cnn(image)
    if model_name == "vgg16":
        return preprocess_vgg16(image)
    raise ValueError(f"Unknown model_name: {model_name!r}")
