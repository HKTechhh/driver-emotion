"""Data pipeline tests for both models. Requires FER2013 at config.FER_DIR (train/ and test/)."""
import pytest
import tensorflow as tf

from config import CNN, FER_DIR, NUM_CLASSES, VGG16
from src.data import get_datasets

requires_data = pytest.mark.skipif(
    not (FER_DIR / "train").exists(),
    reason="FER2013 not downloaded to data/raw/fer2013 (see PLAN.md Phase 1)",
)


@requires_data
@pytest.mark.parametrize("model_name,cfg", [("custom_cnn", CNN), ("vgg16", VGG16)])
def test_get_datasets_shapes_and_labels(model_name: str, cfg: dict) -> None:
    train_ds, val_ds, test_ds, class_weights = get_datasets(model_name, subset=0.01)
    expected_channels = 1 if model_name == "custom_cnn" else 3

    for ds in (train_ds, val_ds, test_ds):
        images, labels = next(iter(ds))
        assert images.shape[1:] == (cfg["img_size"], cfg["img_size"], expected_channels)
        assert tf.reduce_min(labels).numpy() >= 0
        assert tf.reduce_max(labels).numpy() < NUM_CLASSES

    assert set(class_weights.keys()) == set(range(NUM_CLASSES))
    assert all(w > 0 for w in class_weights.values())
