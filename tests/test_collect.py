"""Smoke tests for app_collect.py's non-Streamlit helper functions.

Streamlit apps can't be pytest'd directly: `main()` (and the tab-render functions) call
`st.*` UI functions that need a running Streamlit session. Importing the module itself is
safe though - none of the module-level code calls Streamlit, only `main()` does, and that
only runs under `if __name__ == "__main__"` - so this tests the plain-Python helpers the UI
is built on directly.
"""
import cv2
import numpy as np
import pytest

import config
from app_collect import (
    append_log_row,
    fer2013_train_class_counts,
    infer_model_family,
    list_available_models,
    sample_video_frames,
    save_crop,
)


class _FakeModel:
    """Stands in for a tf.keras.Model just enough for infer_model_family (reads input_shape)."""

    def __init__(self, channels: int):
        self.input_shape = (None, 48, 48, channels)


def test_infer_model_family_grayscale_and_rgb() -> None:
    assert infer_model_family(_FakeModel(1)) == "custom_cnn"
    assert infer_model_family(_FakeModel(3)) == "vgg16"


def test_infer_model_family_unknown_channels_raises() -> None:
    with pytest.raises(ValueError):
        infer_model_family(_FakeModel(4))


def test_list_available_models_finds_keras_files(tmp_path) -> None:
    (tmp_path / "cnn_v1.keras").touch()
    (tmp_path / "vgg16_v1.keras").touch()
    (tmp_path / "notes.txt").touch()
    assert list_available_models(tmp_path) == ["cnn_v1", "vgg16_v1"]


def test_save_crop_and_append_log_row(tmp_path) -> None:
    crop = np.zeros((48, 48, 3), dtype="uint8")
    saved_path = save_crop(crop, "happy", "example.jpg", dest_root=tmp_path / "collected")
    assert saved_path.exists()
    assert saved_path.parent.name == "happy"

    log_path = tmp_path / "collected_log.csv"
    append_log_row(
        {
            "timestamp": "2026-01-01T00:00:00", "source": "image_upload", "original_filename": "example.jpg",
            "model_used": "cnn_v1", "predicted_emotion": "happy", "confidence": 0.9,
            "corrected_label": "", "saved_path": str(saved_path),
        },
        log_path=log_path,
    )
    assert log_path.exists()
    text = log_path.read_text()
    assert "example.jpg" in text and "happy" in text


def test_sample_video_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype="uint8"))
    writer.release()

    frames = list(sample_video_frames(video_path, every_n=3))
    assert [i for i, _ in frames] == [0, 3, 6, 9]
    assert all(frame.shape == (32, 32, 3) for _, frame in frames)


def test_fer2013_train_class_counts_missing_dataset_returns_none() -> None:
    assert fer2013_train_class_counts(fer_dir="does/not/exist") is None


def test_predict_and_save_pipeline_on_a_real_sample_image(tmp_path) -> None:
    """End-to-end smoke test: a real model, a real FER2013 image, through predict_face and save_crop."""
    from src.realtime import build_face_detector, load_model_for_inference, predict_face

    sample_dir = config.FER_DIR / "train" / "happy"
    if not sample_dir.exists():
        pytest.skip("FER2013 not present locally; skipping the real-image smoke test.")
    sample_path = next(sample_dir.glob("*"))
    image = cv2.imread(str(sample_path))

    model_path = config.MODELS_DIR / "cnn_v1.keras"
    if not model_path.exists():
        pytest.skip("cnn_v1.keras not present locally; skipping the real-model smoke test.")
    model, img_size = load_model_for_inference(str(model_path))
    detect_fn = build_face_detector(config.REALTIME["min_face_confidence"])

    result = predict_face(image, detect_fn, model, "custom_cnn", img_size, config.REALTIME["face_padding"])
    assert result is not None
    assert result["probs"].shape == (config.NUM_CLASSES,)

    saved = save_crop(result["crop"], "happy", sample_path.name, dest_root=tmp_path)
    assert saved.exists()
