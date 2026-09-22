"""Checks for src/realtime.py's per-frame logic, without opening a camera or loading a real model."""
import numpy as np

import config
from src.realtime import EmotionTracker, process_frame


def _fake_model(x, training=False):  # noqa: ARG001 - mirrors tf.keras.Model's call signature
    return np.full((1, config.NUM_CLASSES), 1.0 / config.NUM_CLASSES, dtype="float32")


def _tracker() -> EmotionTracker:
    return EmotionTracker(smoothing_window=1, alert_emotions=(), alert_seconds=999.0)


def test_process_frame_no_face_returns_no_log_row_or_crop() -> None:
    frame = np.zeros((100, 100, 3), dtype="uint8")
    annotated, log_row, crop = process_frame(
        frame, detect_fn=lambda f: [], model=_fake_model, model_name="custom_cnn",
        img_size=48, tracker=_tracker(), fps=10.0, face_padding=0.15,
    )
    assert annotated.shape == frame.shape
    assert log_row is None
    assert crop is None


def test_process_frame_with_face_returns_log_row_and_bgr_crop() -> None:
    frame = np.random.randint(0, 255, size=(120, 120, 3), dtype="uint8")
    detect_fn = lambda f: [(10, 10, 80, 80, 0.99)]  # noqa: E731 - one box covering most of the frame
    annotated, log_row, crop = process_frame(
        frame, detect_fn=detect_fn, model=_fake_model, model_name="custom_cnn",
        img_size=48, tracker=_tracker(), fps=10.0, face_padding=0.15,
    )
    assert annotated.shape == frame.shape
    assert log_row is not None
    assert set(log_row) == {"timestamp", "emotion", "confidence", "fps"}
    assert log_row["emotion"] in config.CLASS_NAMES
    assert crop is not None and crop.ndim == 3 and crop.shape[2] == 3  # BGR, before grayscale/resize
