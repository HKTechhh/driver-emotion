"""Checks for src/train_kmu_fed_finetune.py's model surgery and data-prep helpers.

Uses the project's own trained cnn_v1.keras/vgg16_v1.keras (already present locally) to
validate the head-replacement and freezing logic against the real architectures; skips if
they aren't there (e.g. a stripped-down handoff copy) rather than failing.
"""
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import config
from src.data import _safe_class_weights
from src.evaluate import KMU_CODE_TO_FER_NAME, SIX_CLASS_NAMES
from src.kmu_fed_data import PUBLISHED_SUBJECT_CLASS_COUNTS
from src.train_kmu_fed_finetune import (
    UNFREEZE_FROM,
    _append_runs_csv,
    _build_finetune_augmentation,
    _oversample_to_balance,
    _raw_arrays,
    build_finetune_model,
)


@pytest.mark.parametrize("model_name,base_path", [
    ("custom_cnn", config.MODELS_DIR / "cnn_v1.keras"),
    ("vgg16", config.MODELS_DIR / "vgg16_v1.keras"),
])
def test_build_finetune_model_has_a_six_unit_head(model_name, base_path) -> None:
    if not base_path.exists():
        pytest.skip(f"{base_path} not present locally.")
    base = tf.keras.models.load_model(base_path)
    ft = build_finetune_model(base, model_name)
    assert ft.output_shape == (None, len(SIX_CLASS_NAMES))

    dummy = np.zeros((1,) + ft.input_shape[1:], dtype="float32")
    out = ft(dummy, training=False).numpy()
    assert out.shape == (1, len(SIX_CLASS_NAMES))
    assert np.isclose(out.sum(), 1.0)  # still a valid softmax


@pytest.mark.parametrize("model_name,base_path", [
    ("custom_cnn", config.MODELS_DIR / "cnn_v1.keras"),
    ("vgg16", config.MODELS_DIR / "vgg16_v1.keras"),
])
def test_build_finetune_model_freezes_everything_before_the_last_block(model_name, base_path) -> None:
    if not base_path.exists():
        pytest.skip(f"{base_path} not present locally.")
    base = tf.keras.models.load_model(base_path)
    ft = build_finetune_model(base, model_name)

    from_layer = UNFREEZE_FROM[model_name]
    seen_from_layer = False
    for layer in ft.layers:
        if layer.name == from_layer:
            seen_from_layer = True
        if not seen_from_layer:
            assert not layer.trainable, f"{layer.name} should be frozen (before {from_layer})"
    assert seen_from_layer, f"{from_layer!r} was never found in the model's layers"
    # The new head is always trainable (it has fresh, untrained weights).
    assert ft.layers[-1].trainable


def test_build_finetune_model_rejects_a_model_without_a_predictions_layer() -> None:
    inputs = tf.keras.Input((4,))
    outputs = tf.keras.layers.Dense(3, name="not_predictions")(inputs)
    model = tf.keras.Model(inputs, outputs)
    with pytest.raises(ValueError):
        build_finetune_model(model, "custom_cnn")


def test_raw_arrays_shapes_and_label_encoding() -> None:
    crops = [np.zeros((80, 80, 3), dtype="uint8"), np.full((60, 100, 3), 200, dtype="uint8")]
    labels = ["happy", "sad"]
    x, y = _raw_arrays(crops, labels, img_size=48)
    assert x.shape == (2, 48, 48, 1)
    assert x.dtype == np.float32
    assert list(y) == [SIX_CLASS_NAMES.index("happy"), SIX_CLASS_NAMES.index("sad")]


def test_build_finetune_augmentation_preserves_shape() -> None:
    aug = _build_finetune_augmentation()
    batch = np.random.randint(0, 255, size=(4, 48, 48, 1)).astype("float32")
    out = aug(batch, training=True).numpy()
    assert out.shape == batch.shape


def test_append_runs_csv_writes_header_once(tmp_path) -> None:
    runs_path = tmp_path / "runs.csv"
    row = {
        "run_name": "test_run", "model": "custom_cnn", "date": "2026-01-01", "epochs_run": 5,
        "best_val_acc": 0.5, "val_macro_f1": 0.4, "params": 100, "train_minutes": 1.0,
        "img_size": 48, "batch_size": 16, "lr": 1e-5,
    }
    _append_runs_csv(row, runs_path=runs_path)
    _append_runs_csv({**row, "run_name": "test_run_2"}, runs_path=runs_path)

    lines = runs_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert lines[0].startswith("run_name,model,date")


def test_class_weights_upweight_disgust_on_the_real_train_split() -> None:
    """The 8 train subjects give disgust only 60 of 700 images (vs 120-160 for every other
    class) - the first fine-tuning run (no class weights) completely failed to learn it.
    Reuses src.data's own capped square-root weighting (not a re-implementation)."""
    y = []
    for sid in config.KMU_FED["train_subjects"]:
        for code, n in PUBLISHED_SUBJECT_CLASS_COUNTS[sid].items():
            y.extend([SIX_CLASS_NAMES.index(KMU_CODE_TO_FER_NAME[code])] * n)
    y = np.array(y)
    assert len(y) == 700

    weights = _safe_class_weights(y, num_classes=len(SIX_CLASS_NAMES))
    disgust_idx = SIX_CLASS_NAMES.index("disgust")
    happy_idx = SIX_CLASS_NAMES.index("happy")  # the most common class in this split
    assert weights[disgust_idx] > weights[happy_idx]
    assert weights[disgust_idx] < config.CLASS_WEIGHT_MAX  # capped, not the raw ~2x imbalance ratio


def test_oversample_to_balance_evens_out_the_real_train_split() -> None:
    """Class weighting (above) upweighted disgust's loss but didn't fix it learning nothing
    (0/0/0 precision/recall/f1) - this duplicates disgust's images directly so every class
    appears equally often per epoch, attacking the "too few examples" problem more directly."""
    x, y = [], []
    for sid in config.KMU_FED["train_subjects"]:
        for code, n in PUBLISHED_SUBJECT_CLASS_COUNTS[sid].items():
            label = SIX_CLASS_NAMES.index(KMU_CODE_TO_FER_NAME[code])
            x.extend([label] * n)  # stand-in "image" = its own label, so we can check duplicates
            y.extend([label] * n)
    x = np.array(x)
    y = np.array(y)
    assert len(y) == 700

    x_bal, y_bal = _oversample_to_balance(x, y)

    counts = np.bincount(y_bal)
    assert len(counts) == len(SIX_CLASS_NAMES)
    assert len(set(counts)) == 1  # every class now appears exactly as often as the majority class
    assert counts[0] == np.bincount(y).max()
    # the stand-in "image" values still match their own labels after shuffling/duplicating
    assert np.array_equal(x_bal, y_bal)

    # reproducible with the module's fixed SEED (same call, no seed argument, twice)
    x_bal_2, y_bal_2 = _oversample_to_balance(x, y)
    assert np.array_equal(y_bal, y_bal_2)
