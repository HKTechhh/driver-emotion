"""Step 2 of the KMU-FED integration: fine-tune cnn_v1/vgg16_v1 on KMU-FED.

See src/kmu_fed_data.py (Step 1's subject-level split) and src/evaluate.py's --dataset
kmu_fed (the pre-fine-tune domain-gap baseline, run on the exact same held-out test
subjects this script evaluates on at the end, for a fair before/after comparison).

For the chosen base model, replaces its final "predictions" Dense layer with a fresh
6-unit head (KMU-FED's classes - no "neutral"), freezes everything except that head and
the last conv block (reusing src.models.vgg16_tl.unfreeze_top - the exact function
vgg16_v1's own stage-2 training used, not a re-implementation of it), and fine-tunes on
KMU-FED's 8 train subjects with heavier augmentation than FER2013's own (src/data.py),
since well under 900 images makes overfitting the default outcome, not an edge case, and
with class weights (reusing src.data's own capped square-root weighting, not a
re-implementation) since disgust gets only 60 of 700 training images - the first
fine-tuning run (no class weights) completely failed to learn disgust at all. Validates on
the 2 val subjects, early-stops on val_accuracy, and evaluates only on the 2 held-out test
subjects (never seen in training or fine-tuning).

KMU-FED is tiny, so every subject's face crop is detected once and held in memory rather
than built into a lazy tf.data pipeline from disk.

Saves as a NEW file (models/{model}_kmufed_ft_v1.keras) and a NEW results/runs.csv row -
never overwrites cnn_v1.keras/vgg16_v1.keras or their existing rows.

Run as a module from the repo root, e.g.:
    python -m src.train_kmu_fed_finetune --model custom_cnn --base-model-path models/cnn_v1.keras
    python -m src.train_kmu_fed_finetune --model vgg16 --base-model-path models/vgg16_v1.keras
"""
import argparse
import csv
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import EARLY_STOP_PATIENCE, KMU_FED, LR_FACTOR, LR_PATIENCE, MODELS_DIR, REALTIME, RESULTS_DIR, SEED
from src.data import _safe_class_weights
from src.evaluate import KMU_CODE_TO_FER_NAME, SIX_CLASS_NAMES, _save_confusion_matrices
from src.kmu_fed_data import parse_filename
from src.models.vgg16_tl import unfreeze_top
from src.preprocess import preprocess
from src.realtime import _crop_with_padding, build_face_detector
from src.utils import count_params, ensure_dirs, set_seed

UNFREEZE_FROM = {"custom_cnn": "block4_conv1", "vgg16": "block5_conv1"}
RUNS_CSV_COLUMNS = [
    "run_name", "model", "date", "epochs_run", "best_val_acc", "val_macro_f1",
    "params", "train_minutes", "img_size", "batch_size", "lr",
]


def _load_kmu_fed_subject_crops(subject_ids: List[int], detect_fn) -> Tuple[List[np.ndarray], List[str], int]:
    """Detect and crop the face in every KMU-FED image belonging to `subject_ids`.

    Returns (BGR face crops, matching FER2013-style class names, count of images where no
    face was found). Shared across both models - the crop itself doesn't depend on which
    model will later preprocess it, so this is only ever done once per subject group.
    """
    kmu_dir = Path(KMU_FED["dir"])
    crops, labels = [], []
    n_no_face = 0
    for f in sorted(kmu_dir.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        subject_id, cls_code, _initials, _index = parse_filename(f.name)
        if subject_id not in subject_ids:
            continue
        image_bgr = cv2.imread(str(f))
        if image_bgr is None:
            n_no_face += 1
            continue
        boxes = detect_fn(image_bgr)
        if not boxes:
            n_no_face += 1
            continue
        x, y, w, h, _conf = max(boxes, key=lambda b: b[2] * b[3])
        crop = _crop_with_padding(image_bgr, (x, y, w, h), REALTIME["face_padding"])
        if crop.size == 0:
            n_no_face += 1
            continue
        crops.append(crop)
        labels.append(KMU_CODE_TO_FER_NAME[cls_code])
    return crops, labels, n_no_face


def _raw_arrays(crops: List[np.ndarray], labels: List[str], img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Grayscale + resize every crop to `img_size`; RAW [0, 255] pixels, not yet preprocessed
    for a specific model (augmentation must run on this raw range, not after preprocessing -
    this project's own Bug 1, docs/experiment_log.md: RandomBrightness/RandomContrast assume
    [0, 255] by default and silently wreck an already-rescaled image).
    """
    resized = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (img_size, img_size), interpolation=cv2.INTER_AREA)
               for c in crops]
    x = np.stack(resized).astype("float32")[..., np.newaxis]
    y = np.array([SIX_CLASS_NAMES.index(lbl) for lbl in labels], dtype=np.int64)
    return x, y


def _build_finetune_augmentation() -> tf.keras.Sequential:
    """Heavier than FER2013's own augmentation (src/data.py's _build_augmentation): KMU-FED's
    fine-tuning set is well under 900 images, small enough that overfitting is the default
    outcome, not an edge case, per this step's own brief.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),                          # ~36 degrees (vs FER2013's ~18)
        tf.keras.layers.RandomZoom(0.2),                              # vs FER2013's 0.1
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.3, value_range=(0, 255)),  # vs FER2013's 0.2
        tf.keras.layers.RandomContrast(0.3),
    ], name="kmu_fed_finetune_augmentation")


def _make_dataset(x_raw: np.ndarray, y: np.ndarray, model_name: str, batch_size: int, augment: bool) -> tf.data.Dataset:
    """Build a batched, per-model-preprocessed tf.data.Dataset from raw [0, 255] arrays."""
    ds = tf.data.Dataset.from_tensor_slices((x_raw, y))
    if augment:
        ds = ds.shuffle(len(x_raw), seed=SEED)
        aug = _build_finetune_augmentation()
        ds = ds.map(lambda x, lbl: (aug(x, training=True), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, lbl: (preprocess(x, model_name), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_finetune_model(base_model: tf.keras.Model, model_name: str, num_classes: int = len(SIX_CLASS_NAMES)) -> tf.keras.Model:
    """Replace `base_model`'s final "predictions" layer with a fresh `num_classes`-unit head,
    then freeze everything except that head and the last conv block (reuses
    src.models.vgg16_tl.unfreeze_top - the exact mechanism vgg16_v1's own stage-2 training
    used, including its "keep BatchNorm frozen" rule, which matters here too: KMU-FED's
    fine-tuning batches are tiny, and unfrozen BatchNorm stats would be unstable on them).
    """
    if base_model.layers[-1].name != "predictions":
        raise ValueError(f"Expected the final layer to be named 'predictions', got {base_model.layers[-1].name!r}")
    penultimate_output = base_model.layers[-2].output
    new_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(penultimate_output)
    model = tf.keras.Model(base_model.input, new_output, name=f"{base_model.name}_kmufed_ft")

    for layer in model.layers:
        layer.trainable = False
    unfreeze_top(model, from_layer=UNFREEZE_FROM[model_name])
    return model


def _append_runs_csv(row: Dict, runs_path: Path = RESULTS_DIR / "runs.csv") -> None:
    """Append one row to results/runs.csv, matching its existing column order exactly."""
    is_new = not runs_path.exists()
    with open(runs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for fine-tuning one base model on KMU-FED."""
    parser = argparse.ArgumentParser(description="Fine-tune cnn_v1 or vgg16_v1 on KMU-FED (Step 2).")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16"], required=True)
    parser.add_argument("--base-model-path", dest="base_model_path", type=str, required=True,
                         help="The frozen, unchanged model to fine-tune from (e.g. models/cnn_v1.keras).")
    parser.add_argument("--epochs", type=int, default=15, help="Max epochs (early stopping usually ends it sooner).")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--out-name", dest="out_name", type=str, default=None,
        help="Defaults to '{model}_kmufed_ft_v1' -> models/{out_name}.keras.",
    )
    return parser


def main() -> None:
    """Fine-tune one base model on KMU-FED and evaluate it on the held-out test subjects."""
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    out_name = args.out_name or f"{'cnn' if args.model == 'custom_cnn' else 'vgg16'}_kmufed_ft_v1"
    model_path = MODELS_DIR / f"{out_name}.keras"

    base_model = tf.keras.models.load_model(args.base_model_path)
    img_size = base_model.input_shape[1]
    detect_fn = build_face_detector(REALTIME["min_face_confidence"])

    print("Loading and cropping KMU-FED faces (shared across models)...")
    train_crops, train_labels, train_no_face = _load_kmu_fed_subject_crops(KMU_FED["train_subjects"], detect_fn)
    val_crops, val_labels, val_no_face = _load_kmu_fed_subject_crops(KMU_FED["val_subjects"], detect_fn)
    test_crops, test_labels, test_no_face = _load_kmu_fed_subject_crops(KMU_FED["test_subjects"], detect_fn)
    print(f"  train: {len(train_crops)} faces found, {train_no_face} images with no face detected")
    print(f"  val:   {len(val_crops)} faces found, {val_no_face} images with no face detected")
    print(f"  test:  {len(test_crops)} faces found, {test_no_face} images with no face detected")

    x_train, y_train = _raw_arrays(train_crops, train_labels, img_size)
    x_val, y_val = _raw_arrays(val_crops, val_labels, img_size)
    train_ds = _make_dataset(x_train, y_train, args.model, args.batch_size, augment=True)
    val_ds = _make_dataset(x_val, y_val, args.model, args.batch_size, augment=False)

    # Reuses src.data's own capped square-root class weighting (config.CLASS_WEIGHT_MODE/
    # CLASS_WEIGHT_MAX) rather than duplicating it - the same fix this project already made
    # for FER2013's imbalance (disgust ~9.5x with plain "balanced" weights destabilised
    # training there too). KMU-FED's 8 train subjects give disgust only 60 of 700 images
    # (vs 120-160 for every other class), and the first fine-tuning run completely failed to
    # learn disgust (and, for the CNN, angry too) - a classic imbalance symptom.
    class_weight = _safe_class_weights(y_train, num_classes=len(SIX_CLASS_NAMES))
    print("Class weights:", {SIX_CLASS_NAMES[c]: round(w, 2) for c, w in class_weight.items()})

    model = build_finetune_model(base_model, args.model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=LR_FACTOR, patience=LR_PATIENCE),
        tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / f"{out_name}_history.csv")),
    ]

    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks,
        class_weight=class_weight, verbose=2,
    )
    train_minutes = (time.time() - start) / 60
    epochs_run = len(history.history["loss"])
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nFine-tuning done: {epochs_run} epoch(s), {train_minutes:.1f} min, best val_accuracy {best_val_acc:.4f}")

    # ModelCheckpoint already saved the best-val_accuracy weights to model_path; reload it so
    # the held-out test evaluation below uses exactly the saved file, not the in-memory model
    # (EarlyStopping's restore_best_weights should make these identical, but this removes any
    # doubt and matches how every other evaluation script in this project loads from disk).
    model = tf.keras.models.load_model(model_path)

    x_test, y_test = _raw_arrays(test_crops, test_labels, img_size)
    x_test_processed = preprocess(tf.convert_to_tensor(x_test), args.model).numpy()
    y_pred = np.argmax(model.predict(x_test_processed, verbose=0), axis=1)
    report = classification_report(
        y_test, y_pred, labels=range(len(SIX_CLASS_NAMES)), target_names=SIX_CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    print(f"Held-out test subjects {KMU_FED['test_subjects']}: "
          f"accuracy={report['accuracy']:.4f} macro_f1={report['macro avg']['f1-score']:.4f} "
          f"(n={len(y_test)}, {test_no_face} images had no face detected)")
    _save_confusion_matrices(y_test, y_pred, f"{out_name}_test", class_names=SIX_CLASS_NAMES, num_classes=len(SIX_CLASS_NAMES))

    val_macro_f1 = None
    if epochs_run:
        y_val_pred = np.argmax(model.predict(preprocess(tf.convert_to_tensor(x_val), args.model).numpy(), verbose=0), axis=1)
        val_macro_f1 = classification_report(
            y_val, y_val_pred, labels=range(len(SIX_CLASS_NAMES)), output_dict=True, zero_division=0,
        )["macro avg"]["f1-score"]

    _append_runs_csv({
        "run_name": out_name,
        "model": args.model,
        "date": date.today().isoformat(),
        "epochs_run": epochs_run,
        "best_val_acc": best_val_acc,
        "val_macro_f1": val_macro_f1,
        "params": count_params(model),
        "train_minutes": round(train_minutes, 2),
        "img_size": img_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
    })

    metrics_path = RESULTS_DIR / f"{out_name}_kmu_fed_test_metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump({
            "run_name": out_name,
            "model": args.model,
            "base_model_path": args.base_model_path,
            "test_subjects": KMU_FED["test_subjects"],
            "n_train": len(train_crops), "n_val": len(val_crops), "n_test_evaluated": len(y_test),
            "n_test_no_face": test_no_face,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "classification_report": report,
        }, f, indent=2)
    print(f"Wrote {model_path}, {metrics_path}, and appended to results/runs.csv")


if __name__ == "__main__":
    main()
