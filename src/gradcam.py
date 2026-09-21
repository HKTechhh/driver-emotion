"""Grad-CAM explainability grids for both trained models.

Run as a module from the repo root, e.g.:
    python -m src.gradcam --cnn-model-path models/cnn_v1.keras --vgg16-model-path models/vgg16_v1.keras
"""
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, FIGURES_DIR, GRADCAM_LAYER, NUM_CLASSES, SEED
from src.robustness import _load_raw_test_images, _preprocess_for_model  # shared with Phase 5, not duplicated
from src.utils import ensure_dirs, set_seed

DISPLAY_SIZE = 150  # upscaled resolution the original/overlay images are shown at


def _predict_all(
    model: tf.keras.Model, model_name: str, images: np.ndarray, img_size: int, chunk_size: int = 256
) -> np.ndarray:
    """Predicted class index for every raw image, using `model`'s own input size.

    Works `chunk_size` images at a time: preprocessing the whole test set for VGG16 at once
    is a 4.3 GB float32 array plus copies, which gets the process OOM-killed (same bug that
    hit src/robustness.py).
    """
    chunks = []
    for start in range(0, len(images), chunk_size):
        x = _preprocess_for_model(images[start : start + chunk_size], model_name, img_size)
        chunks.append(np.argmax(model.predict(x, batch_size=64, verbose=0), axis=1))
    return np.concatenate(chunks)


def _build_grad_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    """A model that also exposes `layer_name`'s activations, for Grad-CAM."""
    return tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])


def _gradcam_heatmap(grad_model: tf.keras.Model, preprocessed_image: np.ndarray, class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap (HxW, normalised to [0, 1]) explaining `class_idx` for one preprocessed image."""
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(preprocessed_image[np.newaxis, ...])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.squeeze(conv_output @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto a raw uint8 grayscale image. Returns an RGB uint8 array."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    overlay = heatmap_color.astype(np.float32) * alpha + base_rgb * (1 - alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _find_correct_examples(labels: np.ndarray, cnn_preds: np.ndarray, vgg_preds: np.ndarray) -> List[Optional[int]]:
    """One test index per class, preferring an image both models classify correctly."""
    indices: List[Optional[int]] = []
    for class_idx in range(NUM_CLASSES):
        both = np.flatnonzero((labels == class_idx) & (cnn_preds == class_idx) & (vgg_preds == class_idx))
        either = np.flatnonzero((labels == class_idx) & ((cnn_preds == class_idx) | (vgg_preds == class_idx)))
        any_of_class = np.flatnonzero(labels == class_idx)
        for candidates in (both, either, any_of_class):
            if len(candidates):
                indices.append(int(candidates[0]))
                break
        else:
            indices.append(None)
    return indices


def _save_gradcam_grid(
    images: np.ndarray,
    example_indices: List[Optional[int]],
    cnn_grad_model: tf.keras.Model,
    vgg_grad_model: tf.keras.Model,
    cnn_preds: np.ndarray,
    vgg_preds: np.ndarray,
    cnn_size: int,
    vgg_size: int,
    out_path: Path,
) -> None:
    """rows = 7 emotions, columns = original | custom_cnn Grad-CAM | vgg16 Grad-CAM."""
    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(9, 3 * NUM_CLASSES))
    col_titles = ["Original", "Custom CNN Grad-CAM", "VGG16 Grad-CAM"]

    for row, idx in enumerate(example_indices):
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(CLASS_NAMES[row], fontsize=11)
        if idx is None:
            for ax in axes[row]:
                ax.axis("off")
            continue

        raw = images[idx]
        display_img = cv2.resize(raw, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

        cnn_input = _preprocess_for_model(raw[np.newaxis], "custom_cnn", cnn_size)[0]
        vgg_input = _preprocess_for_model(raw[np.newaxis], "vgg16", vgg_size)[0]
        cnn_heatmap = _gradcam_heatmap(cnn_grad_model, cnn_input, int(cnn_preds[idx]))
        vgg_heatmap = _gradcam_heatmap(vgg_grad_model, vgg_input, int(vgg_preds[idx]))

        axes[row, 0].imshow(display_img, cmap="gray")
        axes[row, 1].imshow(_overlay_heatmap(display_img, cnn_heatmap))
        axes[row, 2].imshow(_overlay_heatmap(display_img, vgg_heatmap))

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)

    fig.suptitle("Grad-CAM: one correctly classified example per emotion", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pick_diverse_errors(labels: np.ndarray, preds: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """Up to `n` misclassified indices, at most one per true class, most-error classes first.

    Test images are stored class by class, so just taking the first `n` errors returns
    only the first class (every example was "true: angry" in the first version). This
    picks one error per class instead, seeded so the figure is reproducible.
    """
    rng = np.random.default_rng(seed)
    wrong = np.flatnonzero(preds != labels)
    by_class = {c: wrong[labels[wrong] == c] for c in range(NUM_CLASSES)}
    ranked = sorted((c for c in by_class if len(by_class[c])), key=lambda c: -len(by_class[c]))
    return np.array([int(rng.choice(by_class[c])) for c in ranked[:n]], dtype=int)


def _save_misclassified_grid(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    grad_model: tf.keras.Model,
    model_name: str,
    img_size: int,
    out_path: Path,
    n: int = 6,
) -> None:
    """A grid of up to `n` misclassified examples (spread across true classes) with Grad-CAM overlays."""
    wrong_idx = _pick_diverse_errors(labels, preds, n)
    if len(wrong_idx) == 0:
        print(f"No misclassified examples found for {model_name}; skipping {out_path.name}.")
        return

    cols = 3
    rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    for i, idx in enumerate(wrong_idx):
        raw = images[idx]
        display_img = cv2.resize(raw, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        preprocessed = _preprocess_for_model(raw[np.newaxis], model_name, img_size)[0]
        heatmap = _gradcam_heatmap(grad_model, preprocessed, int(preds[idx]))

        ax = axes.flat[i]
        ax.imshow(_overlay_heatmap(display_img, heatmap))
        ax.set_title(f"true: {CLASS_NAMES[labels[idx]]}\npred: {CLASS_NAMES[preds[idx]]}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{model_name} — misclassified examples (Grad-CAM)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for generating the Grad-CAM figures."""
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explainability grids for both models.")
    parser.add_argument("--cnn-model-path", dest="cnn_model_path", type=str, required=True)
    parser.add_argument("--vgg16-model-path", dest="vgg16_model_path", type=str, required=True)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of each class to use.")
    return parser


def main() -> None:
    """Build the 7-emotion Grad-CAM grid and each model's misclassified-examples grid."""
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    ensure_dirs()

    images, labels = _load_raw_test_images(subset=args.subset)
    cnn_model = tf.keras.models.load_model(args.cnn_model_path)
    vgg_model = tf.keras.models.load_model(args.vgg16_model_path)
    cnn_size, vgg_size = cnn_model.input_shape[1], vgg_model.input_shape[1]

    cnn_preds = _predict_all(cnn_model, "custom_cnn", images, cnn_size)
    vgg_preds = _predict_all(vgg_model, "vgg16", images, vgg_size)

    cnn_grad_model = _build_grad_model(cnn_model, GRADCAM_LAYER["custom_cnn"])
    vgg_grad_model = _build_grad_model(vgg_model, GRADCAM_LAYER["vgg16"])

    example_indices = _find_correct_examples(labels, cnn_preds, vgg_preds)
    _save_gradcam_grid(
        images, example_indices, cnn_grad_model, vgg_grad_model, cnn_preds, vgg_preds,
        cnn_size, vgg_size, FIGURES_DIR / "gradcam_grid.png",
    )

    _save_misclassified_grid(
        images, labels, cnn_preds, cnn_grad_model, "custom_cnn", cnn_size,
        FIGURES_DIR / "gradcam_misclassified_custom_cnn.png",
    )
    _save_misclassified_grid(
        images, labels, vgg_preds, vgg_grad_model, "vgg16", vgg_size,
        FIGURES_DIR / "gradcam_misclassified_vgg16.png",
    )

    print(f"Wrote {FIGURES_DIR / 'gradcam_grid.png'} and misclassified grids.")


if __name__ == "__main__":
    main()
