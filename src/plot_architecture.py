"""Layered architecture diagrams for both models, for the paper (Phase 8).

Run as a module from the repo root:
    python -m src.plot_architecture
"""
import argparse

import tensorflow as tf
import visualkeras

from config import PAPER_FIG_DIR
from src.models.custom_cnn import build_custom_cnn
from src.models.vgg16_tl import build_vgg16
from src.utils import ensure_dirs


def _patch_visualkeras_keras3_compat() -> None:
    """visualkeras 0.2.0 (latest on PyPI) still reads `layer.output_shape`, which Keras 3
    removed from every layer type in favour of `layer.output.shape`. Add it back as a
    read-only property so visualkeras keeps working without forking the package.
    """
    if hasattr(tf.keras.layers.Layer, "output_shape"):
        return

    def _output_shape(self):
        try:
            return tuple(self.output.shape)
        except AttributeError:
            return None

    tf.keras.layers.Layer.output_shape = property(_output_shape)


# Hidden from the diagram: these don't change spatial structure and just add visual
# clutter/width for a deep network. Conv/Pool/Dense stay, which is what a reader needs
# to follow the architecture.
_IGNORED_LAYER_TYPES = [
    tf.keras.layers.BatchNormalization,
    tf.keras.layers.ReLU,
    tf.keras.layers.Dropout,
]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for generating the architecture diagrams."""
    parser = argparse.ArgumentParser(description="Save layered architecture diagrams for both models.")
    parser.add_argument(
        "--vgg16-img-size",
        dest="vgg16_img_size",
        type=int,
        default=None,
        help="Override VGG16's img_size (defaults to config.VGG16['img_size']).",
    )
    return parser


def main() -> None:
    """Build both models and save their layered architecture diagrams to paper/figures/."""
    args = build_arg_parser().parse_args()
    ensure_dirs()
    _patch_visualkeras_keras3_compat()

    custom_cnn = build_custom_cnn()
    custom_cnn_path = PAPER_FIG_DIR / "custom_cnn_architecture.png"
    visualkeras.layered_view(
        custom_cnn, to_file=str(custom_cnn_path), legend=True, draw_volume=True,
        type_ignore=_IGNORED_LAYER_TYPES,
    )
    print(f"Wrote {custom_cnn_path}")

    vgg16 = build_vgg16(img_size=args.vgg16_img_size)
    vgg16_path = PAPER_FIG_DIR / "vgg16_architecture.png"
    visualkeras.layered_view(
        vgg16, to_file=str(vgg16_path), legend=True, draw_volume=True,
        type_ignore=_IGNORED_LAYER_TYPES,
    )
    print(f"Wrote {vgg16_path}")


if __name__ == "__main__":
    main()
