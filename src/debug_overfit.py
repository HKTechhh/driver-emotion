"""Debug script (not part of the pipeline): overfit-one-fixed-batch sanity test.

If the model can't memorize a single fixed batch to near-100% training accuracy, the
bug is in the model or the labels, not in the class-weighting scheme -- run this before
touching any training code.

Run as a module from the repo root:
    python -m src.debug_overfit
"""
import tensorflow as tf

from config import CNN, NUM_CLASSES, SEED
from src.debug_data import load_raw_batch
from src.models.custom_cnn import build_custom_cnn
from src.utils import set_seed


def _overfit_one_batch(cfg: dict, images: tf.Tensor, labels: tf.Tensor, steps: int = 150, log_every: int = 25) -> float:
    """Train on the same fixed batch repeatedly; return final training accuracy."""
    model = build_custom_cnn(cfg=cfg, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    logs = {"loss": None, "accuracy": None}
    for step in range(1, steps + 1):
        logs = model.train_on_batch(images, labels, return_dict=True)
        if step % log_every == 0 or step == steps:
            print(f"step {step:>4}: loss={logs['loss']:.4f} acc={logs['accuracy']:.4f}")
    return logs["accuracy"]


def main() -> None:
    set_seed(SEED)
    images, labels = load_raw_batch("training")

    print("=== Model 1: dropout=0, l2=0 (should reach >95% on this single batch) ===")
    cfg1 = dict(CNN)
    cfg1["dropout"] = 0.0
    cfg1["l2"] = 0.0
    acc1 = _overfit_one_batch(cfg1, images, labels)

    print()
    print("=== Model 2: real config, dropout+l2 (should reach >90%) ===")
    acc2 = _overfit_one_batch(dict(CNN), images, labels)

    print()
    print(f"Final: model1={acc1:.4f} (need >0.95), model2={acc2:.4f} (need >0.90)")


if __name__ == "__main__":
    main()
