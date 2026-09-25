"""Checks for src/robustness.py's CONDITION_FUNCS: still correct on grayscale (H, W) images
(the only shape every existing caller - src.robustness, src.gradcam - ever passes), and now
also usable on color (H, W, 3) frames (app_collect.py's live scenario preview reuses these
functions rather than duplicating them).
"""
import numpy as np
import pytest

from src.robustness import CONDITION_FUNCS

RNG_SEED = 42


@pytest.mark.parametrize("condition", list(CONDITION_FUNCS))
def test_condition_preserves_grayscale_shape_and_dtype(condition: str) -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(48, 48), dtype=np.uint8)
    out = CONDITION_FUNCS[condition](img, np.random.default_rng(RNG_SEED))
    assert out.shape == (48, 48)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("condition", list(CONDITION_FUNCS))
def test_condition_also_works_on_color_frames(condition: str) -> None:
    """The three functions that used to do `h, w = img.shape` (glare, occlusion, head_pose)
    would raise ValueError on a 3-channel frame before this fix."""
    img = np.random.default_rng(0).integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
    out = CONDITION_FUNCS[condition](img, np.random.default_rng(RNG_SEED))
    assert out.shape == (60, 80, 3)
    assert out.dtype == np.uint8


def test_clean_is_a_true_copy_not_a_view() -> None:
    img = np.full((10, 10), 100, dtype=np.uint8)
    out = CONDITION_FUNCS["clean"](img, np.random.default_rng(RNG_SEED))
    out[0, 0] = 0
    assert img[0, 0] == 100  # mutating the output must not mutate the input


def test_occlusion_zeroes_out_part_of_the_image() -> None:
    img = np.full((48, 48), 200, dtype=np.uint8)
    out = CONDITION_FUNCS["occlusion"](img, np.random.default_rng(RNG_SEED))
    assert (out == 0).sum() > 0        # some pixels were blacked out
    assert (out == 200).sum() > 0      # but not the whole image


def test_low_light_darkens_the_image_on_average() -> None:
    img = np.full((48, 48), 200, dtype=np.uint8)
    out = CONDITION_FUNCS["low_light"](img, np.random.default_rng(RNG_SEED))
    assert out.mean() < img.mean()
