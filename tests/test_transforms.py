import cv2
import numpy as np
import pytest

from src.core.transforms import (
    NEUTRAL_ACTION,
    apply_action,
    apply_brightness,
    apply_contrast,
    apply_gamma,
    apply_saturation,
)


def _gray_img(value: int = 128) -> np.ndarray:
    return np.full((32, 48, 3), value, dtype=np.uint8)


# --- brightness (existing behaviour kept) ---

def test_brightness_zero_is_identity():
    img = _gray_img(128)
    assert np.array_equal(apply_brightness(img, 0.0), img)


def test_brightness_clips_to_white():
    img = _gray_img(200)
    out = apply_brightness(img, 255.0)
    assert (out == 255).all()


def test_brightness_clips_to_black():
    img = _gray_img(50)
    out = apply_brightness(img, -255.0)
    assert (out == 0).all()


# --- contrast ---

def test_contrast_one_is_identity():
    img = _gray_img(100)
    assert np.array_equal(apply_contrast(img, 1.0), img)


def test_contrast_pivots_around_midgray():
    # alpha=2.0 expands distance from 128; pixel at 128 stays put.
    img = _gray_img(128)
    assert np.array_equal(apply_contrast(img, 2.0), img)


def test_contrast_clips():
    img = _gray_img(200)
    out = apply_contrast(img, 5.0)  # (200-128)*5+128 = 488 -> clipped to 255
    assert (out == 255).all()


# --- gamma ---

def test_gamma_one_is_identity():
    img = _gray_img(100)
    # LUT round-trip can drift by 1 on edge values; assert close, not exact.
    assert np.abs(apply_gamma(img, 1.0).astype(int) - img.astype(int)).max() <= 1


def test_gamma_below_one_brightens():
    img = _gray_img(64)
    out = apply_gamma(img, 0.5)
    assert out.mean() > img.mean()


def test_gamma_above_one_darkens():
    img = _gray_img(192)
    out = apply_gamma(img, 2.0)
    assert out.mean() < img.mean()


# --- saturation ---

def test_saturation_zero_is_near_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    out = apply_saturation(img, 0.0)
    # RGB <-> HSV round-trip via uint8 can drift by a few levels per channel.
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 5


def test_saturation_positive_increases_saturation():
    # Build an image whose channels differ so it has non-zero saturation.
    img = np.tile(np.array([[180, 120, 60]], dtype=np.uint8), (16, 16, 1))
    out = apply_saturation(img, 30.0)
    s_in = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].astype(int).mean()
    s_out = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 1].astype(int).mean()
    assert s_out > s_in


# --- apply_action ---

def test_apply_action_neutral_is_identity():
    img = _gray_img(100)
    out = apply_action(img, NEUTRAL_ACTION)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_apply_action_rejects_wrong_size():
    img = _gray_img(100)
    with pytest.raises(ValueError):
        apply_action(img, np.array([0.5, 0.0], dtype=np.float32))


def test_apply_action_dtype_preserved():
    img = _gray_img(100)
    out = apply_action(img, NEUTRAL_ACTION)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_apply_action_brightness_dim_takes_effect():
    img = _gray_img(100)
    out = apply_action(img, np.array([1.0, 20.0, 0.0, 1.0], dtype=np.float32))
    # brightness=+20 should add ~20 to flat gray.
    assert np.abs(out.astype(int).mean() - 120) <= 2
