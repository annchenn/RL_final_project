import numpy as np

from src.core.transforms import apply_action, apply_brightness


def _gray_img(value: int = 128) -> np.ndarray:
    return np.full((32, 48, 3), value, dtype=np.uint8)


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


def test_apply_action_uses_first_channel_as_beta():
    img = _gray_img(100)
    out = apply_action(img, np.array([20.0], dtype=np.float32))
    assert (out == 120).all()


def test_apply_action_dtype_preserved():
    img = _gray_img(100)
    out = apply_action(img, np.array([5.0], dtype=np.float32))
    assert out.dtype == np.uint8
    assert out.shape == img.shape
