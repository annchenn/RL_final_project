import cv2
import numpy as np


def apply_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """Multiplicative contrast around midgray. alpha=1.0 is identity."""
    out = (img.astype(np.float32) - 128.0) * float(alpha) + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness(img: np.ndarray, beta: float) -> np.ndarray:
    """Additive brightness shift, clipped to [0, 255]. beta=0 is identity.

    Implemented manually instead of cv2.convertScaleAbs because the latter takes
    abs() before clipping, which silently flips heavy negative shifts into bright
    pixels (e.g. 50 + (-255) -> |-205| = 205 instead of 0).
    """
    shifted = img.astype(np.int16) + int(round(float(beta)))
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Non-linear gamma correction. gamma=1.0 is identity; gamma<1 brightens shadows.

    LUT-based so pow() is paid 256 times per call, not H*W*3 times.
    """
    g = float(gamma)
    if g <= 0:
        g = 1e-3
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return lut[img]


def apply_saturation(img: np.ndarray, delta_s: float) -> np.ndarray:
    """Additive saturation shift in HSV space. delta_s=0 is identity.

    Input/output: HxWx3 uint8 RGB.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 1] = np.clip(hsv[..., 1] + int(round(float(delta_s))), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# Identity action: contrast=1.0, brightness=0, saturation=0, gamma=1.0
NEUTRAL_ACTION = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)


def apply_action(img: np.ndarray, action: np.ndarray) -> np.ndarray:
    """4-D action `[alpha, beta, delta_s, gamma]` applied in a fixed order.

    Order: contrast -> brightness -> gamma -> saturation. Tonal adjustments first
    so saturation operates on the final luminance; matches common photo-editing
    pipelines (e.g. Lightroom basic panel).
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size != 4:
        raise ValueError(f"action must be length 4 [alpha, beta, delta_s, gamma], got {a.size}")
    img = apply_contrast(img, a[0])
    img = apply_brightness(img, a[1])
    img = apply_gamma(img, a[3])
    img = apply_saturation(img, a[2])
    return img
