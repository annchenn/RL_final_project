import numpy as np


def apply_brightness(img: np.ndarray, beta: float) -> np.ndarray:
    """Apply additive brightness shift, clipped to [0, 255]. Input/output: HxWx3 uint8.

    Implemented manually instead of cv2.convertScaleAbs because the latter takes
    abs() before clipping, which silently flips heavy negative shifts into bright
    pixels (e.g. 50 + (-255) -> |-205| = 205 instead of 0).
    """
    shifted = img.astype(np.int16) + int(round(float(beta)))
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_action(img: np.ndarray, action: np.ndarray) -> np.ndarray:
    """1-D action: action[0] = beta. Wrapper so future 4-D extensions only touch this file."""
    return apply_brightness(img, float(action[0]))
