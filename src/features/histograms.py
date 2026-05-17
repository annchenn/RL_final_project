from typing import Sequence

import cv2
import numpy as np


class HistogramFeatureExtractor:
    """Multi-scale grayscale intensity + Sobel-magnitude gradient histograms.

    Output is a single 1-D float32 vector with all scales / channels concatenated,
    each sub-histogram normalised to a probability distribution.
    """

    def __init__(
        self,
        intensity_bins: int = 64,
        gradient_bins: int = 64,
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        max_grad_mag: float = 4 * 255.0,  # rough upper bound for Sobel magnitude on uint8
    ):
        self.intensity_bins = intensity_bins
        self.gradient_bins = gradient_bins
        self.scales = tuple(scales)
        self.max_grad_mag = max_grad_mag
        self.dim = len(self.scales) * (intensity_bins + gradient_bins)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """img: HxWx3 uint8 RGB -> (self.dim,) float32 in [0, 1]."""
        feats = []
        for s in self.scales:
            scaled = self._resize(img, s)
            gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
            feats.append(self._intensity_hist(gray))
            feats.append(self._gradient_hist(gray))
        return np.concatenate(feats, axis=0).astype(np.float32)

    @staticmethod
    def _resize(img: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return img
        h, w = img.shape[:2]
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    def _intensity_hist(self, gray: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([gray], [0], None, [self.intensity_bins], [0, 256]).reshape(-1)
        s = hist.sum()
        return hist / s if s > 0 else hist

    def _gradient_hist(self, gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_clipped = np.clip(mag, 0.0, self.max_grad_mag)
        hist, _ = np.histogram(
            mag_clipped, bins=self.gradient_bins, range=(0.0, self.max_grad_mag)
        )
        s = hist.sum()
        return (hist / s) if s > 0 else hist.astype(np.float32)
