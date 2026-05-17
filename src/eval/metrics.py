from typing import Dict, Optional

import numpy as np
import pyiqa
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


class MetricSuite:
    """Unified PSNR / SSIM / TOPIQ / NIQE for RGB uint8 images."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._topiq = pyiqa.create_metric("topiq_nr", device=device, as_loss=False)
        self._niqe = pyiqa.create_metric("niqe", device=device, as_loss=False)
        self._topiq.eval()
        self._niqe.eval()

    @staticmethod
    def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
        return float(peak_signal_noise_ratio(gt, pred, data_range=255))

    @staticmethod
    def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
        return float(
            structural_similarity(gt, pred, channel_axis=2, data_range=255)
        )

    @torch.no_grad()
    def topiq(self, img: np.ndarray) -> float:
        return float(self._topiq(_to_tensor(img, self.device)).item())

    @torch.no_grad()
    def niqe(self, img: np.ndarray) -> float:
        return float(self._niqe(_to_tensor(img, self.device)).item())

    def all(self, pred: np.ndarray, gt: Optional[np.ndarray] = None) -> Dict[str, float]:
        out: Dict[str, float] = {
            "topiq": self.topiq(pred),
            "niqe": self.niqe(pred),
        }
        if gt is not None:
            out["psnr"] = self.psnr(gt, pred)
            out["ssim"] = self.ssim(gt, pred)
        return out
