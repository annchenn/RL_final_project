from typing import Sequence

import numpy as np
import pyiqa
import torch


def _to_tensor(imgs: Sequence[np.ndarray], device: str) -> torch.Tensor:
    """List of HxWx3 uint8 RGB -> NCHW float [0,1] tensor on device."""
    arr = np.stack(imgs, axis=0).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t.to(device, non_blocking=True)


class TopiqReward:
    def __init__(self, device: str = "cuda", metric_name: str = "topiq_nr"):
        self.device = device
        self.metric = pyiqa.create_metric(metric_name, device=device, as_loss=False)
        self.metric.eval()

    @torch.no_grad()
    def score(self, img: np.ndarray) -> float:
        t = _to_tensor([img], self.device)
        return float(self.metric(t).squeeze().detach().cpu().item())

    @torch.no_grad()
    def score_batch(self, imgs: Sequence[np.ndarray]) -> np.ndarray:
        if len(imgs) == 0:
            return np.zeros((0,), dtype=np.float32)
        t = _to_tensor(imgs, self.device)
        out = self.metric(t).detach().cpu().numpy().reshape(-1)
        return out.astype(np.float32)

    def reward(self, img_before: np.ndarray, img_after: np.ndarray) -> float:
        return self.score(img_after) - self.score(img_before)
