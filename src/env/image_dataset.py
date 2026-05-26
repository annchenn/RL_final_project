from pathlib import Path

import cv2
import numpy as np


class ImageDirDataset:
    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, root: str):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTS
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        path = self.paths[i]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
