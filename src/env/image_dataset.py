from pathlib import Path

import cv2
import numpy as np


class ImageDirDataset:
    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, root: str, image_size: int | None = None):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTS
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {self.root}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        path = self.paths[i]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.image_size is None:
            return rgb
        return self._resize_and_crop(rgb, self.image_size)

    @staticmethod
    def _resize_and_crop(img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = size / min(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (new_h - size) // 2
        left = (new_w - size) // 2
        return resized[top : top + size, left : left + size]
