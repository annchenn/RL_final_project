import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from src.core.reward import TopiqReward


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TOPIQ-NR scores for every image in a folder."
    )
    parser.add_argument("input_dir", help="Folder containing images to evaluate.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path. Example: results/topiq_scores.csv",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for TOPIQ. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier for TOPIQ scores. Use 100 to match training config scaled units.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also evaluate images in subfolders.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "path", "topiq"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    paths = image_paths(input_dir, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"No images found in: {input_dir}")

    device = resolve_device(args.device)
    scorer = TopiqReward(device=device, scale=args.scale)

    rows = []
    print(f"Input dir: {input_dir}")
    print(f"Device: {device}")
    print(f"Scale: {args.scale:g}")
    print()

    for idx, path in enumerate(paths, start=1):
        img = read_rgb(path)
        score = scorer.score(img)
        row = {
            "filename": path.name,
            "path": str(path),
            "topiq": score,
        }
        rows.append(row)
        print(f"[{idx:03d}/{len(paths):03d}] {path.name}: {score:.6f}", flush=True)

    scores = np.array([row["topiq"] for row in rows], dtype=np.float64)
    mean_score = float(scores.mean())

    print()
    print("====================================")
    print("TOPIQ Results")
    print("====================================")
    print(f"Number of images: {len(rows)}")
    print(f"Average TOPIQ: {mean_score:.6f}")

    if args.out:
        out_path = Path(args.out)
        write_csv(rows, out_path)
        print(f"CSV saved to: {out_path}")


if __name__ == "__main__":
    main()
