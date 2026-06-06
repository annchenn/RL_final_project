"""Side-by-side comparison: degraded (left) | enhanced (right).

Matches files by filename stem, so `beach_landscape_10.jpg` (degraded) is paired
with `beach_landscape_10.png` (enhanced). Output filename keeps the same stem.

Usage:
    python tools/make_comparison.py \\
        --degraded-dir degraded-data/landscape-50-test \\
        --enhanced-dir results/ppo_20260523_174656/best_model \\
        --out-dir results/ppo_20260523_174656/best_model/comparison

Defaults to the most recent run via runs/latest. Override any path with the flags.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--degraded-dir", default="degraded-data/landscape-50-test")
    p.add_argument(
        "--enhanced-dir",
        required=True,
        help="Folder of enhanced images (e.g. results/<run>/<ckpt>/).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to <enhanced-dir>/comparison/.",
    )
    p.add_argument(
        "--separator-width",
        type=int,
        default=0,
        help="Pixel width of a vertical bar between the two images. 0 = no bar.",
    )
    p.add_argument(
        "--separator-color",
        default="white",
        choices=["white", "black"],
    )
    return p.parse_args()


def index_by_stem(folder: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def main():
    args = parse_args()
    degraded_dir = Path(args.degraded_dir)
    enhanced_dir = Path(args.enhanced_dir)
    out_dir = Path(args.out_dir) if args.out_dir else enhanced_dir / "comparison"

    if not degraded_dir.is_dir():
        raise SystemExit(f"degraded dir not found: {degraded_dir}")
    if not enhanced_dir.is_dir():
        raise SystemExit(f"enhanced dir not found: {enhanced_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    degraded_by_stem = index_by_stem(degraded_dir)
    enhanced_by_stem = index_by_stem(enhanced_dir)

    written = 0
    skipped: list[str] = []
    for stem in sorted(enhanced_by_stem):
        deg_path = degraded_by_stem.get(stem)
        if deg_path is None:
            skipped.append(enhanced_by_stem[stem].name)
            continue

        deg = cv2.imread(str(deg_path))
        enh = cv2.imread(str(enhanced_by_stem[stem]))
        # In case dimensions diverged (shouldn't, but cheap insurance):
        if deg.shape != enh.shape:
            enh = cv2.resize(enh, (deg.shape[1], deg.shape[0]))

        if args.separator_width > 0:
            color = 255 if args.separator_color == "white" else 0
            sep = np.full(
                (deg.shape[0], args.separator_width, 3), color, dtype=np.uint8
            )
            combined = np.concatenate([deg, sep, enh], axis=1)
        else:
            combined = np.concatenate([deg, enh], axis=1)

        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), combined)
        written += 1
        print(f"[+] {out_path}")

    print(f"\nWrote {written} comparisons -> {out_dir}/")
    if skipped:
        head = ", ".join(skipped[:3])
        more = f" (+{len(skipped) - 3} more)" if len(skipped) > 3 else ""
        print(f"Skipped {len(skipped)} (no matching degraded): {head}{more}")


if __name__ == "__main__":
    main()
