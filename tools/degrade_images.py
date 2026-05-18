"""
uv run ./tools/degrade_images.py `
  --src_dir ./org-data/landscape_dataset/forest_landscape `
  --dest_dir ./adjusted-data/new `
  --alpha_range 0.75 1.30 `
  --beta_range -30 30 `
  --delta_s_range -25 25 `
  --gamma_range 0.75 1.35
"""

import os
import cv2
import csv
import argparse
import random
import numpy as np
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def adjust_contrast_brightness(img, alpha=1.0, beta=0.0):
    """
    OpenCV formula:
        output = alpha * image + beta

    alpha > 1: increase contrast
    alpha < 1: decrease contrast
    beta > 0: brighter
    beta < 0: darker
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def adjust_saturation(img, delta_s=0.0):
    """
    Adjust saturation in HSV color space.
    delta_s > 0: more saturated
    delta_s < 0: less saturated
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + delta_s, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_gamma(img, gamma=1.0):
    """
    gamma < 1: brighter
    gamma > 1: darker
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)


def clipping_ratio(img):
    """
    Calculate ratio of nearly black or nearly white pixels.
    Used to reject overly destroyed images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    too_dark = np.mean(gray <= 5)
    too_bright = np.mean(gray >= 250)
    return too_dark, too_bright


def random_degrade(
    img,
    alpha_range=(0.50, 1.5),
    beta_range=(-60.0, 60.0),
    delta_s_range=(-50.0, 50.0),
    gamma_range=(0.30, 1.70),
    op_weights=(0.2, 0.4, 0.3, 0.1),
):
    """
    Randomly choose 1 to 4 degradation operations.
    """
    ops_all = ["contrast", "brightness", "saturation", "gamma"]

    # four dimensions
    num_ops = random.choices([1, 2, 3, 4], weights=op_weights, k=1)[0]
    ops = random.sample(ops_all, num_ops)

    # specific dimension parameters
    # ops = ["saturation"]

    alpha = 1.0
    beta = 0.0
    delta_s = 0.0
    gamma = 1.0

    out = img.copy()

    if "contrast" in ops:
        alpha = random.uniform(*alpha_range)

    if "brightness" in ops:
        beta = random.uniform(*beta_range)

    if "contrast" in ops or "brightness" in ops:
        out = adjust_contrast_brightness(out, alpha=alpha, beta=beta)

    if "saturation" in ops:
        delta_s = random.uniform(*delta_s_range)
        out = adjust_saturation(out, delta_s=delta_s)

    if "gamma" in ops:
        gamma = random.uniform(*gamma_range)
        out = adjust_gamma(out, gamma=gamma)

    params = {
        "ops": "+".join(ops),
        "alpha": alpha,
        "beta": beta,
        "delta_s": delta_s,
        "gamma": gamma,
    }

    return out, params


def process_images(args):
    src_dir = Path(args.src_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in src_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ]

    metadata_path = dest_dir / "metadata.csv"

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_path",
                "dest_path",
                "ops",
                "alpha",
                "beta",
                "delta_s",
                "gamma",
                "too_dark_ratio",
                "too_bright_ratio",
            ],
        )
        writer.writeheader()

        for img_path in image_paths:
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue

            degraded = None
            params = None
            too_dark = 0.0
            too_bright = 0.0

            for _ in range(args.max_retries):
                degraded, params = random_degrade(
                    img,
                    alpha_range=tuple(args.alpha_range),
                    beta_range=tuple(args.beta_range),
                    delta_s_range=tuple(args.delta_s_range),
                    gamma_range=tuple(args.gamma_range),
                )

                too_dark, too_bright = clipping_ratio(degraded)

                if too_dark <= args.max_clip_ratio and too_bright <= args.max_clip_ratio:
                    break

            rel_path = img_path.relative_to(src_dir)
            out_path = dest_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_path), degraded)

            writer.writerow({
                "src_path": str(img_path),
                "dest_path": str(out_path),
                "ops": params["ops"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "delta_s": params["delta_s"],
                "gamma": params["gamma"],
                "too_dark_ratio": too_dark,
                "too_bright_ratio": too_bright,
            })

            print(
                f"[OK] {img_path} -> {out_path} | "
                f"ops={params['ops']} | "
                f"alpha={params['alpha']:.4f}, "
                f"beta={params['beta']:.2f}, "
                f"delta_s={params['delta_s']:.2f}, "
                f"gamma={params['gamma']:.4f} | "
                f"dark={too_dark:.4f}, bright={too_bright:.4f}"
            )

    print(f"\nDone. Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)

    parser.add_argument("--alpha_range", type=float, nargs=2, default=[0.50, 1.50])
    parser.add_argument("--beta_range", type=float, nargs=2, default=[-60.0, 60.0])
    parser.add_argument("--delta_s_range", type=float, nargs=2, default=[-50.0, 50.0])
    parser.add_argument("--gamma_range", type=float, nargs=2, default=[0.30, 1.70])

    parser.add_argument("--max_clip_ratio", type=float, default=0.08)
    parser.add_argument("--max_retries", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    process_images(args)


if __name__ == "__main__":
    main()