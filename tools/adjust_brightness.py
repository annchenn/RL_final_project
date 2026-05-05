#!/usr/bin/env python3
from pathlib import Path
import argparse
import random
import cv2
import numpy as np

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def is_image_file(p: Path):
    return p.suffix.lower() in SUPPORTED_EXT


def adjust_brightness_cv2(img, factor):
    """
    使用 OpenCV 調整亮度。
    factor = 1.0 代表原圖
    factor < 1.0 變暗
    factor > 1.0 變亮
    """
    if img is None:
        return None

    # 先轉成 float 避免 uint8 溢位
    out = img.astype(np.float32) * factor

    # 限制範圍在 0~255
    out = np.clip(out, 0, 255)

    # 轉回 uint8
    out = out.astype(np.uint8)

    return out


def main():
    parser = argparse.ArgumentParser(description='Randomly adjust brightness using OpenCV')
    parser.add_argument('--src', '-s', default='.', help='Source folder')
    parser.add_argument('--dest', '-d', default=None, help='Destination folder')
    parser.add_argument('--min', type=float, default=0.1, help='Min brightness factor')
    parser.add_argument('--max', type=float, default=1.9, help='Max brightness factor')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    src = Path(args.src)

    if not src.exists():
        print(f'Source folder does not exist: {src}')
        return

    dest = Path(args.dest) if args.dest else src.parent / 'augmented_images'
    dest.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.iterdir() if p.is_file() and is_image_file(p)]

    if not files:
        print(f'No images found in {src}')
        return

    print(f'Found {len(files)} image(s). Saving to: {dest}')

    for p in sorted(files):
        try:
            # OpenCV 讀進來是 BGR 格式
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)

            if img is None:
                print(f'  Error reading {p.name}')
                continue

            factor = random.uniform(args.min, args.max)

            out = adjust_brightness_cv2(img, factor)

            out_path = dest / p.name

            # 避免覆蓋
            if out_path.exists():
                out_path = dest / f"{p.stem}_b{factor:.2f}{p.suffix}"

            success = cv2.imwrite(str(out_path), out)

            if success:
                print(f'  {p.name} -> {out_path.name} (factor={factor:.3f})')
            else:
                print(f'  Error saving {out_path.name}')

        except Exception as e:
            print(f'  Error processing {p.name}: {e}')


if __name__ == '__main__':
    main()