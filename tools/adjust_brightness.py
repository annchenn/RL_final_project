#!/usr/bin/env python3
from pathlib import Path
import argparse
import random
import sys
from PIL import Image, ImageEnhance

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

def adjust_brightness_pil(img, factor):
    if img is None:
        return None
    # 使用 PIL 的 ImageEnhance 模組，1.0 代表原圖亮度
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def is_image_file(p: Path):
    return p.suffix.lower() in SUPPORTED_EXT

def main():
    parser = argparse.ArgumentParser(description='Randomly adjust brightness using PIL')
    parser.add_argument('--src', '-s', default='.', help='Source folder')
    parser.add_argument('--dest', '-d', default=None, help='Destination folder')
    parser.add_argument('--min', type=float, default=0.1, help='Min factor')
    parser.add_argument('--max', type=float, default=1.9, help='Max factor')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    src = Path(args.src)
    # 這裡幫你寫得更彈性，如果沒給 --dest，就自動建立一個資料夾
    dest = Path(args.dest) if args.dest else src.parent / 'augmented_images'
    dest.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.iterdir() if p.is_file() and is_image_file(p)]

    if not files:
        print(f'No images found in {src}')
        return

    print(f'Found {len(files)} image(s). Saving to: {dest}')
    
    for p in sorted(files):
        try:
            with Image.open(p) as img:
                # 確保轉成 RGB (處理 webp 或 png 透明層時比較穩定)
                img = img.convert('RGB')
                factor = random.uniform(args.min, args.max)
                out = adjust_brightness_pil(img, factor)
                
                out_path = dest / p.name
                # 如果檔案已存在，加上亮度數值避免覆蓋
                if out_path.exists():
                    out_path = dest / f"{p.stem}_b{factor:.2f}{p.suffix}"
                
                out.save(out_path)
                print(f'  {p.name} -> {out_path.name} (factor={factor:.3f})')
        except Exception as e:
            print(f'  Error processing {p.name}: {e}')

if __name__ == '__main__':
    main()