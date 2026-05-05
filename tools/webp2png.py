#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from PIL import Image

def convert_webp_to_png(src_dir, dest_dir):
    # Setup paths
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    if not src_path.exists():
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return

    # Create destination if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Find all webp files
    webp_files = list(src_path.glob("*.webp"))
    
    if not webp_files:
        print(f"No .webp files found in {src_dir}")
        return

    print(f"Found {len(webp_files)} files. Starting conversion...")

    for i, file_path in enumerate(webp_files):
        try:
            # Open and convert
            with Image.open(file_path) as img:
                # Construct output filename
                target_file = dest_path / f"{file_path.stem}.png"
                
                # Convert to RGB if necessary (to avoid issues with transparency in some viewers)
                # But typically for RL, keeping RGBA is fine if the source has it.
                img.save(target_file, "PNG")
                
            if (i + 1) % 50 == 0:
                print(f"Converted {i + 1}/{len(webp_files)} images...")
        
        except Exception as e:
            print(f"Failed to convert {file_path.name}: {e}")

    print(f"Done! Images saved to: {dest_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python webp2png.py <src_dir> <dest_dir>")
    else:
        convert_webp_to_png(sys.argv[1], sys.argv[2])