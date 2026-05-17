import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "adjusted-images"
OUTPUT_DIR = PROJECT_ROOT / "random_baseline_outputs"

NUM_TRIALS = 3
BRIGHTNESS_MIN = 0.1
BRIGHTNESS_MAX = 1.9
SEED = 42
USE_GPU_IF_AVAILABLE = False

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def adjust_brightness_opencv(image, brightness_factor):
    adjusted = image.astype(np.float32) * brightness_factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def evaluate_topiq_path(metric, image_path):
    with torch.inference_mode():
        return metric(str(image_path)).item()


def get_image_paths(input_dir):
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if USE_GPU_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("TOPIQ scoring: direct image path, no resize")
    print()

    topiq = pyiqa.create_metric("topiq_nr", device=device)
    image_paths = get_image_paths(INPUT_DIR)

    if not image_paths:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    all_results = []

    for index, image_path in enumerate(image_paths, start=1):
        filename = image_path.name
        image_name = image_path.stem

        print(f"Processing {index}/{len(image_paths)}: {filename}", flush=True)
        degraded_image = cv2.imread(str(image_path))

        if degraded_image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        input_score = evaluate_topiq_path(topiq, image_path)

        best_score = None
        best_factor = None
        best_image = None

        for trial in range(1, NUM_TRIALS + 1):
            brightness_factor = random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
            adjusted_image = adjust_brightness_opencv(degraded_image, brightness_factor)
            trial_output_path = OUTPUT_DIR / f"_tmp_random_trial_{image_name}.png"
            cv2.imwrite(str(trial_output_path), adjusted_image)
            adjusted_score = evaluate_topiq_path(topiq, trial_output_path)

            if best_score is None or adjusted_score > best_score:
                best_score = adjusted_score
                best_factor = brightness_factor
                best_image = adjusted_image

            print(
                f"  trial {trial:02d}/{NUM_TRIALS}: "
                f"factor={brightness_factor:.6f}, TOPIQ={adjusted_score:.6f}",
                flush=True,
            )

        if best_score is None or best_image is None or best_factor is None:
            raise RuntimeError("Random search produced no trial results.")

        improvement = best_score - input_score
        best_output_path = OUTPUT_DIR / f"{image_name}_best_random_brightness.png"
        cv2.imwrite(str(best_output_path), best_image)

        all_results.append({
            "filename": filename,
            "input_topiq": input_score,
            "no_adjustment_output_topiq": input_score,
            "no_adjustment_improvement": 0.0,
            "random_best_factor": best_factor,
            "random_output_topiq": best_score,
            "random_improvement": improvement,
            "best_output_path": str(best_output_path),
        })

        print(f"[{filename}]")
        print(f"  Input TOPIQ: {input_score:.6f}")
        print(f"  Random Best Factor: {best_factor:.6f}")
        print(f"  Random Output TOPIQ: {best_score:.6f}")
        print(f"  Random Improvement: {improvement:.6f}")
        print()

    if not all_results:
        raise RuntimeError("No readable images were processed.")

    avg_input_topiq = sum(r["input_topiq"] for r in all_results) / len(all_results)
    avg_no_adjustment_output_topiq = sum(
        r["no_adjustment_output_topiq"] for r in all_results
    ) / len(all_results)
    avg_no_adjustment_improvement = sum(
        r["no_adjustment_improvement"] for r in all_results
    ) / len(all_results)
    avg_random_output_topiq = sum(
        r["random_output_topiq"] for r in all_results
    ) / len(all_results)
    avg_random_improvement = sum(
        r["random_improvement"] for r in all_results
    ) / len(all_results)

    csv_path = OUTPUT_DIR / "baseline_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename",
            "input_topiq",
            "no_adjustment_output_topiq",
            "no_adjustment_improvement",
            "random_best_factor",
            "random_output_topiq",
            "random_improvement",
            "best_output_path",
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print("====================================")
    print("Average Baseline Results")
    print("====================================")
    print(f"Number of images: {len(all_results)}")
    print()
    print("No Adjustment")
    print(f"  Input TOPIQ: {avg_input_topiq:.3f}")
    print(f"  Output TOPIQ: {avg_no_adjustment_output_topiq:.3f}")
    print(f"  Improvement: {avg_no_adjustment_improvement:.3f}")
    print()
    print("Random Brightness Search")
    print(f"  Input TOPIQ: {avg_input_topiq:.3f}")
    print(f"  Output TOPIQ: {avg_random_output_topiq:.3f}")
    print(f"  Improvement: {avg_random_improvement:+.3f}")
    print()
    print("CSV saved to:", csv_path)
    print()
    print("LaTeX table row:")
    print(
        f"No Adjustment & {avg_input_topiq:.3f} & "
        f"{avg_no_adjustment_output_topiq:.3f} & "
        f"{avg_no_adjustment_improvement:.3f} \\\\"
    )
    print(
        f"Random Brightness Search & {avg_input_topiq:.3f} & "
        f"{avg_random_output_topiq:.3f} & "
        f"{avg_random_improvement:+.3f} \\\\"
    )


if __name__ == "__main__":
    main()
