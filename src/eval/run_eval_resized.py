"""Evaluate any policy on a fixed eval split, with images pre-resized.

Same harness as src.eval.run_eval, except the dataset returns resized+cropped
images (via ImageDirDataset(image_size=...)) so TOPIQ and the action are
applied on the resized image, not the full-resolution one.

Usage:
    python -m src.eval.run_eval_resized --policy identity --env-config configs/toy_v0.yaml --out results/identity_resized.csv
    python -m src.eval.run_eval_resized --policy ppo --ckpt runs/ppo_toy/ppo_best.zip --env-config configs/toy_v0.yaml --out results/ppo_resized.csv
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from src.core.reward import TopiqReward
from src.core.seeding import set_global_seed
from src.env.image_dataset import ImageDirDataset
from src.env.photo_env import PhotoTuneEnv
from src.eval.run_eval import (
    _identity_policy,
    _ppo_policy,
    _random_policy,
    eval_policy,
    write_csv,
)
from src.features.histograms import HistogramFeatureExtractor


def build_env_resized(env_cfg: dict, image_size: int) -> PhotoTuneEnv:
    dataset = ImageDirDataset(env_cfg["env"]["image_dir"], image_size=image_size)
    fx = HistogramFeatureExtractor(
        intensity_bins=env_cfg["features"]["intensity_bins"],
        gradient_bins=env_cfg["features"]["gradient_bins"],
        scales=tuple(env_cfg["features"]["scales"]),
    )
    reward_fn = TopiqReward(device=env_cfg["reward"]["device"])
    return PhotoTuneEnv(env_cfg, dataset, fx, reward_fn)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, choices=["identity", "random", "ppo"])
    p.add_argument("--ckpt", default=None, help="PPO checkpoint path (required for --policy ppo)")
    p.add_argument("--env-config", default="configs/toy_v0.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--qual-dir", default=None)
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Short-side size for resize+center-crop applied to dataset images. "
             "Defaults to env.image_size from the config.",
    )
    p.add_argument(
        "--all-images",
        action="store_true",
        help="Sweep the full dataset deterministically, one episode per image (overrides --n-episodes).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg = yaml.safe_load(open(args.env_config))
    set_global_seed(int(env_cfg.get("seed", 42)))

    image_size = int(args.image_size) if args.image_size is not None else int(env_cfg["env"]["image_size"])
    env = build_env_resized(env_cfg, image_size=image_size)

    if args.policy == "identity":
        policy = _identity_policy(env)
    elif args.policy == "random":
        policy = _random_policy(env, np.random.default_rng(args.seed))
    elif args.policy == "ppo":
        if not args.ckpt:
            raise SystemExit("--ckpt is required when --policy=ppo")
        policy = _ppo_policy(args.ckpt)
    else:
        raise ValueError(f"unknown policy: {args.policy}")

    qual_dir = Path(args.qual_dir) if args.qual_dir else None
    if qual_dir is not None:
        qual_dir.mkdir(parents=True, exist_ok=True)

    fixed_indices = list(range(len(env.dataset))) if args.all_images else None

    rows = eval_policy(
        policy_fn=policy,
        env=env,
        n_episodes=int(args.n_episodes),
        seed=int(args.seed),
        qual_dir=qual_dir,
        fixed_indices=fixed_indices,
    )
    write_csv(rows, Path(args.out))

    rewards = np.array([r["reward"] for r in rows], dtype=np.float64)
    print(
        f"[{args.policy} | resized@{image_size}] n={len(rewards)}  "
        f"mean_reward={rewards.mean():+.4f}  std={rewards.std():.4f}  "
        f"out={args.out}"
    )

    inp = float(np.mean([r["score_before"] for r in rows]))
    out = float(np.mean([r["score_after"] for r in rows]))
    print()
    print(f"=== Our RL Model (n={len(rows)} images, resized@{image_size}) ===")
    print(f"Input TOPIQ:   {inp:.4f}")
    print(f"Output TOPIQ:  {out:.4f}")
    print(f"Improvement:   {out - inp:+.4f}")


if __name__ == "__main__":
    main()
