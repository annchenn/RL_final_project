"""Evaluate any policy on a fixed eval split.

Usage:
    python -m src.eval.run_eval --policy identity --env-config configs/toy_v0.yaml --out results/identity.csv
    python -m src.eval.run_eval --policy ppo --ckpt runs/ppo_toy/ppo_best.zip --env-config configs/toy_v0.yaml --out results/ppo.csv

Baselines lookup table is intentionally minimal -- full baseline implementations
live under src/baselines/ and are owned by the team member working on baselines.
This file just provides the eval harness + a couple of trivial reference policies
(identity, random) so the harness is self-contained and testable.

One row per episode. Each episode runs the env for `episode_horizon` steps and
accumulates the per-step actions; the CSV reports the per-dimension mean action
and the total reward (= final TOPIQ - initial TOPIQ since reward is delta).
"""
import argparse
import csv
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import yaml

from src.core.reward import TopiqReward
from src.core.seeding import set_global_seed
from src.core.transforms import NEUTRAL_ACTION
from src.env.image_dataset import ImageDirDataset
from src.env.photo_env import PhotoTuneEnv
from src.features.histograms import HistogramFeatureExtractor

PolicyFn = Callable[[np.ndarray], np.ndarray]


def _identity_policy(env: PhotoTuneEnv) -> PolicyFn:
    return lambda obs: NEUTRAL_ACTION.copy()


def _random_policy(env: PhotoTuneEnv, rng: np.random.Generator) -> PolicyFn:
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    return lambda obs: rng.uniform(low, high).astype(np.float32)


def _ppo_policy(ckpt_path: str) -> PolicyFn:
    from stable_baselines3 import PPO

    model = PPO.load(ckpt_path, device="auto")

    def _act(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return _act


def build_env(env_cfg: dict) -> PhotoTuneEnv:
    dataset = ImageDirDataset(env_cfg["env"]["image_dir"])
    fx = HistogramFeatureExtractor(
        intensity_bins=env_cfg["features"]["intensity_bins"],
        gradient_bins=env_cfg["features"]["gradient_bins"],
        scales=tuple(env_cfg["features"]["scales"]),
    )
    reward_fn = TopiqReward(
        device=env_cfg["reward"]["device"],
        scale=float(env_cfg["reward"].get("scale", 1.0)),
    )
    return PhotoTuneEnv(env_cfg, dataset, fx, reward_fn)


def _run_episode(env: PhotoTuneEnv, policy_fn: PolicyFn, seed: int,
                 reset_opts: Optional[dict]) -> dict:
    """Run one episode, returning per-episode aggregates and the final image."""
    obs, info = env.reset(seed=seed, options=reset_opts)
    before_img = env._cur_img.copy()
    score_before = info["score_before"]

    actions: List[np.ndarray] = []
    total_reward = 0.0
    score_after = score_before
    terminated = False
    truncated = False
    while not (terminated or truncated):
        a = policy_fn(obs)
        obs, r, terminated, truncated, step_info = env.step(a)
        actions.append(np.asarray(a, dtype=np.float32).reshape(-1))
        total_reward += float(r)
        score_after = step_info["score_after"]

    action_stack = np.stack(actions, axis=0)
    mean_action = action_stack.mean(axis=0)
    return {
        "image_idx": info["image_idx"],
        "n_steps": len(actions),
        "alpha_mean": float(mean_action[0]),
        "beta_mean": float(mean_action[1]),
        "delta_s_mean": float(mean_action[2]),
        "gamma_mean": float(mean_action[3]),
        "score_before": float(score_before),
        "score_after": float(score_after),
        "reward": total_reward,
        "_before_img": before_img,
        "_after_img": env._cur_img,
    }


def eval_policy(
    policy_fn: PolicyFn,
    env: PhotoTuneEnv,
    n_episodes: int,
    seed: int,
    qual_dir: Optional[Path] = None,
    qual_count: int = 5,
    fixed_indices: Optional[List[int]] = None,
) -> List[dict]:
    n = len(fixed_indices) if fixed_indices is not None else n_episodes
    rows: List[dict] = []
    for ep in range(n):
        reset_opts = {"image_idx": fixed_indices[ep]} if fixed_indices is not None else None
        ep_data = _run_episode(env, policy_fn, seed=seed + ep, reset_opts=reset_opts)

        before_img = ep_data.pop("_before_img")
        after_img = ep_data.pop("_after_img")
        row = {"episode": ep, **ep_data}
        rows.append(row)

        if qual_dir is not None and ep < qual_count:
            grid = np.concatenate([before_img, after_img], axis=1)
            grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(qual_dir / f"ep{ep:03d}.png"), grid_bgr)
    return rows


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        "--all-images",
        action="store_true",
        help="Sweep the full dataset deterministically, one episode per image (overrides --n-episodes).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg = yaml.safe_load(open(args.env_config))
    set_global_seed(int(env_cfg.get("seed", 42)))

    env = build_env(env_cfg)

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
        f"[{args.policy}] n={len(rewards)}  "
        f"mean_reward={rewards.mean():+.4f}  std={rewards.std():.4f}  "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
