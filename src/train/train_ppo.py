import argparse
import os
import shutil
from pathlib import Path

import yaml

# Force wandb to write to a known-writable dir under $HOME. The default fallback
# is /tmp/wandb, but on this shared box that's owned by another user. The project
# dir is technically writable but lives on NFS, where wandb's pre-flight
# os.access() check sometimes returns False even when writes succeed.
_WANDB_HOME = Path.home() / ".cache" / "wandb-rl-final"
_WANDB_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(_WANDB_HOME))
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_sb3 import make_ppo
from src.core.reward import TopiqReward
from src.core.seeding import set_global_seed
from src.env.image_dataset import ImageDirDataset
from src.env.photo_env import PhotoTuneEnv
from src.features.histograms import HistogramFeatureExtractor
from src.train.callbacks import WandbRewardCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/toy_v0.yaml")
    p.add_argument("--ppo-config", default="configs/ppo_default.yaml")
    p.add_argument("--total-steps", type=int, default=None, help="Override total_timesteps")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--run-name", default="ppo_toy_v0")
    p.add_argument(
        "--log-dir",
        default=None,
        help="Override ppo_cfg['log_dir']. Use to keep separate runs from overwriting each other.",
    )
    p.add_argument(
        "--full-res",
        action="store_true",
        help="Train and in-loop eval on full-resolution images (no dataset resize). "
             "Heavy: TOPIQ runs on full-res. Default off.",
    )
    return p.parse_args()


def make_env_fn(env_cfg: dict, image_size: int | None = None, reward_fn=None):
    def _make():
        dataset = ImageDirDataset(env_cfg["env"]["image_dir"], image_size=image_size)
        fx = HistogramFeatureExtractor(
            intensity_bins=env_cfg["features"]["intensity_bins"],
            gradient_bins=env_cfg["features"]["gradient_bins"],
            scales=tuple(env_cfg["features"]["scales"]),
        )
        rfn = reward_fn if reward_fn is not None else TopiqReward(device=env_cfg["reward"]["device"])
        env = PhotoTuneEnv(env_cfg, dataset, fx, rfn)
        return Monitor(env)  # SB3 reads ep_info from Monitor

    return _make


def main():
    args = parse_args()
    env_cfg = yaml.safe_load(open(args.env_config))
    ppo_cfg = yaml.safe_load(open(args.ppo_config))

    set_global_seed(int(env_cfg.get("seed", 42)))

    log_dir = Path(args.log_dir) if args.log_dir else Path(ppo_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # One TOPIQ model on GPU, shared between training and eval envs to avoid OOM.
    shared_reward = TopiqReward(device=env_cfg["reward"]["device"])

    # Training uses resized images (full-res TOPIQ is too heavy when other GPU
    # jobs are running). Final post-training eval (scripts/eval.sh) reports
    # full-res numbers via run_eval.py, which builds its own full-res dataset.
    # Pass --full-res to force full-resolution training instead.
    train_image_size = None if args.full_res else int(env_cfg["env"]["image_size"])
    model, vec = make_ppo(
        make_env_fn(env_cfg, image_size=train_image_size, reward_fn=shared_reward),
        ppo_cfg,
    )

    wandb_cb = WandbRewardCallback(
        project=ppo_cfg.get("wandb_project", "rl-photo-tune"),
        run_name=args.run_name,
        cfg={**env_cfg, **ppo_cfg},
        enabled=not args.no_wandb,
    )

    eval_image_size = None if args.full_res else ppo_cfg.get("eval_image_size", None)
    eval_env = DummyVecEnv(
        [make_env_fn(env_cfg, image_size=eval_image_size, reward_fn=shared_reward)]
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=int(ppo_cfg.get("eval_freq", 500)),
        n_eval_episodes=int(ppo_cfg.get("n_eval_episodes", 20)),
        deterministic=True,
        verbose=1,
    )

    total = int(args.total_steps if args.total_steps is not None else ppo_cfg["total_timesteps"])
    model.learn(
        total_timesteps=total,
        callback=CallbackList([wandb_cb, eval_cb]),
        log_interval=int(ppo_cfg["log_interval"]),
    )

    # EvalCallback saves the best-by-eval-reward model as best_model.zip.
    # Promote it to ppo_best.zip (the canonical name) and save the final
    # post-training weights separately as ppo_final.zip for reference.
    best_eval_zip = log_dir / "best_model.zip"
    if best_eval_zip.exists():
        shutil.copy(best_eval_zip, log_dir / "ppo_best.zip")
        model.save(str(log_dir / "ppo_final"))
        print(f"[train_ppo] best-by-eval checkpoint -> {log_dir / 'ppo_best.zip'}")
        print(f"[train_ppo] final checkpoint        -> {log_dir / 'ppo_final.zip'}")
    else:
        model.save(str(log_dir / "ppo_best"))
        print(f"[train_ppo] saved checkpoint to {log_dir / 'ppo_best.zip'} (no eval ran)")

    eval_env.close()
    vec.close()


if __name__ == "__main__":
    main()
