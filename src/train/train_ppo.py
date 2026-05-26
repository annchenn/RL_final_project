import argparse
import os
from pathlib import Path

import yaml

# Force wandb to write to a known-writable dir under $HOME. The default fallback
# is /tmp/wandb, but on this shared box that's owned by another user. The project
# dir is technically writable but lives on NFS, where wandb's pre-flight
# os.access() check sometimes returns False even when writes succeed.
_WANDB_HOME = Path.home() / ".cache" / "wandb-rl-final"
_WANDB_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(_WANDB_HOME))
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.ppo_sb3 import make_ppo
from src.core.reward import TopiqReward
from src.core.seeding import set_global_seed
from src.env.image_dataset import ImageDirDataset
from src.env.photo_env import PhotoTuneEnv
from src.features.histograms import HistogramFeatureExtractor
from src.train.callbacks import WandbEvalCallback, WandbRewardCallback


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
        help="Override ppo_cfg['log_dir']. scripts/run_ppo.sh always passes a timestamped path.",
    )
    return p.parse_args()


def make_env_fn(env_cfg: dict, reward_fn=None):
    def _make():
        dataset = ImageDirDataset(env_cfg["env"]["image_dir"])
        fx = HistogramFeatureExtractor(
            intensity_bins=env_cfg["features"]["intensity_bins"],
            gradient_bins=env_cfg["features"]["gradient_bins"],
            scales=tuple(env_cfg["features"]["scales"]),
        )
        rfn = reward_fn if reward_fn is not None else TopiqReward(
            device=env_cfg["reward"]["device"],
            scale=float(env_cfg["reward"].get("scale", 1.0)),
        )
        env = PhotoTuneEnv(env_cfg, dataset, fx, rfn)
        return Monitor(env)  # SB3 reads ep_info from Monitor

    return _make


def _update_latest_symlink(log_dir: Path) -> None:
    """Maintain `<log_dir.parent>/latest` symlink pointing at this run so
    eval.sh can find the most recent run without globbing."""
    link = log_dir.parent / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(log_dir.name, target_is_directory=True)
    except OSError as e:
        # NFS or Windows may refuse symlinks; not worth crashing training over.
        print(f"[train_ppo] WARN: could not update latest symlink ({e})")


def main():
    args = parse_args()
    env_cfg = yaml.safe_load(open(args.env_config))
    ppo_cfg = yaml.safe_load(open(args.ppo_config))

    set_global_seed(int(env_cfg.get("seed", 42)))

    log_dir = Path(args.log_dir) if args.log_dir else Path(ppo_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # One TOPIQ model on GPU, shared between training and eval envs to avoid OOM.
    shared_reward = TopiqReward(
        device=env_cfg["reward"]["device"],
        scale=float(env_cfg["reward"].get("scale", 1.0)),
    )

    model, vec = make_ppo(
        make_env_fn(env_cfg, reward_fn=shared_reward),
        ppo_cfg,
    )

    wandb_cb = WandbRewardCallback(
        project=ppo_cfg.get("wandb_project", "rl-photo-tune"),
        run_name=args.run_name,
        cfg={**env_cfg, **ppo_cfg},
        enabled=not args.no_wandb,
    )

    eval_env = DummyVecEnv(
        [make_env_fn(env_cfg, reward_fn=shared_reward)]
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=int(ppo_cfg.get("eval_freq", 500)),
        n_eval_episodes=int(ppo_cfg.get("n_eval_episodes", 20)),
        deterministic=True,
        callback_after_eval=WandbEvalCallback(wandb_cb),
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=int(ppo_cfg.get("save_freq", 5000)),
        save_path=str(ckpt_dir),
        name_prefix="ppo",
    )

    total = int(args.total_steps if args.total_steps is not None else ppo_cfg["total_timesteps"])
    model.learn(
        total_timesteps=total,
        callback=CallbackList([wandb_cb, eval_cb, ckpt_cb]),
        log_interval=int(ppo_cfg["log_interval"]),
    )

    # EvalCallback writes the best-by-eval-reward weights to <log_dir>/best_model.zip
    # whenever a new high is reached during training. Save the final-step weights
    # separately so reports can compare best-eval vs end-of-training behavior.
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"[train_ppo] best-by-eval -> {log_dir / 'best_model.zip'}")
    print(f"[train_ppo] final-step   -> {final_path}")
    print(f"[train_ppo] intermediate -> {ckpt_dir}/")

    _update_latest_symlink(log_dir)

    eval_env.close()
    vec.close()


if __name__ == "__main__":
    main()
