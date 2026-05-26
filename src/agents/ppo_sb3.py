from typing import Callable, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def make_ppo(env_fn: Callable[[], "object"], cfg: dict) -> Tuple[PPO, VecEnv]:
    """Build a vectorised env and SB3 PPO model.

    Note: pyiqa TOPIQ lives on GPU. SubprocVecEnv forks a fresh CUDA context per worker
    which can OOM; default to DummyVecEnv unless n_envs > 1 is explicitly requested AND
    the user has verified GPU memory headroom.
    """
    n_envs = int(cfg.get("n_envs", 1))
    vec: VecEnv
    if n_envs > 1:
        vec = SubprocVecEnv([env_fn for _ in range(n_envs)])
    else:
        vec = DummyVecEnv([env_fn])

    # tensorboard_log intentionally omitted: we use wandb only. Re-enable by
    # passing tensorboard_log=cfg.get("log_dir") if you want SB3's TB writer back.
    model = PPO(
        cfg.get("policy", "MlpPolicy"),
        vec,
        learning_rate=float(cfg["learning_rate"]),
        n_steps=int(cfg["n_steps"]),
        batch_size=int(cfg["batch_size"]),
        n_epochs=int(cfg["n_epochs"]),
        gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg["gae_lambda"]),
        clip_range=float(cfg["clip_range"]),
        ent_coef=float(cfg["ent_coef"]),
        vf_coef=float(cfg["vf_coef"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        verbose=1,
    )
    return model, vec
