from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WandbRewardCallback(BaseCallback):
    """Lightweight WandB logger. Uploads rolling mean reward + episode count.

    Avoids importing wandb at module load so this file is safe even when wandb is
    disabled. Pass `enabled=False` to make every callback hook a no-op.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        cfg: dict,
        enabled: bool = True,
        log_every: int = 256,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.project = project
        self.run_name = run_name
        self.cfg = cfg
        self.enabled = enabled
        self.log_every = log_every
        self._wandb = None
        self._ep_rewards: list[float] = []
        self._ep_count = 0
        self._last_logged_step = 0

    def _on_training_start(self) -> None:
        if not self.enabled:
            return
        import wandb  # local import keeps wandb optional

        self._wandb = wandb
        wandb.init(project=self.project, name=self.run_name, config=self.cfg)

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        infos = self.locals.get("infos", []) or []
        for info in infos:
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is not None:
                self._ep_rewards.append(float(ep.get("r", 0.0)))
                self._ep_count += 1

        steps_since = self.num_timesteps - self._last_logged_step
        if steps_since >= self.log_every and self._ep_rewards:
            window = self._ep_rewards[-100:]
            payload = {
                "train/mean_ep_reward_last100": float(np.mean(window)),
                "train/std_ep_reward_last100": float(np.std(window)),
                "train/ep_count": self._ep_count,
                "train/timesteps": self.num_timesteps,
            }
            self._wandb.log(payload, step=self.num_timesteps)
            self._last_logged_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


class WandbEvalCallback(BaseCallback):
    """Pushes EvalCallback results to wandb after each in-loop evaluation.

    Wire as `EvalCallback(..., callback_after_eval=WandbEvalCallback(wandb_cb))`.
    Reads `self.parent.last_mean_reward` (SB3 sets `parent` on the child callback
    when constructing EvalCallback) so we get the freshly-computed eval reward
    instead of polling stale state.
    """

    def __init__(self, wandb_cb: "WandbRewardCallback", verbose: int = 0):
        super().__init__(verbose)
        self.wandb_cb = wandb_cb

    def _on_step(self) -> bool:
        if not self.wandb_cb.enabled or self.wandb_cb._wandb is None:
            return True
        ec = self.parent  # the EvalCallback that owns us
        self.wandb_cb._wandb.log(
            {
                "eval/mean_reward": float(ec.last_mean_reward),
                "eval/best_mean_reward": float(ec.best_mean_reward),
                "eval/timesteps": self.num_timesteps,
            },
            step=self.num_timesteps,
        )
        return True
