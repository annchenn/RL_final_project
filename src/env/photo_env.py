import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.transforms import apply_action


class PhotoTuneEnv(gym.Env):
    """Multi-step env for photo parameter tuning.

    reset()  -> obs (features of degraded image)
    step(a)  -> obs', reward = TOPIQ(image after action) - TOPIQ(image before action)

    Action is 4-D continuous `[alpha, beta, delta_s, gamma]`. Each step applies
    the action to the *current* image, so over `episode_horizon` steps the
    effects accumulate (contrast/gamma multiplicatively, brightness/saturation
    additively).
    """

    metadata = {"render_modes": []}

    ACTION_KEYS = ("alpha", "beta", "delta_s", "gamma")

    def __init__(self, cfg, dataset, feature_extractor, reward_fn):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.fx = feature_extractor
        self.reward_fn = reward_fn

        env_cfg = cfg["env"]
        lows = np.array(
            [
                env_cfg["alpha_range"][0],
                env_cfg["beta_range"][0],
                env_cfg["delta_s_range"][0],
                env_cfg["gamma_range"][0],
            ],
            dtype=np.float32,
        )
        highs = np.array(
            [
                env_cfg["alpha_range"][1],
                env_cfg["beta_range"][1],
                env_cfg["delta_s_range"][1],
                env_cfg["gamma_range"][1],
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=lows, high=highs, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.fx.dim,), dtype=np.float32
        )
        self.horizon = int(cfg["env"]["episode_horizon"])

        self._step = 0
        self._cur_img: np.ndarray | None = None
        self._cur_score: float | None = None

    def _features(self, img: np.ndarray) -> np.ndarray:
        return self.fx(img).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "image_idx" in options:
            idx = int(options["image_idx"])
        else:
            idx = int(self.np_random.integers(0, len(self.dataset)))
        self._cur_img = self.dataset[idx]
        self._cur_score = float(self.reward_fn.score(self._cur_img))
        self._step = 0

        obs = self._features(self._cur_img)
        info = {
            "image_idx": idx,
            "score_before": self._cur_score,
        }
        return obs, info

    def step(self, action):
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        new_img = apply_action(self._cur_img, action_np)
        new_score = float(self.reward_fn.score(new_img))
        reward = new_score - self._cur_score

        self._step += 1
        terminated = self._step >= self.horizon
        truncated = False
        next_obs = self._features(new_img)

        info = {
            "score_after": new_score,
            "score_before": self._cur_score,
            "action_alpha": float(action_np[0]),
            "action_beta": float(action_np[1]),
            "action_delta_s": float(action_np[2]),
            "action_gamma": float(action_np[3]),
        }

        # Cache for the next step within the same episode.
        self._cur_img = new_img
        self._cur_score = new_score

        return next_obs, float(reward), terminated, truncated, info

    def close(self):
        pass
