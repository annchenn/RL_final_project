import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.transforms import apply_action
from src.env.image_dataset import ImageDirDataset


class PhotoTuneEnv(gym.Env):
    """Single-step bandit env: dataset images are already degraded.

    reset()  -> obs (features of degraded image)
    step(a)  -> obs', reward = TOPIQ(image after action) - TOPIQ(degraded image)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg, dataset, feature_extractor, reward_fn):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.fx = feature_extractor
        self.reward_fn = reward_fn

        # TOPIQ runs on the full-resolution image; features are computed on a
        # downsized copy (shape stable, fast histograms). cfg["env"]["image_size"]
        # is the feature-input size, not a global resize knob.
        self.feature_size = int(cfg["env"]["image_size"])

        beta_lo, beta_hi = cfg["env"]["beta_range"]
        self.action_space = spaces.Box(
            low=np.array([beta_lo], dtype=np.float32),
            high=np.array([beta_hi], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.fx.dim,), dtype=np.float32
        )
        self.horizon = int(cfg["env"]["episode_horizon"])

        self._step = 0
        self._cur_img: np.ndarray | None = None
        self._cur_score: float | None = None

    def _features(self, img: np.ndarray) -> np.ndarray:
        small = ImageDirDataset._resize_and_crop(img, self.feature_size)
        return self.fx(small).astype(np.float32)

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
            "action_beta": float(action_np[0]),
        }

        # Cache for any subsequent step within the same episode (T>1 future use).
        self._cur_img = new_img
        self._cur_score = new_score

        return next_obs, float(reward), terminated, truncated, info

    def close(self):
        pass
