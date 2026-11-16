import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SurrogateExerciseEnv(gym.Env):
    """
    DEMO ONLY.
    Action effects are configurable placeholders, not validated physiology.
    """
    metadata = {"render_modes":[]}

    def __init__(self, states, columns, cfg, observation_columns=None):
        super().__init__()
        self.base = np.asarray(states, np.float32)
        self.columns = list(columns)
        self.observation_columns = (
            list(self.columns) if observation_columns is None else list(observation_columns)
        )
        missing = [name for name in self.observation_columns if name not in self.columns]
        if missing:
            raise ValueError(f"Unknown observation columns: {missing}")
        if not self.observation_columns:
            raise ValueError("At least one observation column is required")
        self.observation_indices = np.asarray(
            [self.columns.index(name) for name in self.observation_columns], dtype=int
        )
        self.cfg = cfg
        self.mean = self.base.mean(0)
        self.std = self.base.std(0)+1e-6
        self.z = (self.base-self.mean)/self.std
        self.observations = self.z[:, self.observation_indices]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-20, high=20,
            shape=(len(self.observation_columns),), dtype=np.float32
        )

        hr_idx = self.columns.index("HR_current")
        q0, q1 = cfg["rl_demo"]["reward"]["hr_quantiles"]
        self.hr_low = float(np.quantile(self.z[:,hr_idx], q0))
        self.hr_high = float(np.quantile(self.z[:,hr_idx], q1))
        self.t = 0
        self.prev_action = 1
        self.last_full_state = self.z[0].copy()

    def _observe(self, full_state):
        return np.asarray(full_state, np.float32)[self.observation_indices].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.prev_action = 1
        self.last_full_state = self.z[0].copy()
        return self._observe(self.last_full_state), {}

    def _apply_effect(self, state, action):
        label = ["reduce","maintain","increase"][int(action)]
        effect = self.cfg["rl_demo"]["action_effect"][label]
        out = state.copy()
        for key, delta in effect.items():
            col = "HR_current" if key=="HR" else f"{key}_current"
            if col in self.columns:
                out[self.columns.index(col)] += float(delta)
        return out

    def _risk(self, state):
        risk = 0.0
        n = 0
        uw = float(self.cfg["rl_demo"]["reward"]["uncertainty_weight"])
        for i, name in enumerate(self.columns):
            if "_pred_end_delta" in name:
                risk += abs(float(state[i]))
                n += 1
            elif "_endpoint_sd" in name:
                risk += uw*abs(float(state[i]))
        return risk/max(n,1)

    def risk(self, state):
        return float(self._risk(np.asarray(state,float)))

    def hr_zone(self, state):
        hr=float(np.asarray(state,float)[self.columns.index("HR_current")])
        if hr < self.hr_low:
            return "below_demo_range"
        if hr > self.hr_high:
            return "above_demo_range"
        return "within_demo_range"

    def step(self, action):
        state = self._apply_effect(self.z[self.t], action)
        self.last_full_state = state.copy()
        hr = state[self.columns.index("HR_current")]
        hr_reward = 1.0 if self.hr_low <= hr <= self.hr_high else -1.0

        reward = (
            hr_reward
            - float(self.cfg["rl_demo"]["reward"]["future_risk_weight"])*self._risk(state)
            - float(self.cfg["rl_demo"]["reward"]["action_change_penalty"])*abs(int(action)-self.prev_action)
        )

        self.prev_action = int(action)
        self.t += 1
        done = self.t >= len(self.z)
        next_full_state = state if done else self.z[self.t]
        return self._observe(next_full_state), float(reward), done, False, {}
