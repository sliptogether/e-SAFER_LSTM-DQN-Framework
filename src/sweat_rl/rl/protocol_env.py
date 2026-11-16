"""Protocol-informed surrogate environment for the Day 4 exercise schedule.

This module does not alter measured sensor values.  It estimates a one-minute
exercise-versus-rest response contrast from the denoised Day 4 trajectory and
uses that contrast for transparent counterfactual action projections.
"""

import numpy as np

from sweat_rl.rl.env import SurrogateExerciseEnv


def protocol_phase(minutes):
    """Return 1 for exercise and 0 for rest using the confirmed Day 4 protocol."""
    minutes = np.asarray(minutes, dtype=int)
    return (
        (minutes < 15)
        | ((minutes >= 20) & (minutes < 40))
        | (minutes >= 45)
    ).astype(int)


def estimate_protocol_response(processed_day4, state_df, features, horizon=5):
    """Estimate action-response coefficients in normalized RL-state units."""
    day = processed_day4.sort_values("Minute").reset_index(drop=True).copy()
    day["Protocol_phase"] = protocol_phase(day["Minute"].to_numpy(int))
    same_phase = day["Protocol_phase"].eq(day["Protocol_phase"].shift())
    contrasts = {}
    for feature in [*features, "HR"]:
        change = day[feature].diff()
        exercise = change[same_phase & day.Protocol_phase.eq(1)].median()
        rest = change[same_phase & day.Protocol_phase.eq(0)].median()
        contrasts[feature] = float(exercise - rest)

    state_columns = [column for column in state_df.columns if column != "Origin_min"]
    state_std = state_df[state_columns].std(ddof=0).replace(0, 1e-6)
    response = {column: 0.0 for column in state_columns}
    response["HR_current"] = float(
        np.clip(contrasts["HR"] / state_std["HR_current"], -1.5, 1.5)
    )
    for feature in features:
        contrast = contrasts[feature]
        multipliers = {
            f"{feature}_current": 1.0,
            f"{feature}_pred_end_delta{horizon}": float(horizon),
            f"{feature}_pred_mean_delta{horizon}": float(horizon + 1) / 2.0,
            f"{feature}_pred_slope{horizon}": 1.0,
        }
        for column, multiplier in multipliers.items():
            if column in response:
                response[column] = float(
                    np.clip(multiplier * contrast / state_std[column], -1.5, 1.5)
                )
    details = {
        "protocol": [
            {"start": 0, "end": 15, "phase": "exercise"},
            {"start": 15, "end": 20, "phase": "rest"},
            {"start": 20, "end": 40, "phase": "exercise"},
            {"start": 40, "end": 45, "phase": "rest"},
            {"start": 45, "end": 55, "phase": "exercise"},
        ],
        "median_one_minute_exercise_minus_rest": contrasts,
        "normalized_state_response": response,
    }
    return response, details


class ProtocolInformedSurrogateEnv(SurrogateExerciseEnv):
    """DQN environment with protocol-derived action effects on all five signals."""

    def __init__(
        self,
        states,
        columns,
        cfg,
        minutes,
        response,
        features,
        observation_columns=None,
    ):
        super().__init__(
            states,
            columns,
            cfg,
            observation_columns=observation_columns,
        )
        self.minutes = np.asarray(minutes, dtype=int)
        if len(self.minutes) != len(self.z):
            raise ValueError("minutes and states must have the same length")
        self.protocol = protocol_phase(self.minutes)
        self.features = list(features)
        self.response = np.asarray(
            [float(response.get(column, 0.0)) for column in self.columns],
            dtype=np.float32,
        )
        protocol_cfg = cfg.get("protocol_surrogate", {})
        self.response_strength = float(protocol_cfg.get("response_strength", 1.0))
        self.response_decay = float(protocol_cfg.get("response_decay", 0.50))
        self.delta_weight = float(protocol_cfg.get("delta_weight", 1.0))
        self.mean_delta_weight = float(protocol_cfg.get("mean_delta_weight", 0.50))
        self.slope_weight = float(protocol_cfg.get("slope_weight", 0.35))
        self.uncertainty_weight = float(
            protocol_cfg.get("uncertainty_weight", 0.25)
        )
        self.offset = np.zeros(len(self.columns), dtype=np.float32)
        self.risk_scales = self._make_risk_scales()

    def _make_risk_scales(self):
        scales = {}
        for feature in self.features:
            for marker in (
                "_pred_end_delta", "_pred_mean_delta", "_pred_slope", "_endpoint_sd"
            ):
                column = next(
                    (name for name in self.columns if name.startswith(feature + marker)),
                    None,
                )
                if column is not None:
                    values = np.abs(self.base[:, self.columns.index(column)])
                    positive = values[values > 1e-9]
                    scale = float(np.median(positive)) if len(positive) else 1.0
                    scales[column] = max(scale, 1e-6)
        return scales

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.offset = np.zeros(len(self.columns), dtype=np.float32)
        self.last_full_state = self.z[0].copy()
        return self._observe(self.last_full_state), info

    def risk_components(self, normalized_state):
        raw = np.asarray(normalized_state, dtype=float) * self.std + self.mean
        components = {}
        for feature in self.features:
            values = {}
            for marker in (
                "_pred_end_delta", "_pred_mean_delta", "_pred_slope", "_endpoint_sd"
            ):
                column = next(
                    (name for name in self.columns if name.startswith(feature + marker)),
                    None,
                )
                if column is not None:
                    values[marker] = abs(float(raw[self.columns.index(column)])) / self.risk_scales[column]
            components[feature] = (
                self.delta_weight * values.get("_pred_end_delta", 0.0)
                + self.mean_delta_weight * values.get("_pred_mean_delta", 0.0)
                + self.slope_weight * values.get("_pred_slope", 0.0)
                + self.uncertainty_weight * values.get("_endpoint_sd", 0.0)
            )
        return components

    def _risk(self, state):
        components = self.risk_components(state)
        return float(np.mean(list(components.values()))) if components else 0.0

    def step(self, action):
        scheduled_level = int(self.protocol[self.t])
        if int(action) == 0:
            requested_level = max(0, scheduled_level - 1)
        elif int(action) == 2:
            requested_level = min(1, scheduled_level + 1)
        else:
            requested_level = scheduled_level
        level_change = requested_level - scheduled_level

        current = self.z[self.t] + self.offset
        projected = (
            current
            + float(level_change) * self.response_strength * self.response
        )
        self.last_full_state = projected.copy()
        hr = projected[self.columns.index("HR_current")]
        hr_reward = 1.0 if self.hr_low <= hr <= self.hr_high else -1.0
        reward = (
            hr_reward
            - float(self.cfg["rl_demo"]["reward"]["future_risk_weight"])
            * self._risk(projected)
            - float(self.cfg["rl_demo"]["reward"]["action_change_penalty"])
            * abs(int(action) - self.prev_action)
        )

        self.offset = self.response_decay * (
            self.offset
            + float(level_change) * self.response_strength * self.response
        )
        self.prev_action = int(action)
        self.t += 1
        done = self.t >= len(self.z)
        next_full = projected if done else self.z[self.t] + self.offset
        return self._observe(next_full), float(reward), done, False, {
            "scheduled_level": scheduled_level,
            "requested_level": requested_level,
            "level_change": level_change,
            "risk_components": self.risk_components(projected),
        }
