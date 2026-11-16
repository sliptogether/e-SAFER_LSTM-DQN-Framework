"""Protect interpretation and portability of the public research workflow."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sweat_rl.config import load_config
from sweat_rl.rl.env import SurrogateExerciseEnv


ROOT = Path(__file__).resolve().parents[1]


def test_standard_surrogate_replays_record_and_keeps_action_independent_risk():
    cfg = load_config(ROOT / "configs/synthetic_smoke.yaml")
    columns = ["HR_current", "TEMP_current", "FR_current",
               "Na_pred_end_delta5", "Na_endpoint_sd"]
    states = np.array([[80, 31, 1, .1, .01], [100, 32, 2, .3, .02],
                       [90, 33, 3, .2, .03]], dtype=float)
    env = SurrogateExerciseEnv(states, columns, cfg)
    env.reset()
    next_reduce, *_ = env.step(0)
    reduce_risk = env.risk(env.last_full_state)
    env.reset()
    next_increase, *_ = env.step(2)
    increase_risk = env.risk(env.last_full_state)
    np.testing.assert_allclose(next_reduce, next_increase)
    np.testing.assert_allclose(next_reduce, env.z[1])
    assert reduce_risk == pytest.approx(increase_risk)


@pytest.mark.parametrize("horizon", [3, 10])
def test_dqn_export_uses_configured_forecast_horizon(tmp_path, horizon):
    """Retained horizon variants must not receive mislabeled five-minute CSVs."""
    cfg = yaml.safe_load((ROOT / "configs/synthetic_smoke.yaml").read_text())
    cfg["data"]["forecast_horizon_min"] = horizon
    cfg["paths"]["output_dir"] = str(tmp_path / "outputs")
    cfg["rl_demo"]["episodes"] = 1
    cfg["rl_demo"]["batch_size"] = 32
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rl_dir = tmp_path / "outputs/rl"
    rl_dir.mkdir(parents=True)
    state = {"Origin_min": [44, 50, 53], "HR_current": [90, 100, 110]}
    for j, feature in enumerate(cfg["data"]["features"]):
        state[f"{feature}_current"] = np.array([1., 2., 3.]) + j
        state[f"{feature}_pred_end_delta{horizon}"] = [.1, .2, .3]
        state[f"{feature}_endpoint_sd"] = [.01, .02, .03]
    pd.DataFrame(state).to_csv(rl_dir / "rl_state.csv", index=False)
    spec = importlib.util.spec_from_file_location(
        "dqn_export_stage", ROOT / "scripts/05_train_dqn_demo.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec.loader.exec_module(module)
    module.main(cfg_path)
    result = pd.read_csv(rl_dir / "decision_timeline.csv")
    column = f"Full_{horizon}min_outcome_observed"
    assert result[column].tolist() == [m + horizon <= 55 for m in [44, 50, 53]]
    assert f"Na_predicted_{horizon}min_change" in result.columns
    assert "Na_predicted_5min_change" not in result.columns
