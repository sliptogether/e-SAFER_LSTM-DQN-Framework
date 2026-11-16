"""Exercise preprocessing, LSTM checkpoints, forecasts, and DQN together."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_synthetic_data", ROOT / "scripts" / "generate_synthetic_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_is_reproducible_and_protects_existing_files(tmp_path):
    generator = load_generator()
    first = generator.generate(tmp_path / "first", seed=7)
    second = generator.generate(tmp_path / "second", seed=7)
    for day in range(1, 5):
        a = pd.read_excel(first / f"Day{day}.xlsx", sheet_name=None)
        b = pd.read_excel(second / f"Day{day}.xlsx", sheet_name=None)
        assert a.keys() == b.keys()
        for sheet in a:
            pd.testing.assert_frame_equal(a[sheet], b[sheet])
    before = (first / "Day1.xlsx").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generator.generate(first, seed=8)
    assert (first / "Day1.xlsx").read_bytes() == before


def test_full_synthetic_pipeline(tmp_path):
    raw = load_generator().generate(tmp_path / "synthetic", seed=11)
    with (ROOT / "configs" / "synthetic_smoke.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    output = tmp_path / "pipeline_output"
    cfg["paths"] = {"raw_dir": str(raw), "output_dir": str(output)}
    cfg["model"]["epochs"] = 2
    cfg["model"]["ensemble_seeds"] = [101, 202]
    cfg["model"]["early_stopping"].update(min_epochs=1, patience=1)
    cfg["rl_demo"]["episodes"] = 3
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--config", str(config_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pipeline completed successfully" in result.stdout
    features = ["Na", "K", "Ca", "TEMP", "FR"]
    for name in ("metrics.json", "metrics_against_raw.json", "rolling_metrics.json"):
        metrics = json.loads((output / "evaluation" / name).read_text(encoding="utf-8"))
        assert set(metrics) == set(features)
        assert all(np.isfinite(value["rmse"]) for value in metrics.values())
    predictions = pd.read_csv(output / "evaluation" / "day4_trajectory.csv")
    assert len(predictions) == 5 * 5 * 5
    assert predictions.Predicted.notna().all()
    assert predictions.Predicted_minute.max() == 55
    for target in features:
        for seed in (101, 202):
            checkpoint = torch.load(
                output / "models" / target / f"seed_{seed}.pt", weights_only=False,
            )
            assert checkpoint["selected_epoch"] in (1, 2)
            assert checkpoint["epoch_selection"]["validation_day"] == 3
    state = pd.read_csv(output / "rl" / "rl_state.csv")
    assert state.shape == (46, 32)  # minute + 31 internal reward-state values
    checkpoint = torch.load(output / "rl" / "dqn_demo.pt", weights_only=False)
    assert len(checkpoint["columns"]) == 30
    assert "HR_current" not in checkpoint["columns"]
    assert len(checkpoint["reward_state_columns"]) == 31
    assert checkpoint["state_dict"]["net.0.weight"].shape == (64, 30)
    decisions = pd.read_csv(output / "rl" / "decision_timeline.csv")
    assert len(decisions) == 46
    assert set(decisions.Action_id).issubset({0, 1, 2})
    assert decisions.Status.eq("simulation_only").all()
    scores = decisions[["Score_reduce", "Score_maintain", "Score_increase"]].to_numpy()
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)
