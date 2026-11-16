"""Run the portable Python forecasting and surrogate-DQN pipeline."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import _bootstrap
from sweat_rl.config import load_config, project_path


STAGES = (
    "01_preprocess.py",
    "02_train_lstm.py",
    "03_evaluate_day4.py",
    "04_build_rl_state.py",
    "05_train_dqn_demo.py",
    "06_make_figures.py",
)


def required_inputs(cfg, start_at):
    """List inputs which must already exist when resuming at a stage."""
    output = project_path(cfg, cfg["paths"]["output_dir"])
    processed = [output / "processed" / f"Day{day}.csv" for day in range(1, 5)]
    if start_at == 1:
        raw = project_path(cfg, cfg["paths"]["raw_dir"])
        return [raw / f"Day{day}.xlsx" for day in range(1, 5)]
    if start_at == 2:
        return processed
    if start_at == 3:
        return processed + [
            output / "models" / target / f"seed_{seed}.pt"
            for target in cfg["data"]["features"]
            for seed in cfg["model"]["ensemble_seeds"]
        ]
    if start_at == 4:
        rolling = cfg["rl_state"].get("use_rolling_predictions", False)
        predictions = (
            "rolling_ensemble_predictions.json" if rolling
            else "ensemble_predictions.json"
        )
        return processed + [output / "evaluation" / predictions]
    if start_at == 5:
        return [output / "rl" / "rl_state.csv"]
    return [
        output / "evaluation" / "day4_trajectory.csv",
        output / "evaluation" / "metrics.json",
        output / "rl" / "rl_state.csv",
    ]


def main(config="configs/denoised_5min.yaml", figures=False, start_at=1):
    root = Path(__file__).resolve().parents[1]
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if start_at == 6 and not figures:
        raise ValueError("--start-at 6 requires --figures")

    cfg = load_config(config_path)
    missing = [str(path) for path in required_inputs(cfg, start_at) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline inputs:\n  " + "\n  ".join(missing)
            + "\nFor the included example, run with --config configs/synthetic_smoke.yaml."
        )

    final_stage = 6 if figures else 5
    print("e-SAFER: LSTM forecasting + surrogate DQN demonstration", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Output: {project_path(cfg, cfg['paths']['output_dir'])}", flush=True)
    print("DQN outputs are simulation-only recommendations.", flush=True)
    env = os.environ.copy()
    # Small CPU networks are faster with one BLAS/OpenMP thread. Respect any
    # explicit user setting, and use the same environment for every stage.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUTF8", "1")
    for number in range(start_at, final_stage + 1):
        script = root / "scripts" / STAGES[number - 1]
        print(f"\n[{number}/{final_stage}] {script.name}", flush=True)
        subprocess.run(
            [sys.executable, str(script), "--config", str(config_path)],
            cwd=root, env=env, check=True,
        )
    print("\nPipeline completed successfully.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    parser.add_argument("--figures", action="store_true", help="Also generate basic figures (stage 6).")
    parser.add_argument("--start-at", type=int, choices=range(1, 7), default=1, help="Resume at an existing stage (1–6).")
    args = parser.parse_args()
    try:
        main(args.config, args.figures, args.start_at)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
