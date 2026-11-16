"""Generate multi-seed RL and raw-versus-denoised robustness figures."""

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.rl.dqn import train_dqn
from sweat_rl.rl.env import SurrogateExerciseEnv
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.robustness_figures import (
    multiseed_convergence_figure,
    policy_benchmark_figure,
    raw_denoised_performance_figure,
    raw_denoised_performance_table,
)


def _rollout(env, choose_action, reset_seed=None):
    state, _ = env.reset(seed=reset_seed)
    done = False
    total = 0.0
    while not done:
        action = int(choose_action(state))
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    return total


def _greedy_action(network, state):
    network.eval()
    with torch.no_grad():
        values = network(torch.tensor(state[None], dtype=torch.float32))
    return int(torch.argmax(values, dim=1).item())


def _rule_action(env, state):
    state_index = min(env.t, len(env.z) - 1)
    hr = float(env.z[state_index, env.columns.index("HR_current")])
    if hr > env.hr_high:
        return 0
    if hr < env.hr_low:
        return 2
    return 1


def _make_env(state_values, columns, cfg):
    policy_columns = [column for column in columns if column != "HR_current"]
    return SurrogateExerciseEnv(
        state_values,
        columns,
        cfg,
        observation_columns=policy_columns,
    )


def _export_origin_tables(reward_df, benchmark_df, metric_df, origin_dir):
    """Write plot-ready CSV/XLSX tables with simple Origin-friendly columns."""
    origin_dir = ensure_dir(origin_dir)

    reward_wide = reward_df.pivot(index="Episode", columns="Seed", values="Reward").sort_index()
    reward_wide.columns = [f"Reward_seed_{int(seed)}" for seed in reward_wide.columns]
    rolling_wide = reward_wide.rolling(20, min_periods=20).mean()
    rolling_wide.columns = [column.replace("Reward_", "Rolling20_") for column in rolling_wide.columns]
    convergence_origin = pd.concat([reward_wide, rolling_wide], axis=1)
    convergence_origin["Rolling20_mean"] = rolling_wide.mean(axis=1)
    convergence_origin["Rolling20_SD"] = rolling_wide.std(axis=1, ddof=1)
    convergence_origin = convergence_origin.reset_index()

    benchmark_origin = benchmark_df[["Method", "Run", "Seed", "Reward"]].copy()
    benchmark_summary = (
        benchmark_origin.groupby("Method", as_index=False)
        .agg(
            N=("Reward", "count"),
            Mean_reward=("Reward", "mean"),
            SD_reward=("Reward", "std"),
            Min_reward=("Reward", "min"),
            Max_reward=("Reward", "max"),
        )
    )
    metric_wide = metric_df.pivot(index="Target", columns="Reference", values=["r2", "mae", "rmse"])
    metric_wide.columns = [
        f"{metric}_{reference.replace(' ', '_')}" for metric, reference in metric_wide.columns
    ]
    metric_wide = metric_wide.reset_index()
    metric_wide["RMSE_denoised_over_raw"] = (
        metric_wide["rmse_Denoised_target"] / metric_wide["rmse_Raw_target"]
    )

    tables = {
        "Fig16_dqn_multiseed_convergence_origin.csv": convergence_origin,
        "Fig17_policy_benchmark_origin.csv": benchmark_origin,
        "Fig17_policy_benchmark_summary_origin.csv": benchmark_summary,
        "Fig18_raw_vs_denoised_origin.csv": metric_wide,
    }
    for name, table in tables.items():
        table.to_csv(origin_dir / name, index=False)

    workbook = origin_dir / "Origin_plot_data_Fig16_Fig18.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        convergence_origin.to_excel(writer, sheet_name="Fig16_convergence", index=False)
        benchmark_origin.to_excel(writer, sheet_name="Fig17_runs", index=False)
        benchmark_summary.to_excel(writer, sheet_name="Fig17_summary", index=False)
        metric_wide.to_excel(writer, sheet_name="Fig18_raw_denoised", index=False)

    return tables, workbook


def main(config):
    cfg = load_config(config)
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    rl_dir = output_root / "rl"
    figure_dir = ensure_dir(output_root / "prediction_decision_figures")
    source_dir = ensure_dir(figure_dir / "source_data")
    origin_dir = ensure_dir(source_dir / "origin")
    robustness_dir = ensure_dir(rl_dir / "robustness")

    state_csv = rl_dir / "rl_state.csv"
    if not state_csv.exists():
        raise FileNotFoundError(f"Missing RL state: {state_csv}")
    state_df = pd.read_csv(state_csv)
    columns = [column for column in state_df.columns if column != "Origin_min"]
    state_values = state_df[columns].to_numpy(float)

    seeds = list(
        cfg.get("rl_demo", {}).get(
            "paper_seeds",
            cfg.get("model", {}).get("ensemble_seeds", [101, 202, 303, 404, 505]),
        )
    )
    reward_rows = []
    benchmark_rows = []
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["project"]["seed"] = int(seed)
        train_env = _make_env(state_values, columns, seed_cfg)
        network, rewards = train_dqn(train_env, seed_cfg)
        reward_rows.extend(
            {
                "Seed": int(seed),
                "Episode": episode,
                "Reward": float(reward),
            }
            for episode, reward in enumerate(rewards, start=1)
        )
        eval_env = _make_env(state_values, columns, seed_cfg)
        benchmark_rows.append(
            {
                "Method": "DQN",
                "Run": f"seed_{seed}",
                "Seed": int(seed),
                "Reward": _rollout(
                    eval_env,
                    lambda state, model=network: _greedy_action(model, state),
                    reset_seed=int(seed),
                ),
            }
        )

    reference_cfg = copy.deepcopy(cfg)
    reference_env = _make_env(state_values, columns, reference_cfg)
    benchmark_rows.append(
        {
            "Method": "Maintain",
            "Run": "deterministic",
            "Seed": np.nan,
            "Reward": _rollout(reference_env, lambda state: 1),
        }
    )
    rule_env = _make_env(state_values, columns, reference_cfg)
    benchmark_rows.append(
        {
            "Method": "Rule-based",
            "Run": "deterministic",
            "Seed": np.nan,
            "Reward": _rollout(rule_env, lambda state: _rule_action(rule_env, state)),
        }
    )
    random_runs = int(cfg.get("rl_demo", {}).get("random_baseline_runs", 50))
    for run in range(random_runs):
        random_seed = 10000 + run
        rng = np.random.default_rng(random_seed)
        random_env = _make_env(state_values, columns, reference_cfg)
        benchmark_rows.append(
            {
                "Method": "Random",
                "Run": f"random_{run + 1}",
                "Seed": random_seed,
                "Reward": _rollout(
                    random_env,
                    lambda state, generator=rng: int(generator.integers(0, 3)),
                    reset_seed=random_seed,
                ),
            }
        )

    reward_df = pd.DataFrame(reward_rows)
    reward_csv = robustness_dir / "dqn_multiseed_rewards.csv"
    reward_df.to_csv(reward_csv, index=False)
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_csv = robustness_dir / "policy_benchmark.csv"
    benchmark_df.to_csv(benchmark_csv, index=False)

    denoised_json = output_root / "evaluation" / "metrics.json"
    raw_json = output_root / "evaluation" / "metrics_against_raw.json"
    targets = tuple(cfg.get("figures", {}).get("prediction_targets", ["Na", "K", "Ca", "FR"]))
    metric_df = raw_denoised_performance_table(denoised_json, raw_json, targets=targets)
    metric_csv = robustness_dir / "raw_vs_denoised_metrics.csv"
    metric_df.to_csv(metric_csv, index=False)

    convergence_summary = multiseed_convergence_figure(
        reward_csv,
        figure_dir / "Fig16_dqn_multiseed_convergence",
    )
    convergence_summary.to_csv(
        robustness_dir / "dqn_multiseed_convergence_summary.csv",
        index=False,
    )
    policy_benchmark_figure(
        benchmark_csv,
        figure_dir / "Fig17_policy_benchmark",
    )
    raw_denoised_performance_figure(
        metric_df,
        figure_dir / "Fig18_raw_vs_denoised_performance",
        targets=targets,
    )

    origin_tables, origin_workbook = _export_origin_tables(
        reward_df,
        benchmark_df,
        metric_df,
        origin_dir,
    )

    for source in (
        reward_csv,
        benchmark_csv,
        metric_csv,
        robustness_dir / "dqn_multiseed_convergence_summary.csv",
    ):
        shutil.copy2(source, source_dir / source.name)

    figure_bases = sorted({path.stem for path in figure_dir.glob("*.png")})
    pd.DataFrame(
        {
            "Figure": figure_bases,
            "PNG": [f"{name}.png" for name in figure_bases],
            "PDF": [f"{name}.pdf" for name in figure_bases],
            "SVG": [f"{name}.svg" for name in figure_bases],
            "TIFF": [f"{name}.tiff" for name in figure_bases],
        }
    ).to_csv(figure_dir / "figure_manifest.csv", index=False)

    print(figure_dir)
    print(f"Generated robustness figures using {len(seeds)} DQN seeds.")
    print(f"Origin-ready data: {origin_workbook}")
    print(f"Origin-ready CSV files: {len(origin_tables)}")
    print("WARNING: all RL comparisons are simulation-only surrogate results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
