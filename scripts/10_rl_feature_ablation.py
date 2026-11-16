"""Run feature-group ablation for the surrogate DQN observation state.

The reward is always computed from the complete 31-column environment state,
including reward-only HR. Each DQN observation contains only the 30 sweat and
environment features (or an ablated subset), avoiding an invalid comparison
in which reward terms disappear with dropped observation columns.
"""

import argparse
import copy
import json
import shutil

import numpy as np
import pandas as pd
import torch

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.rl.ablation import (
    FEATURE_TYPE_FIGURE_VARIANTS,
    SENSOR_FIGURE_VARIANTS,
    figure_subset,
    make_feature_sets,
    variant_family,
)
from sweat_rl.rl.dqn import train_dqn
from sweat_rl.rl.env import SurrogateExerciseEnv
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.robustness_figures import feature_ablation_figure


def _evaluate_greedy(network, env, minutes, seed):
    state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    actions = []
    rows = []
    step = 0
    network.eval()
    while not done:
        with torch.no_grad():
            q_values = network(torch.tensor(state[None], dtype=torch.float32))[0]
            scores = torch.softmax(q_values, dim=0)
            action = int(torch.argmax(q_values).item())
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        full_state = env.last_full_state
        zone = env.hr_zone(full_state)
        risk = env.risk(full_state)
        actions.append(action)
        total_reward += float(reward)
        rows.append(
            {
                "Minute": int(minutes[step]),
                "Action_id": action,
                "Action": ["Reduce", "Maintain", "Increase"][action],
                "Step_reward": float(reward),
                "Surrogate_risk": float(risk),
                "HR_zone": zone,
                "Policy_score": float(scores[action].item()),
                "Q_reduce": float(q_values[0].item()),
                "Q_maintain": float(q_values[1].item()),
                "Q_increase": float(q_values[2].item()),
            }
        )
        step += 1

    actions = np.asarray(actions, dtype=int)
    result = {
        "Evaluation_reward": float(total_reward),
        "Mean_risk": float(np.mean([row["Surrogate_risk"] for row in rows])),
        "HR_in_range_fraction": float(
            np.mean([row["HR_zone"] == "within_demo_range" for row in rows])
        ),
        "Action_switches": int(np.sum(actions[1:] != actions[:-1])),
        "Reduce_count": int(np.sum(actions == 0)),
        "Maintain_count": int(np.sum(actions == 1)),
        "Increase_count": int(np.sum(actions == 2)),
    }
    return result, rows


def _summary_table(runs):
    summary = (
        runs.groupby(
            ["Variant_order", "Variant", "Ablation_family", "State_dim"],
            as_index=False,
        )
        .agg(
            N_seeds=("Seed", "nunique"),
            Mean_evaluation_reward=("Evaluation_reward", "mean"),
            SD_evaluation_reward=("Evaluation_reward", "std"),
            Mean_final50_training_reward=("Final50_training_reward", "mean"),
            SD_final50_training_reward=("Final50_training_reward", "std"),
            Mean_risk=("Mean_risk", "mean"),
            Mean_HR_in_range_fraction=("HR_in_range_fraction", "mean"),
            Mean_action_switches=("Action_switches", "mean"),
        )
        .sort_values("Variant_order")
    )
    full_mean = float(
        summary.loc[
            summary.Variant == "Full", "Mean_evaluation_reward"
        ].iloc[0]
    )
    summary["Mean_reward_delta_vs_full"] = (
        summary.Mean_evaluation_reward - full_mean
    )
    return summary


def main(config):
    cfg = load_config(config)
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    rl_dir = output_root / "rl"
    ablation_dir = ensure_dir(rl_dir / "ablation")
    figure_dir = ensure_dir(output_root / "prediction_decision_figures")
    source_dir = ensure_dir(figure_dir / "source_data")
    origin_dir = ensure_dir(source_dir / "origin")

    state_df = pd.read_csv(rl_dir / "rl_state.csv")
    full_columns = [column for column in state_df.columns if column != "Origin_min"]
    full_values = state_df[full_columns].to_numpy(float)
    minutes = state_df["Origin_min"].to_numpy(int)
    variants = make_feature_sets(full_columns)
    seeds = list(
        cfg.get("rl_demo", {}).get(
            "paper_seeds",
            cfg.get("model", {}).get("ensemble_seeds", [101, 202, 303, 404, 505]),
        )
    )

    run_rows = []
    episode_rows = []
    decision_rows = []
    for variant_order, (variant, observation_columns) in enumerate(variants):
        print(
            f"[{variant_order + 1}/{len(variants)}] {variant}: "
            f"{len(observation_columns)} observed features",
            flush=True,
        )
        for seed in seeds:
            seed_cfg = copy.deepcopy(cfg)
            seed_cfg["project"]["seed"] = int(seed)
            train_env = SurrogateExerciseEnv(
                full_values,
                full_columns,
                seed_cfg,
                observation_columns=observation_columns,
            )
            network, rewards = train_dqn(train_env, seed_cfg)
            eval_env = SurrogateExerciseEnv(
                full_values,
                full_columns,
                seed_cfg,
                observation_columns=observation_columns,
            )
            evaluation, decisions = _evaluate_greedy(
                network, eval_env, minutes, int(seed)
            )
            run_rows.append(
                {
                    "Variant_order": variant_order,
                    "Variant": variant,
                    "Ablation_family": variant_family(variant),
                    "State_dim": len(observation_columns),
                    "Seed": int(seed),
                    "Final50_training_reward": float(np.mean(rewards[-50:])),
                    **evaluation,
                }
            )
            episode_rows.extend(
                {
                    "Variant_order": variant_order,
                    "Variant": variant,
                    "Ablation_family": variant_family(variant),
                    "State_dim": len(observation_columns),
                    "Seed": int(seed),
                    "Episode": episode,
                    "Reward": float(reward),
                }
                for episode, reward in enumerate(rewards, start=1)
            )
            decision_rows.extend(
                {
                    "Variant_order": variant_order,
                    "Variant": variant,
                    "Ablation_family": variant_family(variant),
                    "State_dim": len(observation_columns),
                    "Seed": int(seed),
                    **row,
                }
                for row in decisions
            )

    runs = pd.DataFrame(run_rows)
    full_by_seed = runs.loc[
        runs.Variant == "Full", ["Seed", "Evaluation_reward"]
    ].rename(columns={"Evaluation_reward": "Full_seed_reward"})
    runs = runs.merge(full_by_seed, on="Seed", how="left")
    runs["Reward_delta_vs_full_same_seed"] = (
        runs.Evaluation_reward - runs.Full_seed_reward
    )
    episodes = pd.DataFrame(episode_rows)
    decisions = pd.DataFrame(decision_rows)
    summary = _summary_table(runs)

    runs_path = ablation_dir / "feature_ablation_runs.csv"
    summary_path = ablation_dir / "feature_ablation_summary.csv"
    episodes_path = ablation_dir / "feature_ablation_episode_rewards.csv"
    decisions_path = ablation_dir / "feature_ablation_decisions.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    episodes.to_csv(episodes_path, index=False)
    decisions.to_csv(decisions_path, index=False)

    feature_set_rows = []
    feature_set_json = {}
    for variant_order, (variant, observation_columns) in enumerate(variants):
        feature_set_json[variant] = observation_columns
        feature_set_rows.extend(
            {
                "Variant_order": variant_order,
                "Variant": variant,
                "Ablation_family": variant_family(variant),
                "State_dim": len(observation_columns),
                "Feature": column,
            }
            for column in observation_columns
        )
    feature_sets = pd.DataFrame(feature_set_rows)
    feature_sets.to_csv(ablation_dir / "feature_sets.csv", index=False)
    with open(ablation_dir / "feature_sets.json", "w", encoding="utf-8") as handle:
        json.dump(feature_set_json, handle, ensure_ascii=False, indent=2)

    type_runs = figure_subset(runs, FEATURE_TYPE_FIGURE_VARIANTS)
    type_summary = figure_subset(summary, FEATURE_TYPE_FIGURE_VARIANTS)
    type_features = figure_subset(feature_sets, FEATURE_TYPE_FIGURE_VARIANTS)
    sensor_runs = figure_subset(runs, SENSOR_FIGURE_VARIANTS)
    sensor_summary = figure_subset(summary, SENSOR_FIGURE_VARIANTS)
    sensor_features = figure_subset(feature_sets, SENSOR_FIGURE_VARIANTS)

    out_base = figure_dir / "Fig19_rl_feature_ablation"
    feature_ablation_figure(type_runs, out_base)
    sensor_out_base = figure_dir / "Fig22_rl_sensor_ablation"
    feature_ablation_figure(
        sensor_runs,
        sensor_out_base,
        title="Sensor-wise ablation of the DQN observation state",
    )

    origin_runs = origin_dir / "Fig19_rl_feature_ablation_runs_origin.csv"
    origin_summary = origin_dir / "Fig19_rl_feature_ablation_summary_origin.csv"
    origin_features = origin_dir / "Fig19_rl_feature_sets_origin.csv"
    type_runs.to_csv(origin_runs, index=False, encoding="utf-8-sig")
    type_summary.to_csv(origin_summary, index=False, encoding="utf-8-sig")
    type_features.to_csv(origin_features, index=False, encoding="utf-8-sig")
    workbook = origin_dir / "Origin_plot_data_Fig19.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        type_runs.to_excel(writer, sheet_name="Fig19_runs", index=False)
        type_summary.to_excel(writer, sheet_name="Fig19_summary", index=False)
        type_features.to_excel(writer, sheet_name="Feature_sets", index=False)

    sensor_runs.to_csv(
        origin_dir / "Fig22_rl_sensor_ablation_runs_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_summary.to_csv(
        origin_dir / "Fig22_rl_sensor_ablation_summary_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_features.to_csv(
        origin_dir / "Fig22_rl_sensor_feature_sets_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_workbook = origin_dir / "Origin_plot_data_Fig22.xlsx"
    with pd.ExcelWriter(sensor_workbook, engine="openpyxl") as writer:
        sensor_runs.to_excel(writer, sheet_name="Fig22_runs", index=False)
        sensor_summary.to_excel(writer, sheet_name="Fig22_summary", index=False)
        sensor_features.to_excel(writer, sheet_name="Feature_sets", index=False)

    for source in (runs_path, summary_path, episodes_path, decisions_path):
        shutil.copy2(source, source_dir / source.name)

    print(summary.to_string(index=False), flush=True)
    print(out_base.with_suffix(".png"), flush=True)
    print(sensor_out_base.with_suffix(".png"), flush=True)
    print(workbook, flush=True)
    print(sensor_workbook, flush=True)
    print(
        "WARNING: ablation results are simulation-only surrogate comparisons; "
        "the reward remains fixed across observation variants.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
