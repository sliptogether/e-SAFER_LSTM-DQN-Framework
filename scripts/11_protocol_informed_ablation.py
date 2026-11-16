"""Run Day 4 protocol-informed DQN ablation without modifying measurements."""

import argparse
import copy
import json

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
from sweat_rl.rl.protocol_env import (
    ProtocolInformedSurrogateEnv,
    estimate_protocol_response,
    protocol_phase,
)
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.robustness_figures import feature_ablation_figure


def _evaluate(network, env, minutes, seed):
    state, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    actions = []
    rows = []
    index = 0
    network.eval()
    while not done:
        with torch.no_grad():
            q_values = network(torch.tensor(state[None], dtype=torch.float32))[0]
            scores = torch.softmax(q_values, dim=0)
            action = int(torch.argmax(q_values).item())
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        actions.append(action)
        total += float(reward)
        row = {
            "Minute": int(minutes[index]),
            "Protocol_phase": "Exercise" if env.protocol[index] else "Rest",
            "Action_id": action,
            "Action": ["Reduce", "Maintain", "Increase"][action],
            "Step_reward": float(reward),
            "Total_risk": float(sum(info["risk_components"].values()) / len(info["risk_components"])),
            "HR_zone": env.hr_zone(env.last_full_state),
            "Policy_score": float(scores[action].item()),
            "Requested_level": int(info["requested_level"]),
            "Scheduled_level": int(info["scheduled_level"]),
        }
        for feature, value in info["risk_components"].items():
            row[f"Risk_{feature}"] = float(value)
        rows.append(row)
        index += 1

    actions = np.asarray(actions, dtype=int)
    result = {
        "Evaluation_reward": float(total),
        "Mean_risk": float(np.mean([row["Total_risk"] for row in rows])),
        "HR_in_range_fraction": float(np.mean([row["HR_zone"] == "within_demo_range" for row in rows])),
        "Action_switches": int(np.sum(actions[1:] != actions[:-1])),
        "Reduce_count": int(np.sum(actions == 0)),
        "Maintain_count": int(np.sum(actions == 1)),
        "Increase_count": int(np.sum(actions == 2)),
    }
    for feature in env.features:
        result[f"Mean_risk_{feature}"] = float(
            np.mean([row[f"Risk_{feature}"] for row in rows])
        )
    return result, rows


def main(config):
    cfg = load_config(config)
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    rl_dir = output_root / "rl"
    result_dir = ensure_dir(rl_dir / "protocol_informed_ablation")
    figure_dir = ensure_dir(output_root / "prediction_decision_figures")
    origin_dir = ensure_dir(figure_dir / "source_data" / "origin")

    state_df = pd.read_csv(rl_dir / "rl_state.csv")
    processed = pd.read_csv(output_root / "processed" / "Day4.csv")
    full_columns = [column for column in state_df.columns if column != "Origin_min"]
    full_values = state_df[full_columns].to_numpy(float)
    minutes = state_df.Origin_min.to_numpy(int)
    features = list(cfg["data"]["features"])
    horizon = int(cfg["data"]["forecast_horizon_min"])
    response, details = estimate_protocol_response(
        processed, state_df, features, horizon=horizon
    )
    variants = make_feature_sets(full_columns)
    seeds = list(cfg.get("model", {}).get("ensemble_seeds", [101, 202, 303, 404, 505]))

    protocol_table = pd.DataFrame(
        {
            "Minute": processed.Minute.astype(int),
            "Protocol_level": protocol_phase(processed.Minute.to_numpy(int)),
        }
    )
    protocol_table["Protocol_phase"] = protocol_table.Protocol_level.map(
        {0: "Rest", 1: "Exercise"}
    )
    protocol_table.to_csv(result_dir / "day4_protocol_labels.csv", index=False)
    pd.DataFrame(
        [{"State_feature": key, "Normalized_response": value} for key, value in response.items()]
    ).to_csv(result_dir / "protocol_response_coefficients.csv", index=False)
    with open(result_dir / "protocol_response_details.json", "w", encoding="utf-8") as handle:
        json.dump(details, handle, ensure_ascii=False, indent=2)

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
            kwargs = dict(
                states=full_values,
                columns=full_columns,
                cfg=seed_cfg,
                minutes=minutes,
                response=response,
                features=features,
                observation_columns=observation_columns,
            )
            train_env = ProtocolInformedSurrogateEnv(**kwargs)
            network, rewards = train_dqn(train_env, seed_cfg)
            evaluation, decisions = _evaluate(
                network, ProtocolInformedSurrogateEnv(**kwargs), minutes, int(seed)
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
    full_seed = runs[runs.Variant.eq("Full")][["Seed", "Evaluation_reward"]].rename(
        columns={"Evaluation_reward": "Full_seed_reward"}
    )
    runs = runs.merge(full_seed, on="Seed", how="left")
    runs["Reward_delta_vs_full_same_seed"] = runs.Evaluation_reward - runs.Full_seed_reward
    summary = (
        runs.groupby(
            ["Variant_order", "Variant", "Ablation_family", "State_dim"],
            as_index=False,
        )
        .agg(
            N_seeds=("Seed", "nunique"),
            Mean_evaluation_reward=("Evaluation_reward", "mean"),
            SD_evaluation_reward=("Evaluation_reward", "std"),
            Mean_delta_vs_full=("Reward_delta_vs_full_same_seed", "mean"),
            SD_delta_vs_full=("Reward_delta_vs_full_same_seed", "std"),
            Mean_final50_training_reward=("Final50_training_reward", "mean"),
            Mean_risk=("Mean_risk", "mean"),
            Mean_HR_in_range_fraction=("HR_in_range_fraction", "mean"),
            Mean_action_switches=("Action_switches", "mean"),
            Mean_risk_Na=("Mean_risk_Na", "mean"),
            Mean_risk_K=("Mean_risk_K", "mean"),
            Mean_risk_Ca=("Mean_risk_Ca", "mean"),
            Mean_risk_TEMP=("Mean_risk_TEMP", "mean"),
            Mean_risk_FR=("Mean_risk_FR", "mean"),
        )
        .sort_values("Variant_order")
    )
    episodes = pd.DataFrame(episode_rows)
    decisions = pd.DataFrame(decision_rows)
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
    runs.to_csv(result_dir / "protocol_ablation_runs.csv", index=False)
    summary.to_csv(result_dir / "protocol_ablation_summary.csv", index=False)
    episodes.to_csv(result_dir / "protocol_ablation_episode_rewards.csv", index=False)
    decisions.to_csv(result_dir / "protocol_ablation_decisions.csv", index=False)
    feature_sets.to_csv(result_dir / "protocol_feature_sets.csv", index=False)
    with open(result_dir / "protocol_feature_sets.json", "w", encoding="utf-8") as handle:
        json.dump(feature_set_json, handle, ensure_ascii=False, indent=2)

    type_runs = figure_subset(runs, FEATURE_TYPE_FIGURE_VARIANTS)
    type_summary = figure_subset(summary, FEATURE_TYPE_FIGURE_VARIANTS)
    type_decisions = figure_subset(decisions, FEATURE_TYPE_FIGURE_VARIANTS)
    type_features = figure_subset(feature_sets, FEATURE_TYPE_FIGURE_VARIANTS)
    sensor_runs = figure_subset(runs, SENSOR_FIGURE_VARIANTS)
    sensor_summary = figure_subset(summary, SENSOR_FIGURE_VARIANTS)
    sensor_decisions = figure_subset(decisions, SENSOR_FIGURE_VARIANTS)
    sensor_features = figure_subset(feature_sets, SENSOR_FIGURE_VARIANTS)
    out_base = figure_dir / "Fig21_protocol_informed_feature_ablation"
    feature_ablation_figure(
        type_runs,
        out_base,
        title="Protocol-informed ablation of the surrogate DQN state",
    )
    sensor_out_base = figure_dir / "Fig23_protocol_informed_sensor_ablation"
    feature_ablation_figure(
        sensor_runs,
        sensor_out_base,
        title="Protocol-informed sensor-wise ablation",
    )
    workbook = origin_dir / "Origin_plot_data_Fig21.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        type_runs.to_excel(writer, sheet_name="Fig21_runs", index=False)
        type_summary.to_excel(writer, sheet_name="Fig21_summary", index=False)
        type_decisions.to_excel(writer, sheet_name="Decision_traces", index=False)
        type_features.to_excel(writer, sheet_name="Feature_sets", index=False)
        protocol_table.to_excel(writer, sheet_name="Day4_protocol", index=False)
        pd.DataFrame(
            [{"State_feature": key, "Normalized_response": value} for key, value in response.items()]
        ).to_excel(writer, sheet_name="Response_coefficients", index=False)
    type_runs.to_csv(origin_dir / "Fig21_protocol_ablation_runs_origin.csv", index=False, encoding="utf-8-sig")
    type_summary.to_csv(origin_dir / "Fig21_protocol_ablation_summary_origin.csv", index=False, encoding="utf-8-sig")

    sensor_runs.to_csv(
        origin_dir / "Fig23_protocol_sensor_ablation_runs_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_summary.to_csv(
        origin_dir / "Fig23_protocol_sensor_ablation_summary_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_features.to_csv(
        origin_dir / "Fig23_protocol_sensor_feature_sets_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensor_workbook = origin_dir / "Origin_plot_data_Fig23.xlsx"
    with pd.ExcelWriter(sensor_workbook, engine="openpyxl") as writer:
        sensor_runs.to_excel(writer, sheet_name="Fig23_runs", index=False)
        sensor_summary.to_excel(writer, sheet_name="Fig23_summary", index=False)
        sensor_decisions.to_excel(writer, sheet_name="Decision_traces", index=False)
        sensor_features.to_excel(writer, sheet_name="Feature_sets", index=False)
        protocol_table.to_excel(writer, sheet_name="Day4_protocol", index=False)
        pd.DataFrame(
            [{"State_feature": key, "Normalized_response": value} for key, value in response.items()]
        ).to_excel(writer, sheet_name="Response_coefficients", index=False)

    print(summary.to_string(index=False), flush=True)
    print(out_base.with_suffix(".png"), flush=True)
    print(sensor_out_base.with_suffix(".png"), flush=True)
    print(workbook, flush=True)
    print(sensor_workbook, flush=True)
    print(
        "WARNING: protocol-informed counterfactual surrogate only; not validated human closed-loop control.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
