"""Export Origin-ready source data for every generated project figure."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.rl.ablation import (
    FEATURE_TYPE_FIGURE_VARIANTS,
    SENSOR_FIGURE_VARIANTS,
    figure_subset,
)
from sweat_rl.utils import ensure_dir


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _add_bounds(df, mean_col, sd_col, prefix="Predicted"):
    out = df.copy()
    out[f"{prefix}_lower_1SD"] = out[mean_col] - out[sd_col]
    out[f"{prefix}_upper_1SD"] = out[mean_col] + out[sd_col]
    return out


def _schematic_tables():
    nodes = pd.DataFrame(
        [
            (1, "sensing", "Wearable multimodal sensing", "Na+, K+, Ca2+, TEMP, FR, HR", 0.128, 0.723, 0.098, 0.490, 0.270, 0.760),
            (2, "preprocess", "Causal preprocessing", "1-min alignment; Holt denoising; raw retained", 0.338, 0.723, 0.258, 0.490, 0.730, 0.760),
            (3, "lstm", "LSTM ensemble", "Past 5 min to future 5 min; five seeds", 0.553, 0.723, 0.418, 0.490, 0.730, 0.500),
            (4, "forecast", "Rolling forecast", "Minute 10-55; prediction and uncertainty", 0.813, 0.723, 0.578, 0.490, 0.270, 0.500),
            (5, "state", "Prediction-derived RL state", "Current signals, predicted change, slope, uncertainty", 0.413, 0.253, 0.750, 0.490, 0.270, 0.180),
            (6, "dqn", "Surrogate DQN policy", "Q scores for Reduce, Maintain and Increase", 0.650, 0.253, 0.910, 0.490, 0.730, 0.180),
            (7, "actions", "Action recommendation", "Reduce; Maintain; Increase", 0.860, 0.253, np.nan, np.nan, np.nan, np.nan),
        ],
        columns=[
            "Sequence",
            "Node_ID",
            "Title",
            "Description",
            "Fig13_X",
            "Fig13_Y",
            "Fig14_X",
            "Fig14_Y",
            "Fig15_X",
            "Fig15_Y",
        ],
    )
    edges = pd.DataFrame(
        [
            ("sensing", "preprocess", "signal flow", "all"),
            ("preprocess", "lstm", "processed history", "all"),
            ("lstm", "forecast", "ensemble prediction", "all"),
            ("forecast", "state", "forecast features", "all"),
            ("state", "dqn", "state vector", "all"),
            ("dqn", "actions", "selected action", "Fig13"),
            ("actions", "state", "surrogate action effect", "Fig13 feedback"),
        ],
        columns=["From_Node", "To_Node", "Edge_Label", "Used_In"],
    )
    return nodes, edges


def main(config):
    cfg = load_config(config)
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    evaluation_dir = output_root / "evaluation"
    history_dir = output_root / "training_history"
    rl_dir = output_root / "rl"
    suite_source_dir = output_root / "prediction_decision_figures" / "source_data"
    export_dir = ensure_dir(output_root / "origin_data_all")
    csv_dir = ensure_dir(export_dir / "csv")

    targets = tuple(cfg.get("figures", {}).get("prediction_targets", ["Na", "K", "Ca", "FR"]))
    target_order = {target: index for index, target in enumerate(targets)}
    origin = int(cfg.get("figures", {}).get("prediction_origin", 20))

    tables = []
    guide_rows = []

    def add(csv_name, sheet_name, data, figures, origin_mapping, note):
        table = data.copy()
        table.to_csv(csv_dir / csv_name, index=False, encoding="utf-8-sig")
        tables.append((sheet_name, table))
        guide_rows.append(
            {
                "Figures": figures,
                "CSV_file": csv_name,
                "Excel_sheet": sheet_name,
                "Origin_column_mapping": origin_mapping,
                "Notes": note,
                "Rows": len(table),
                "Columns": len(table.columns),
            }
        )

    # LSTM epoch-selection loss: raw seed traces plus the exact mean/SD summary.
    history_files = sorted(history_dir.glob("*/seed_*_loss.csv"))
    if not history_files:
        raise FileNotFoundError(f"No training-history files found in {history_dir}")
    history_long = pd.concat(
        [pd.read_csv(path).assign(Source_file=path.name) for path in history_files],
        ignore_index=True,
    )
    history_long = history_long[history_long.phase == "epoch_selection"].copy()
    loss_summary = (
        history_long.groupby(["target", "epoch"], as_index=False)
        .agg(
            Train_total_mean=("train_total_loss", "mean"),
            Train_total_SD=("train_total_loss", "std"),
            Validation_total_mean=("validation_total_loss", "mean"),
            Validation_total_SD=("validation_total_loss", "std"),
            N_seeds=("seed", "nunique"),
            Median_selected_epoch=("selected_epoch", "median"),
        )
    )
    add(
        "Fig_LSTM_loss_curves_long.csv",
        "LSTM_loss_long",
        history_long,
        "Fig_LSTM_loss_curves",
        "X=epoch; Y=train_total_loss or validation_total_loss; group=target and seed",
        "All individual seed traces used by the loss figure.",
    )
    add(
        "Fig_LSTM_loss_curves_summary.csv",
        "LSTM_loss_summary",
        loss_summary,
        "Fig_LSTM_loss_curves",
        "X=epoch; Y=Train_total_mean/Validation_total_mean; Y Error=corresponding SD",
        "Across-seed summary; five seeds when N_seeds=5.",
    )

    metrics = _read_json(evaluation_dir / "metrics.json")
    heatmap = pd.DataFrame(
        {
            "Metric": ["R2", "MAE", "RMSE", "Direction_accuracy"],
            **{
                target: [
                    metrics[target]["r2"],
                    metrics[target]["mae"],
                    metrics[target]["rmse"],
                    metrics[target]["direction_accuracy"],
                ]
                for target in metrics
            },
        }
    )
    add(
        "Fig_performance_heatmap.csv",
        "Performance_heatmap",
        heatmap,
        "Fig_performance_heatmap",
        "X=target columns; Y=Metric; Z=cell values",
        "Fixed 5-min Day 4 metrics against denoised targets.",
    )

    trajectory = pd.read_csv(evaluation_dir / "day4_trajectory.csv")
    origin_prediction = trajectory[
        (trajectory.Origin == origin) & trajectory.Target.isin(targets)
    ].copy()
    origin_prediction = _add_bounds(origin_prediction, "Predicted", "Ensemble_SD")
    origin_prediction["Target_order"] = origin_prediction.Target.map(target_order)
    origin_prediction = origin_prediction.sort_values(["Target_order", "Horizon"]).drop(columns="Target_order")
    add(
        f"Fig_prediction_all_ions_origin{origin}.csv",
        "Origin20_forecast",
        origin_prediction,
        f"Fig_prediction_all_ions_origin{origin}",
        "X=Horizon or Predicted_minute; Y=Predicted and Actual; error band=lower/upper 1SD; group=Target",
        "Fixed-origin ensemble forecast for Na, K, Ca and FR.",
    )

    rl_state = pd.read_csv(rl_dir / "rl_state.csv")
    state_origin = 30 if 30 in set(rl_state.Origin_min) else int(rl_state.Origin_min.iloc[0])
    state_row = rl_state[rl_state.Origin_min == state_origin].iloc[0]
    state_rows = []
    for target in ("Na", "K", "Ca", "TEMP", "FR"):
        column = next(column for column in rl_state.columns if column.startswith(f"{target}_pred_end_delta"))
        state_rows.append(
            {
                "Origin_min": state_origin,
                "Target": target,
                "Predicted_endpoint_change": float(state_row[column]),
                "Source_column": column,
            }
        )
    add(
        "Fig_prediction_to_RL.csv",
        "RL_state_origin30",
        pd.DataFrame(state_rows),
        "Fig_prediction_to_RL",
        "X=Target; Y=Predicted_endpoint_change",
        "Prediction-derived RL-state bar chart at minute 30.",
    )

    rolling = pd.read_csv(evaluation_dir / "day4_rolling_consensus.csv")
    rolling = rolling[rolling.Target.isin(targets)].copy()
    rolling = _add_bounds(rolling, "Predicted", "Consensus_SD", prefix="Consensus")
    rolling["Residual"] = rolling.Predicted - rolling.Actual
    rolling["Absolute_error"] = rolling.Residual.abs()
    rolling["Target_order"] = rolling.Target.map(target_order)
    rolling = rolling.sort_values(["Target_order", "Minute"]).drop(columns="Target_order")
    add(
        "Fig_rolling_5min_all_ions_10to55.csv",
        "Rolling_all_ions",
        rolling,
        "Fig_rolling_5min_all_ions_10to55",
        "X=Minute; Y=Actual and Predicted; error band=Consensus_lower/upper_1SD; group=Target",
        "Consensus of all available 1-5 min forecasts for each target minute.",
    )
    for target in targets:
        add(
            f"Fig_rolling_5min_{target}_10to55.csv",
            f"Rolling_{target}",
            rolling[rolling.Target == target].copy(),
            f"Fig_rolling_5min_{target}_10to55",
            "X=Minute; Y=Actual and Predicted; error band=Consensus_lower/upper_1SD",
            f"Individual rolling forecast for {target}.",
        )

    parity = rolling[rolling.Predicted.notna()].copy()
    add(
        "Fig02_prediction_parity.csv",
        "Fig02_parity",
        parity[["Target", "Minute", "Actual", "Predicted", "Residual", "Absolute_error"]],
        "Fig02_prediction_parity",
        "X=Actual; Y=Predicted; group=Target",
        "Use a Y=X reference line in Origin.",
    )
    add(
        "Fig03_residuals_over_time.csv",
        "Fig03_residuals",
        parity[["Target", "Minute", "Residual", "Consensus_SD"]],
        "Fig03_residuals_over_time",
        "X=Minute; Y=Residual; auxiliary band=plus/minus Consensus_SD; group=Target",
        "Residual is Predicted minus Actual.",
    )

    lead_path = suite_source_dir / "lead_time_metrics.csv"
    lead_metrics = pd.read_csv(lead_path)
    add(
        "Fig04_lead_time_performance.csv",
        "Fig04_lead_time",
        lead_metrics,
        "Fig04_lead_time_performance",
        "X=Lead_min; Y=r2 or nrmse; group=Target",
        "Metrics from individual rolling forecasts at each lead time.",
    )
    horizon_metrics = pd.read_csv(suite_source_dir / "fixed_horizon_metrics.csv")
    add(
        "Fig05_fixed_horizon_comparison.csv",
        "Fig05_horizons",
        horizon_metrics,
        "Fig05_fixed_horizon_comparison",
        "X=Target; Y=r2; group=Horizon_min",
        "Independently trained 3-, 5- and 10-min configurations.",
    )
    add(
        "Fig06_uncertainty_vs_error.csv",
        "Fig06_uncertainty_error",
        parity[["Target", "Minute", "Consensus_SD", "Absolute_error"]],
        "Fig06_uncertainty_vs_error",
        "X=Consensus_SD; Y=Absolute_error; group=Target",
        "Consensus SD is model/forecast disagreement, not calibrated confidence.",
    )

    decisions = pd.read_csv(rl_dir / "decision_timeline.csv").sort_values("Minute")
    add(
        "Fig07_decision_timeline.csv",
        "Fig07_decisions",
        decisions[[
            "Minute", "Action_id", "Action_en", "Action_zh", "Policy_score",
            "Full_5min_outcome_observed", "Status"
        ]],
        "Fig07_decision_timeline",
        "X=Minute; Y=Policy_score; color/category=Action_en",
        "Action_id: 0 Reduce, 1 Maintain, 2 Increase.",
    )
    add(
        "Fig08_policy_action_scores.csv",
        "Fig08_action_scores",
        decisions[["Minute", "Score_reduce", "Score_maintain", "Score_increase"]],
        "Fig08_policy_action_scores",
        "X=Minute; Y=three Score columns",
        "Softmax scores are relative action scores, not calibrated probabilities.",
    )
    add(
        "Fig09_surrogate_risk.csv",
        "Fig09_risk",
        decisions[["Minute", "Surrogate_future_risk", "Action_id", "Action_en", "Policy_score"]],
        "Fig09_surrogate_risk",
        "X=Minute; Y=Surrogate_future_risk; symbol color=Action_en",
        "Risk belongs to the configured surrogate reward only.",
    )

    change_rows = []
    for target in ("Na", "K", "Ca", "TEMP", "FR"):
        column = f"{target}_predicted_5min_change"
        values = decisions[column].to_numpy(float)
        sd = float(values.std(ddof=0))
        standardized = (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)
        for minute, raw_value, z_value, action in zip(
            decisions.Minute, values, standardized, decisions.Action_en
        ):
            change_rows.append(
                {
                    "Minute": int(minute),
                    "Target": target,
                    "Predicted_5min_change": float(raw_value),
                    "Standardized_change_Z": float(z_value),
                    "Action_en": action,
                }
            )
    add(
        "Fig10_predicted_state_change_heatmap.csv",
        "Fig10_state_heatmap",
        pd.DataFrame(change_rows),
        "Fig10_predicted_state_change_heatmap",
        "X=Minute; Y=Target; Z=Standardized_change_Z; action strip=Action_en",
        "Z values are standardized within each target across decision minutes.",
    )

    rewards = pd.read_csv(rl_dir / "dqn_rewards.csv")
    rewards["Rolling20_mean"] = rewards.Reward.rolling(20, min_periods=20).mean()
    add(
        "Fig11_dqn_training_reward.csv",
        "Fig11_DQN_reward",
        rewards,
        "Fig11_dqn_training_reward and Fig_DQN_reward",
        "X=Episode; Y=Reward and Rolling20_mean",
        "Single-seed surrogate DQN training curve.",
    )
    action_counts = (
        decisions.Action_en.value_counts()
        .reindex(["Reduce", "Maintain", "Increase"], fill_value=0)
        .rename_axis("Action_en")
        .reset_index(name="Count")
    )
    add(
        "Fig12_decision_summary_counts.csv",
        "Fig12_action_counts",
        action_counts,
        "Fig12_decision_summary panel a",
        "X=Action_en; Y=Count",
        "Number of decision minutes assigned to each action.",
    )
    add(
        "Fig12_decision_summary_scores.csv",
        "Fig12_score_distribution",
        decisions[["Minute", "Action_en", "Policy_score"]],
        "Fig12_decision_summary panel b",
        "X=Action_en; Y=Policy_score; labels/dots=Minute",
        "Plot raw points plus the median for each action.",
    )

    nodes, edges = _schematic_tables()
    add(
        "Fig13_Fig15_schematic_nodes.csv",
        "Schematic_nodes",
        nodes,
        "Fig13, Fig14 and Fig15",
        "Use Node_ID, Title, Description and the X/Y columns for the chosen figure",
        "Structural source table; these schematics have no experimental numerical series.",
    )
    add(
        "Fig13_Fig15_schematic_edges.csv",
        "Schematic_edges",
        edges,
        "Fig13, Fig14 and Fig15",
        "Connect From_Node to To_Node; label with Edge_Label",
        "Structural source table for workflow arrows.",
    )

    robustness_dir = rl_dir / "robustness"
    multiseed_wide = pd.read_csv(
        suite_source_dir / "origin" / "Fig16_dqn_multiseed_convergence_origin.csv"
    )
    add(
        "Fig16_dqn_multiseed_convergence.csv",
        "Fig16_multiseed",
        multiseed_wide,
        "Fig16_dqn_multiseed_convergence",
        "X=Episode; Y=Rolling20_seed_* or Rolling20_mean; Y Error=Rolling20_SD",
        "Five DQN seeds; use rolling columns for the published convergence curve.",
    )
    policy_runs = pd.read_csv(robustness_dir / "policy_benchmark.csv")
    policy_summary = pd.read_csv(
        suite_source_dir / "origin" / "Fig17_policy_benchmark_summary_origin.csv"
    )
    add(
        "Fig17_policy_benchmark_runs.csv",
        "Fig17_policy_runs",
        policy_runs,
        "Fig17_policy_benchmark",
        "X=Method; Y=Reward; individual points=Run",
        "Random n=50, DQN n=5 seeds, deterministic baselines n=1.",
    )
    add(
        "Fig17_policy_benchmark_summary.csv",
        "Fig17_policy_summary",
        policy_summary,
        "Fig17_policy_benchmark",
        "X=Method; Y=Mean_reward; Y Error=SD_reward",
        "Do not infer human-control efficacy from surrogate rewards.",
    )
    raw_denoised = pd.read_csv(
        suite_source_dir / "origin" / "Fig18_raw_vs_denoised_origin.csv"
    )
    add(
        "Fig18_raw_vs_denoised_performance.csv",
        "Fig18_raw_denoised",
        raw_denoised,
        "Fig18_raw_vs_denoised_performance",
        "X=Target; Y=r2_Raw_target/r2_Denoised_target or RMSE ratio columns",
        "Same forecasts evaluated against two different target definitions.",
    )

    ablation_dir = rl_dir / "ablation"
    ablation_runs_path = ablation_dir / "feature_ablation_runs.csv"
    if ablation_runs_path.exists():
        ablation_runs = pd.read_csv(ablation_runs_path)
        ablation_summary = pd.read_csv(
            ablation_dir / "feature_ablation_summary.csv"
        )
        feature_sets = pd.read_csv(ablation_dir / "feature_sets.csv")
        type_runs = figure_subset(ablation_runs, FEATURE_TYPE_FIGURE_VARIANTS)
        type_summary = figure_subset(ablation_summary, FEATURE_TYPE_FIGURE_VARIANTS)
        type_features = figure_subset(feature_sets, FEATURE_TYPE_FIGURE_VARIANTS)
        sensor_runs = figure_subset(ablation_runs, SENSOR_FIGURE_VARIANTS)
        sensor_summary = figure_subset(ablation_summary, SENSOR_FIGURE_VARIANTS)
        sensor_features = figure_subset(feature_sets, SENSOR_FIGURE_VARIANTS)
        add(
            "Fig19_rl_feature_ablation_runs.csv",
            "Fig19_ablation_runs",
            type_runs,
            "Fig19_rl_feature_ablation",
            "X=Evaluation_reward; Y=Variant; individual points=Seed; X Error=SD in summary table",
            "The environment retains 31 columns for reward calculation; the Full DQN observation excludes reward-only HR and has 30 features.",
        )
        add(
            "Fig19_rl_feature_ablation_summary.csv",
            "Fig19_ablation_summary",
            type_summary,
            "Fig19_rl_feature_ablation",
            "X=Mean_evaluation_reward; Y=Variant; X Error=SD_evaluation_reward",
            "Mean and SD across matched DQN seeds; higher reward is better within the surrogate environment.",
        )
        add(
            "Fig19_rl_feature_sets.csv",
            "Fig19_feature_sets",
            type_features,
            "Fig19_rl_feature_ablation",
            "Group=Variant; listed rows are the features visible to the DQN",
            "Audit table for the exact observation columns in every ablation variant.",
        )
        add(
            "Fig22_rl_sensor_ablation_runs.csv",
            "Fig22_sensor_runs",
            sensor_runs,
            "Fig22_rl_sensor_ablation",
            "X=Evaluation_reward; Y=Variant; individual points=Seed; X Error=SD in summary",
            "Each No-sensor variant removes the current and all derived features for that sensor; reward-only HR is not a DQN input.",
        )
        add(
            "Fig22_rl_sensor_ablation_summary.csv",
            "Fig22_sensor_summary",
            sensor_summary,
            "Fig22_rl_sensor_ablation",
            "X=Mean_evaluation_reward; Y=Variant; X Error=SD_evaluation_reward",
            "Mean and SD across five matched DQN seeds in the standard surrogate environment.",
        )
        add(
            "Fig22_rl_sensor_feature_sets.csv",
            "Fig22_sensor_features",
            sensor_features,
            "Fig22_rl_sensor_ablation",
            "Group=Variant; listed rows are the features visible to the DQN",
            "Audit table confirming complete removal of the selected sensor feature family.",
        )

    include_protocol = bool(
        cfg.get("reporting", {}).get("include_protocol_surrogate", False)
    )
    protocol_ablation_dir = rl_dir / "protocol_informed_ablation"
    protocol_runs_path = protocol_ablation_dir / "protocol_ablation_runs.csv"
    if include_protocol and protocol_runs_path.exists():
        protocol_runs = pd.read_csv(protocol_runs_path)
        protocol_summary = pd.read_csv(
            protocol_ablation_dir / "protocol_ablation_summary.csv"
        )
        protocol_response = pd.read_csv(
            protocol_ablation_dir / "protocol_response_coefficients.csv"
        )
        protocol_labels = pd.read_csv(
            protocol_ablation_dir / "day4_protocol_labels.csv"
        )
        protocol_features = pd.read_csv(
            protocol_ablation_dir / "protocol_feature_sets.csv"
        )
        type_runs = figure_subset(protocol_runs, FEATURE_TYPE_FIGURE_VARIANTS)
        type_summary = figure_subset(protocol_summary, FEATURE_TYPE_FIGURE_VARIANTS)
        sensor_runs = figure_subset(protocol_runs, SENSOR_FIGURE_VARIANTS)
        sensor_summary = figure_subset(protocol_summary, SENSOR_FIGURE_VARIANTS)
        sensor_features = figure_subset(protocol_features, SENSOR_FIGURE_VARIANTS)
        add(
            "Fig21_protocol_ablation_runs.csv",
            "Fig21_protocol_runs",
            type_runs,
            "Fig21_protocol_informed_feature_ablation",
            "X=Evaluation_reward; Y=Variant; individual points=Seed; X Error=SD from summary",
            "Day 4 protocol-informed counterfactual surrogate; measured sensor values were not modified.",
        )
        add(
            "Fig21_protocol_ablation_summary.csv",
            "Fig21_protocol_summary",
            type_summary,
            "Fig21_protocol_informed_feature_ablation",
            "X=Mean_evaluation_reward; Y=Variant; X Error=SD_evaluation_reward",
            "Exercise/rest response coefficients are estimated from the confirmed Day 4 phase schedule.",
        )
        add(
            "Fig21_protocol_response.csv",
            "Fig21_response",
            protocol_response,
            "Fig21 supporting data",
            "X=State_feature; Y=Normalized_response",
            "Counterfactual response coefficients in normalized RL-state units.",
        )
        add(
            "Fig21_day4_protocol_labels.csv",
            "Fig21_protocol_labels",
            protocol_labels,
            "Fig21 supporting data",
            "X=Minute; Y=Protocol_level; category=Protocol_phase",
            "Confirmed schedule: exercise 0-15, 20-40 and 45-55 min; rest 15-20 and 40-45 min.",
        )
        add(
            "Fig23_protocol_sensor_ablation_runs.csv",
            "Fig23_sensor_runs",
            sensor_runs,
            "Fig23_protocol_informed_sensor_ablation",
            "X=Evaluation_reward; Y=Variant; individual points=Seed; X Error=SD in summary",
            "Day 4 protocol-informed sensor-wise ablation; measurements remain unchanged.",
        )
        add(
            "Fig23_protocol_sensor_ablation_summary.csv",
            "Fig23_sensor_summary",
            sensor_summary,
            "Fig23_protocol_informed_sensor_ablation",
            "X=Mean_evaluation_reward; Y=Variant; X Error=SD_evaluation_reward",
            "Paired five-seed comparison against the complete 30-dimensional DQN observation state.",
        )
        add(
            "Fig23_protocol_sensor_feature_sets.csv",
            "Fig23_sensor_features",
            sensor_features,
            "Fig23_protocol_informed_sensor_ablation",
            "Group=Variant; listed rows are the features visible to the DQN",
            "HR is reward-only and excluded from every DQN observation; six columns are removed for each selected sensor.",
        )

    comparison_dir = rl_dir / "sensor_ablation_comparison"
    paired_ablation_path = comparison_dir / "sensor_ablation_paired_runs.csv"
    if include_protocol and paired_ablation_path.exists():
        paired_ablation = pd.read_csv(paired_ablation_path)
        paired_ablation_summary = pd.read_csv(
            comparison_dir / "sensor_ablation_summary.csv"
        )
        add(
            "Fig24_sensor_ablation_paired_runs.csv",
            "Fig24_paired_runs",
            paired_ablation,
            "Fig24_sensor_ablation_comparison",
            "X=Reward_delta_vs_full; Y=Sensor; panels=Environment; individual points=Seed",
            "Positive values mean that removing the sensor improved surrogate reward; negative values favor Full.",
        )
        add(
            "Fig24_sensor_ablation_summary.csv",
            "Fig24_summary",
            paired_ablation_summary,
            "Fig24_sensor_ablation_comparison",
            "X=Mean_reward_delta; X Error=SD_reward_delta; Y=Sensor; panels=Environment",
            "Paired five-seed statistics include 95% CI, paired t-test, Wilcoxon test and within-environment Holm correction.",
        )

    # Preserve the complete lower-level forecast table for re-analysis in Origin.
    rolling_forecasts = pd.read_csv(evaluation_dir / "day4_rolling_forecasts.csv")
    add(
        "Supplement_all_individual_rolling_forecasts.csv",
        "All_rolling_forecasts",
        rolling_forecasts,
        "Underlying data for rolling, lead-time and consensus figures",
        "X=Predicted_minute or Horizon; Y=Predicted/Actual; group=Target and Origin",
        "All individual rolling forecasts before consensus aggregation.",
    )
    add(
        "Supplement_full_RL_state.csv",
        "Full_RL_state",
        rl_state,
        "Underlying data for RL-state and decision figures",
        "X=Origin_min; select desired current/predicted/slope/uncertainty columns as Y",
        "Complete prediction-derived state supplied to the surrogate environment.",
    )

    guide = pd.DataFrame(guide_rows)
    guide.to_csv(export_dir / "origin_data_manifest.csv", index=False, encoding="utf-8-sig")

    workbook = export_dir / "Origin_all_figure_data.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        guide.to_excel(writer, sheet_name="00_Guide", index=False)
        for sheet_name, table in tables:
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    readme_lines = [
        "# 全部图片的 Origin 源数据",
        "",
        "- `Origin_all_figure_data.xlsx`：包含全部数据图的工作表，第一张表 `00_Guide` 给出每张图的列映射。",
        "- `csv/`：每张图对应的独立 UTF-8 CSV，可逐个导入 Origin。",
        "- `origin_data_manifest.csv`：图名、CSV、Excel sheet、X/Y/Error 列和解释边界的总索引。",
        "",
        "## 重要说明",
        "",
        "- Fig13–Fig15 是流程示意图，不存在实验数值序列；已提供节点坐标、节点文本和连线结构表。",
        "- `Processed signal value` 的物理单位尚未完成标定，导入 Origin 后不要自行改成浓度单位。",
        "- RL reward、risk、action score 和 action 均来自 surrogate/demo environment，不代表已验证的人体闭环干预效果。",
        "- Fig16 的阴影对应 5 个随机种子 20-episode rolling reward 的 across-seed SD。",
        "- Fig18 的 raw 和 denoised 是两种不同评价参照，论文中必须明确区分。",
        "- Fig19 的 Full DQN 输入为 30 维，不包含 HR；内部环境仍保留 HR，仅用于相同的安全约束和 reward 计算。",
        "- Fig22 在普通固定代理环境中逐个移除 Na/K/Ca/TEMP/FR 的完整特征组；HR 不输入 Q-network。",
        "",
        "## Origin 导入",
        "",
        "1. 在 Origin 中选择 File → Import → Excel 或 Single ASCII。",
        "2. 优先打开 `Origin_all_figure_data.xlsx`，从 `00_Guide` 查找对应图的工作表。",
        "3. 按 `Origin_column_mapping` 指定 X、Y、Y Error 或热图 Z 列。",
        "4. 对 Na/K/Ca/FR 等长表使用 Target 作为分组列。",
    ]
    (export_dir / "README_Origin数据说明.md").write_text("\n".join(readme_lines), encoding="utf-8")

    figure_dir = output_root / "prediction_decision_figures"
    figure_bases = sorted({path.stem for path in figure_dir.glob("*.png")})
    if not include_protocol:
        protocol_figures = {
            "Fig21_protocol_informed_feature_ablation",
            "Fig23_protocol_informed_sensor_ablation",
            "Fig24_sensor_ablation_comparison",
        }
        figure_bases = [name for name in figure_bases if name not in protocol_figures]
    pd.DataFrame(
        {
            "Figure": figure_bases,
            "PNG": [f"{name}.png" for name in figure_bases],
            "PDF": [f"{name}.pdf" for name in figure_bases],
            "SVG": [f"{name}.svg" for name in figure_bases],
            "TIFF": [f"{name}.tiff" for name in figure_bases],
        }
    ).to_csv(figure_dir / "figure_manifest.csv", index=False)

    print(export_dir)
    print(f"Exported {len(tables)} data tables plus guide and manifest.")
    print(workbook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
