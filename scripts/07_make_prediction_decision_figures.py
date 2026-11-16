import argparse
import shutil
from pathlib import Path

import pandas as pd

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.analysis_figures import (
    decision_summary_figure,
    decision_timeline_figure,
    dqn_reward_figure,
    horizon_comparison_figure,
    horizon_metrics_table,
    lead_time_figure,
    lead_time_metrics,
    parity_figure,
    policy_score_figure,
    residual_time_figure,
    state_change_heatmap_figure,
    surrogate_risk_figure,
    uncertainty_error_figure,
)
from sweat_rl.visualization.figures import rolling_prediction_figures


def main(config):
    cfg = load_config(config)
    root = Path(cfg["_root"])
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    evaluation_dir = output_root / "evaluation"
    rl_dir = output_root / "rl"
    figure_dir = ensure_dir(output_root / "prediction_decision_figures")
    source_dir = ensure_dir(figure_dir / "source_data")

    targets = tuple(cfg.get("figures", {}).get("prediction_targets", ["Na", "K", "Ca", "FR"]))
    horizon = int(cfg["data"]["forecast_horizon_min"])
    consensus_csv = evaluation_dir / "day4_rolling_consensus.csv"
    forecasts_csv = evaluation_dir / "day4_rolling_forecasts.csv"
    decision_csv = rl_dir / "decision_timeline.csv"
    reward_csv = rl_dir / "dqn_rewards.csv"

    required = (consensus_csv, forecasts_csv, decision_csv, reward_csv)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required generated data are missing: " + ", ".join(missing))

    rolling_prediction_figures(consensus_csv, figure_dir, targets=targets, horizon=horizon)
    parity_figure(consensus_csv, figure_dir / "Fig02_prediction_parity", targets=targets)
    residual_time_figure(consensus_csv, figure_dir / "Fig03_residuals_over_time", targets=targets)

    lead_df = lead_time_metrics(forecasts_csv, targets=targets)
    lead_df.to_csv(source_dir / "lead_time_metrics.csv", index=False)
    lead_time_figure(lead_df, figure_dir / "Fig04_lead_time_performance", targets=targets)

    metric_paths = {
        3: root / "outputs_denoised_3min" / "evaluation" / "metrics.json",
        5: root / "outputs_denoised_5min" / "evaluation" / "metrics.json",
        10: root / "outputs_denoised_10min" / "evaluation" / "metrics.json",
    }
    horizon_df = horizon_metrics_table(metric_paths, targets=targets)
    horizon_df.to_csv(source_dir / "fixed_horizon_metrics.csv", index=False)
    horizon_comparison_figure(horizon_df, figure_dir / "Fig05_fixed_horizon_comparison", targets=targets)

    uncertainty_error_figure(consensus_csv, figure_dir / "Fig06_uncertainty_vs_error", targets=targets)
    decision_timeline_figure(decision_csv, figure_dir / "Fig07_decision_timeline")
    policy_score_figure(decision_csv, figure_dir / "Fig08_policy_action_scores")
    surrogate_risk_figure(decision_csv, figure_dir / "Fig09_surrogate_risk")
    state_change_heatmap_figure(decision_csv, figure_dir / "Fig10_predicted_state_change_heatmap")
    dqn_reward_figure(reward_csv, figure_dir / "Fig11_dqn_training_reward")
    decision_summary_figure(decision_csv, figure_dir / "Fig12_decision_summary")

    source_files = {
        consensus_csv: "day4_rolling_consensus.csv",
        forecasts_csv: "day4_rolling_forecasts.csv",
        decision_csv: "decision_timeline.csv",
        reward_csv: "dqn_rewards.csv",
        rl_dir / "decision_segments.csv": "decision_segments.csv",
        rl_dir / "rl_state.csv": "rl_state.csv",
    }
    for source, name in source_files.items():
        if source.exists():
            shutil.copy2(source, source_dir / name)

    figure_bases = sorted({path.stem for path in figure_dir.glob("*.png")})
    manifest = pd.DataFrame(
        {
            "Figure": figure_bases,
            "PNG": [f"{name}.png" for name in figure_bases],
            "PDF": [f"{name}.pdf" for name in figure_bases],
            "SVG": [f"{name}.svg" for name in figure_bases],
            "TIFF": [f"{name}.tiff" for name in figure_bases],
        }
    )
    manifest.to_csv(figure_dir / "figure_manifest.csv", index=False)
    print(figure_dir)
    print(f"Generated {len(figure_bases)} figure sets in PNG/PDF/SVG/TIFF.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
