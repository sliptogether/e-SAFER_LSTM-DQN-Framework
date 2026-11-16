"""Create a publication-ready paired sensor-ablation comparison figure."""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.rl.ablation import SENSOR_VARIANTS
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.robustness_figures import paired_sensor_ablation_figure


def _holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _paired_rows(runs, environment):
    sensor_runs = runs.loc[runs.Variant.isin(SENSOR_VARIANTS)].copy()
    sensor_runs["Environment"] = environment
    sensor_runs["Sensor"] = sensor_runs.Variant.str.replace("No ", "", regex=False)
    sensor_runs["Reward_delta_vs_full"] = sensor_runs[
        "Reward_delta_vs_full_same_seed"
    ]
    return sensor_runs[
        [
            "Environment",
            "Sensor",
            "Variant",
            "Seed",
            "Evaluation_reward",
            "Full_seed_reward",
            "Reward_delta_vs_full",
        ]
    ]


def _summary(paired):
    rows = []
    sensor_order = {sensor: index for index, sensor in enumerate(("Na", "K", "Ca", "TEMP", "FR"))}
    for (environment, sensor), group in paired.groupby(["Environment", "Sensor"]):
        delta = group.Reward_delta_vs_full.to_numpy(float)
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1))
        sem = stats.sem(delta)
        ci_half = float(stats.t.ppf(0.975, len(delta) - 1) * sem)
        paired_t_p = float(stats.ttest_1samp(delta, popmean=0.0).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "Environment": environment,
                "Sensor_order": sensor_order[sensor],
                "Sensor": sensor,
                "N_matched_seeds": len(delta),
                "Mean_reward_delta": mean,
                "SD_reward_delta": sd,
                "CI95_low": mean - ci_half,
                "CI95_high": mean + ci_half,
                "Paired_t_p": paired_t_p,
                "Wilcoxon_p": wilcoxon_p,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["Environment", "Sensor_order"])
    summary["Holm_adjusted_p"] = np.nan
    for environment, indices in summary.groupby("Environment").groups.items():
        summary.loc[indices, "Holm_adjusted_p"] = _holm_adjust(
            summary.loc[indices, "Paired_t_p"].to_numpy(float)
        )
    return summary.reset_index(drop=True)


def main(config):
    cfg = load_config(config)
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    standard = pd.read_csv(output_root / "rl" / "ablation" / "feature_ablation_runs.csv")
    protocol = pd.read_csv(
        output_root
        / "rl"
        / "protocol_informed_ablation"
        / "protocol_ablation_runs.csv"
    )
    paired = pd.concat(
        [
            _paired_rows(standard, "Standard surrogate"),
            _paired_rows(protocol, "Day 4 protocol-informed"),
        ],
        ignore_index=True,
    )
    summary = _summary(paired)

    result_dir = ensure_dir(output_root / "rl" / "sensor_ablation_comparison")
    figure_dir = ensure_dir(output_root / "prediction_decision_figures")
    origin_dir = ensure_dir(figure_dir / "source_data" / "origin")
    paired.to_csv(result_dir / "sensor_ablation_paired_runs.csv", index=False)
    summary.to_csv(result_dir / "sensor_ablation_summary.csv", index=False)

    out_base = figure_dir / "Fig24_sensor_ablation_comparison"
    paired_sensor_ablation_figure(paired, out_base)

    paired.to_csv(
        origin_dir / "Fig24_sensor_ablation_paired_runs_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        origin_dir / "Fig24_sensor_ablation_summary_origin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    workbook = origin_dir / "Origin_plot_data_Fig24.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        paired.to_excel(writer, sheet_name="Fig24_paired_runs", index=False)
        summary.to_excel(writer, sheet_name="Fig24_summary", index=False)

    print(summary.to_string(index=False), flush=True)
    print(out_base.with_suffix(".png"), flush=True)
    print(workbook, flush=True)
    print(
        "WARNING: paired differences compare surrogate policies across matched seeds; "
        "they do not validate human closed-loop control.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
