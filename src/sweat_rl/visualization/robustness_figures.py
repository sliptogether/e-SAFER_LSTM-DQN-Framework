"""Robustness figures for the 5-min forecasting and surrogate RL demo.

The RL panels describe performance inside the configured surrogate environment
only.  They are not evidence of validated physiological control.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sweat_rl.visualization.figures import _PUB_STYLE, _save_figure_bundle

from matplotlib import pyplot as plt


TARGETS = ("Na", "K", "Ca", "FR")
TARGET_LABELS = {
    "Na": r"Na$^+$",
    "K": r"K$^+$",
    "Ca": r"Ca$^{2+}$",
    "FR": "FR",
}


def _grid_y(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)


def _decision_footer(fig):
    fig.text(
        0.995,
        0.008,
        "Simulation-only surrogate environment; not validated human control",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#666666",
    )


def multiseed_convergence_figure(reward_csv, out_base, rolling_window=20):
    """Plot the across-seed mean and SD of rolling episodic reward."""
    df = pd.read_csv(reward_csv).sort_values(["Seed", "Episode"])
    df["Rolling_reward"] = df.groupby("Seed")["Reward"].transform(
        lambda values: values.rolling(rolling_window, min_periods=rolling_window).mean()
    )
    summary = (
        df.dropna(subset=["Rolling_reward"])
        .groupby("Episode", as_index=False)
        .agg(
            Mean_reward=("Rolling_reward", "mean"),
            SD_reward=("Rolling_reward", "std"),
            n_seeds=("Seed", "nunique"),
        )
    )
    seeds = sorted(df.Seed.unique())

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.25))
        for seed in seeds:
            d = df[(df.Seed == seed) & df.Rolling_reward.notna()]
            ax.plot(
                d.Episode,
                d.Rolling_reward,
                color="#A9B7C6",
                linewidth=0.65,
                alpha=0.48,
                zorder=1,
            )
        lower = summary.Mean_reward - summary.SD_reward.fillna(0)
        upper = summary.Mean_reward + summary.SD_reward.fillna(0)
        ax.fill_between(
            summary.Episode,
            lower,
            upper,
            color="#3775BA",
            alpha=0.18,
            linewidth=0,
            label="Across-seed SD",
            zorder=2,
        )
        ax.plot(
            summary.Episode,
            summary.Mean_reward,
            color="#174A7E",
            linewidth=1.8,
            label=f"Mean {rolling_window}-episode reward",
            zorder=3,
        )
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Surrogate episodic reward")
        ax.set_title(
            f"DQN training convergence is reproducible across {len(seeds)} seeds",
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.06,
            f"Lines: individual seeds; shading: mean ± SD (n = {len(seeds)})",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#555555",
        )
        ax.legend(loc="lower right")
        _grid_y(ax)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)

    return summary


def policy_benchmark_figure(benchmark_csv, out_base):
    """Compare greedy DQN with random and deterministic surrogate policies."""
    df = pd.read_csv(benchmark_csv)
    order = ["Random", "Maintain", "Rule-based", "DQN"]
    colors = {
        "Random": "#A7A7A7",
        "Maintain": "#7D8B99",
        "Rule-based": "#4A8F6A",
        "DQN": "#3775BA",
    }
    rng = np.random.default_rng(2026)

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.35))
        for index, method in enumerate(order):
            values = df.loc[df.Method == method, "Reward"].to_numpy(float)
            if not len(values):
                continue
            jitter = rng.normal(0, 0.045, size=len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=24,
                color=colors[method],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.85,
                zorder=3,
            )
            mean = float(values.mean())
            ax.plot(
                [index - 0.18, index + 0.18],
                [mean, mean],
                color="#202020",
                linewidth=1.5,
                zorder=4,
            )
            ax.text(
                index,
                values.min() - 0.45,
                f"n={len(values)}",
                ha="center",
                va="top",
                fontsize=6,
                color="#555555",
            )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order)
        ax.set_ylabel("Greedy-policy surrogate reward\n(higher is better)")
        ax.set_title(
            "Policy benchmark within the configured surrogate environment",
            fontweight="bold",
        )
        _grid_y(ax)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def raw_denoised_performance_table(denoised_json, raw_json, targets=TARGETS):
    with open(denoised_json, "r", encoding="utf-8") as handle:
        denoised = json.load(handle)
    with open(raw_json, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = []
    for target in targets:
        for reference, metrics in (
            ("Raw target", raw[target]),
            ("Denoised target", denoised[target]),
        ):
            rows.append({"Target": target, "Reference": reference, **metrics})
    return pd.DataFrame(rows)


def raw_denoised_performance_figure(metric_df, out_base, targets=TARGETS):
    """Show how the evaluation target definition changes reported 5-min metrics."""
    references = ["Raw target", "Denoised target"]
    colors = {"Raw target": "#A6A6A6", "Denoised target": "#3775BA"}
    x = np.arange(len(targets), dtype=float)
    width = 0.34

    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
        for j, reference in enumerate(references):
            subset = metric_df[metric_df.Reference == reference].set_index("Target")
            r2_values = [float(subset.loc[target, "r2"]) for target in targets]
            rmse_ratios = []
            for target in targets:
                raw_rmse = float(
                    metric_df[
                        (metric_df.Target == target) & (metric_df.Reference == "Raw target")
                    ].rmse.iloc[0]
                )
                rmse_ratios.append(float(subset.loc[target, "rmse"]) / raw_rmse)
            offset = (j - 0.5) * width
            bars = axes[0].bar(
                x + offset,
                r2_values,
                width=width,
                color=colors[reference],
                label=reference,
            )
            axes[1].bar(
                x + offset,
                rmse_ratios,
                width=width,
                color=colors[reference],
                label=reference,
            )
            for bar, value in zip(bars, r2_values):
                y = value + (0.035 if value >= 0 else -0.035)
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=5.8,
                )

        axes[0].axhline(0, color="#333333", linewidth=0.8)
        axes[0].axhline(0.75, color="#777777", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Fixed 5-min Day 4 R²")
        axes[0].set_title("Reported R²", fontweight="bold")
        axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("RMSE relative to raw-target evaluation")
        axes[1].set_title("Relative error", fontweight="bold")
        for index, ax in enumerate(axes):
            ax.set_xticks(x)
            ax.set_xticklabels([TARGET_LABELS[target] for target in targets])
            _grid_y(ax)
            ax.text(
                -0.15,
                1.04,
                chr(ord("a") + index),
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=8,
            )
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=2,
        )
        fig.suptitle(
            "The evaluation reference materially affects apparent forecast performance",
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.01,
            "The same 5-min forecasts are evaluated against raw or causally denoised Day 4 targets.",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.86), w_pad=1.7)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def feature_ablation_figure(
    runs_df,
    out_base,
    title="Feature-group ablation of the DQN observation state",
):
    """Compare DQN observation feature sets with a fixed full-state reward."""
    required = {
        "Variant", "Variant_order", "State_dim", "Seed", "Evaluation_reward"
    }
    missing = required.difference(runs_df.columns)
    if missing:
        raise ValueError(f"Missing ablation columns: {sorted(missing)}")

    ordered = (
        runs_df[["Variant", "Variant_order", "State_dim"]]
        .drop_duplicates()
        .sort_values("Variant_order")
    )
    full_mean = float(
        runs_df.loc[runs_df.Variant == "Full", "Evaluation_reward"].mean()
    )
    rng = np.random.default_rng(20260907)

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.7))
        for y, row in enumerate(ordered.itertuples(index=False)):
            values = runs_df.loc[
                runs_df.Variant == row.Variant, "Evaluation_reward"
            ].to_numpy(float)
            color = (
                "#2868A8" if row.Variant == "Full"
                else "#C66B32" if row.Variant == "Current only"
                else "#7D8B99"
            )
            jitter = rng.normal(0.0, 0.055, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), y) + jitter,
                s=25,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.82,
                zorder=3,
            )
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ax.errorbar(
                mean,
                y,
                xerr=sd,
                fmt="D",
                markersize=4.2,
                color="#202020",
                markerfacecolor=color,
                markeredgewidth=0.7,
                capsize=2.5,
                linewidth=1.1,
                zorder=4,
            )
            ax.text(
                mean,
                y - 0.19,
                f"Δ={mean - full_mean:+.2f}",
                ha="center",
                va="bottom",
                fontsize=5.6,
                color="#555555",
            )

        labels = [
            f"{row.Variant}  ({int(row.State_dim)}D)"
            for row in ordered.itertuples(index=False)
        ]
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(full_mean, color="#2868A8", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Greedy-policy surrogate reward (higher is better)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
        fig.text(
            0.50,
            0.045,
            "Points: seeds; diamond and bar: mean ± SD; Δ relative to Full",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.085, 1, 1))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def paired_sensor_ablation_figure(paired_df, out_base):
    """Compare matched sensor-ablation reward deltas across two environments."""
    required = {"Environment", "Sensor", "Seed", "Reward_delta_vs_full"}
    missing = required.difference(paired_df.columns)
    if missing:
        raise ValueError(f"Missing paired-ablation columns: {sorted(missing)}")

    environments = ["Standard surrogate", "Day 4 protocol-informed"]
    sensors = ["Na", "K", "Ca", "TEMP", "FR"]
    sensor_labels = {
        "Na": r"Na$^+$",
        "K": r"K$^+$",
        "Ca": r"Ca$^{2+}$",
        "TEMP": "TEMP",
        "FR": "FR",
    }
    colors = {
        "Standard surrogate": "#557A9E",
        "Day 4 protocol-informed": "#C97845",
    }
    limit = max(1.0, float(np.nanmax(np.abs(paired_df.Reward_delta_vs_full))) * 1.18)
    rng = np.random.default_rng(20260907)

    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
        for panel_index, (ax, environment) in enumerate(zip(axes, environments)):
            panel = paired_df.loc[paired_df.Environment == environment]
            color = colors[environment]
            for y, sensor in enumerate(sensors):
                values = panel.loc[
                    panel.Sensor == sensor, "Reward_delta_vs_full"
                ].to_numpy(float)
                jitter = rng.normal(0.0, 0.055, size=len(values))
                ax.scatter(
                    values,
                    np.full(len(values), y) + jitter,
                    s=24,
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.78,
                    zorder=3,
                )
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ax.errorbar(
                    mean,
                    y,
                    xerr=sd,
                    fmt="D",
                    markersize=4.1,
                    color="#202020",
                    markerfacecolor=color,
                    markeredgewidth=0.7,
                    capsize=2.4,
                    linewidth=1.0,
                    zorder=4,
                )
                ax.text(
                    mean,
                    y - 0.18,
                    f"{mean:+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color="#555555",
                )

            ax.axvline(0.0, color="#303030", linestyle="--", linewidth=0.8)
            ax.set_xlim(-limit, limit)
            ax.set_title(
                f"{chr(ord('a') + panel_index)}  {environment}",
                loc="left",
                fontweight="bold",
            )
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.75)
            ax.set_axisbelow(True)
            ax.text(
                0.02,
                0.98,
                "Full better",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color="#557A9E",
            )
            ax.text(
                0.98,
                0.98,
                "Removal better",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.5,
                color="#9B5A31",
            )

        axes[0].set_yticks(np.arange(len(sensors)))
        axes[0].set_yticklabels([sensor_labels[sensor] for sensor in sensors])
        axes[0].invert_yaxis()
        fig.suptitle(
            "Sensor-wise ablation of the surrogate DQN policy",
            fontweight="bold",
            y=0.995,
        )
        fig.supxlabel(
            r"Paired reward difference, $R_{No\ sensor}-R_{Full}$",
            y=0.065,
        )
        fig.text(
            0.5,
            0.025,
            "Points: five matched seeds; diamond and bar: mean ± SD",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.10, 1, 0.94), w_pad=1.5)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)
