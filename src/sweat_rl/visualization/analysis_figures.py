"""Publication-style diagnostics for rolling forecasts and the RL demo.

The decision panels in this module visualise a surrogate demonstration only.
They must not be interpreted as validated physiological intervention effects.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Import the existing figure backend first so it configures the project-local
# Matplotlib cache and the non-interactive Agg backend before pyplot starts.
from sweat_rl.visualization.figures import _PUB_STYLE, _save_figure_bundle

from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from sweat_rl.metrics import regression_metrics


TARGETS = ("Na", "K", "Ca", "FR")
TARGET_LABELS = {
    "Na": r"Na$^+$",
    "K": r"K$^+$",
    "Ca": r"Ca$^{2+}$",
    "FR": "FR",
}
TARGET_COLORS = {
    "Na": "#2563A6",
    "K": "#168C89",
    "Ca": "#7656A5",
    "FR": "#C15A42",
}
ACTION_ORDER = ("Reduce", "Maintain", "Increase")
ACTION_COLORS = {
    "Reduce": "#3775BA",
    "Maintain": "#8C8C8C",
    "Increase": "#E28E2C",
}


def _panel_label(ax, index):
    ax.text(
        -0.15,
        1.04,
        chr(ord("a") + index),
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
    )


def _grid_y(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)


def _mark_unobserved_tail(ax, start=50.5, end=55.5):
    ax.axvspan(start, end, color="#D8D8D8", alpha=0.35, linewidth=0)
    ax.text(
        0.985,
        0.97,
        "Full 5-min outcome\nnot observed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#666666",
        fontsize=6,
    )


def _decision_footer(fig):
    fig.text(
        0.995,
        0.008,
        "Simulation-only policy demonstration; not validated human control",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#666666",
    )


def parity_figure(consensus_csv, out_base, targets=TARGETS):
    df = pd.read_csv(consensus_csv)
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
        for i, (ax, target) in enumerate(zip(axes.ravel(), targets)):
            d = df[(df.Target == target) & df.Predicted.notna()].copy()
            actual = d.Actual.to_numpy(float)
            predicted = d.Predicted.to_numpy(float)
            metrics = regression_metrics(actual, predicted)
            lower = float(min(actual.min(), predicted.min()))
            upper = float(max(actual.max(), predicted.max()))
            pad = max((upper - lower) * 0.07, 1e-6)
            bounds = (lower - pad, upper + pad)
            ax.plot(bounds, bounds, color="#777777", linestyle="--", linewidth=0.9)
            ax.scatter(
                actual,
                predicted,
                s=20,
                color=TARGET_COLORS[target],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
            )
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(TARGET_LABELS[target], fontweight="bold")
            ax.set_xlabel("Measured processed value")
            ax.set_ylabel("Rolling predicted value")
            ax.text(
                0.04,
                0.95,
                f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nn = {len(d)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
            )
            _grid_y(ax)
            _panel_label(ax, i)
        fig.suptitle(
            "Agreement of Day 4 rolling forecasts with denoised observations",
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2, w_pad=1.2)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def residual_time_figure(consensus_csv, out_base, targets=TARGETS):
    df = pd.read_csv(consensus_csv)
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
        for i, (ax, target) in enumerate(zip(axes.ravel(), targets)):
            d = df[(df.Target == target) & df.Predicted.notna()].sort_values("Minute")
            minute = d.Minute.to_numpy(float)
            residual = (d.Predicted - d.Actual).to_numpy(float)
            sd = d.Consensus_SD.to_numpy(float)
            ax.fill_between(
                minute,
                -sd,
                sd,
                color=TARGET_COLORS[target],
                alpha=0.14,
                linewidth=0,
                label="Consensus SD around zero",
            )
            ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9)
            ax.plot(
                minute,
                residual,
                color=TARGET_COLORS[target],
                linewidth=1.3,
                marker="o",
                markersize=2.4,
                markevery=4,
                label="Prediction − measured",
            )
            ax.set_title(TARGET_LABELS[target], fontweight="bold")
            ax.set_xlim(10, 55)
            ax.set_xticks(np.arange(10, 56, 5))
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Residual")
            _grid_y(ax)
            _panel_label(ax, i)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle("Rolling forecast residuals across Day 4", fontweight="bold", y=1.035)
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2, w_pad=1.2)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def lead_time_metrics(forecasts_csv, targets=TARGETS):
    df = pd.read_csv(forecasts_csv)
    rows = []
    for target in targets:
        for lead in sorted(df.Horizon.unique()):
            d = df[(df.Target == target) & (df.Horizon == lead) & df.Actual.notna()]
            if len(d) < 2:
                continue
            metrics = regression_metrics(d.Actual, d.Predicted)
            actual_sd = float(d.Actual.std(ddof=0))
            rows.append(
                {
                    "Target": target,
                    "Lead_min": int(lead),
                    "n": len(d),
                    **metrics,
                    "nrmse": metrics["rmse"] / actual_sd if actual_sd > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def lead_time_figure(lead_df, out_base, targets=TARGETS):
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
        for target in targets:
            d = lead_df[lead_df.Target == target].sort_values("Lead_min")
            axes[0].plot(
                d.Lead_min,
                d.r2,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=TARGET_COLORS[target],
                label=TARGET_LABELS[target],
            )
            axes[1].plot(
                d.Lead_min,
                d.nrmse,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=TARGET_COLORS[target],
                label=TARGET_LABELS[target],
            )
        axes[0].axhline(0.75, color="#777777", linestyle="--", linewidth=0.8)
        axes[0].text(5.05, 0.75, "0.75", va="center", fontsize=6, color="#666666")
        axes[0].set_ylabel("R²")
        axes[1].set_ylabel("Normalized RMSE")
        for i, ax in enumerate(axes):
            ax.set_xlabel("Forecast lead (min)")
            ax.set_xticks(np.arange(1, 6))
            _grid_y(ax)
            _panel_label(ax, i)
        axes[1].legend(loc="upper left", ncol=2, fontsize=6.5)
        fig.suptitle(
            "Day 4 forecast performance by lead time",
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.6)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def horizon_metrics_table(metric_paths, targets=TARGETS):
    import json

    rows = []
    for horizon, path in sorted(metric_paths.items()):
        with open(path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for target in targets:
            rows.append({"Horizon_min": horizon, "Target": target, **metrics[target]})
    return pd.DataFrame(rows)


def horizon_comparison_figure(horizon_df, out_base, targets=TARGETS):
    horizons = sorted(horizon_df.Horizon_min.unique())
    x = np.arange(len(targets), dtype=float)
    width = 0.22
    colors = {3: "#2F7F5F", 5: "#3D73A8", 10: "#B55B52"}
    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.45))
        for j, horizon in enumerate(horizons):
            values = [
                float(
                    horizon_df[
                        (horizon_df.Horizon_min == horizon) & (horizon_df.Target == target)
                    ].r2.iloc[0]
                )
                for target in targets
            ]
            offset = (j - (len(horizons) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=colors.get(horizon, "#777777"),
                label=f"{horizon} min",
            )
            for bar, value in zip(bars, values):
                va = "bottom" if value >= 0 else "top"
                y = value + (0.025 if value >= 0 else -0.025)
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.2f}", ha="center", va=va, fontsize=5.8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.axhline(0.75, color="#777777", linestyle="--", linewidth=0.8)
        ax.text(len(targets) - 0.55, 0.77, "R² = 0.75", fontsize=6, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels([TARGET_LABELS[target] for target in targets])
        ax.set_ylabel("Fixed-horizon Day 4 R²")
        ax.set_title("Prediction accuracy declines as the forecast horizon increases", fontweight="bold")
        ax.legend(ncol=3, loc="lower left")
        _grid_y(ax)
        fig.tight_layout()
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def uncertainty_error_figure(consensus_csv, out_base, targets=TARGETS):
    df = pd.read_csv(consensus_csv)
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
        for i, (ax, target) in enumerate(zip(axes.ravel(), targets)):
            d = df[(df.Target == target) & df.Predicted.notna()].copy()
            uncertainty = d.Consensus_SD.to_numpy(float)
            error = np.abs(d.Predicted.to_numpy(float) - d.Actual.to_numpy(float))
            valid = np.isfinite(uncertainty) & np.isfinite(error)
            correlation = (
                float(np.corrcoef(uncertainty[valid], error[valid])[0, 1])
                if valid.sum() > 2 and np.std(uncertainty[valid]) > 0 and np.std(error[valid]) > 0
                else np.nan
            )
            ax.scatter(
                uncertainty[valid],
                error[valid],
                s=20,
                color=TARGET_COLORS[target],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
            )
            ax.set_title(TARGET_LABELS[target], fontweight="bold")
            ax.set_xlabel("Rolling consensus SD")
            ax.set_ylabel("Absolute prediction error")
            ax.text(
                0.96,
                0.94,
                f"Pearson r = {correlation:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
            )
            _grid_y(ax)
            _panel_label(ax, i)
        fig.suptitle(
            "Model disagreement versus observed forecast error",
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2, w_pad=1.2)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def decision_timeline_figure(decision_csv, out_base):
    df = pd.read_csv(decision_csv).sort_values("Minute")
    minute = df.Minute.to_numpy(float)
    code_map = {name: i for i, name in enumerate(ACTION_ORDER)}
    codes = df.Action_en.map(code_map).to_numpy(int)
    cmap = ListedColormap([ACTION_COLORS[name] for name in ACTION_ORDER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 3.7),
            gridspec_kw={"height_ratios": [0.7, 2.3]},
            sharex=True,
        )
        axes[0].imshow(
            codes[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(minute.min() - 0.5, minute.max() + 0.5, 0, 1),
        )
        axes[0].set_yticks([])
        axes[0].set_ylabel("Action", rotation=0, ha="right", va="center")
        axes[0].spines["left"].set_visible(False)
        axes[0].legend(
            handles=[Patch(facecolor=ACTION_COLORS[name], label=name) for name in ACTION_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.65),
            ncol=3,
        )
        axes[1].plot(minute, df.Policy_score, color="#292929", linewidth=1.3)
        for action in ACTION_ORDER:
            d = df[df.Action_en == action]
            axes[1].scatter(
                d.Minute,
                d.Policy_score,
                s=20,
                color=ACTION_COLORS[action],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axes[1].axhline(1 / 3, color="#777777", linestyle="--", linewidth=0.8)
        axes[1].text(10.4, 1 / 3 + 0.012, "Equal-score reference", fontsize=6, color="#666666")
        axes[1].set_ylabel("Selected-action score")
        axes[1].set_xlabel("Decision time (min)")
        axes[1].set_xticks(np.arange(10, 56, 5))
        axes[1].set_xlim(9.5, 55.5)
        _grid_y(axes[1])
        _mark_unobserved_tail(axes[1])
        fig.suptitle("Surrogate policy actions from minute 10 to 55", fontweight="bold", y=0.995)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.035, 1, 0.94), h_pad=0.35)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def policy_score_figure(decision_csv, out_base):
    df = pd.read_csv(decision_csv).sort_values("Minute")
    score_columns = {
        "Reduce": "Score_reduce",
        "Maintain": "Score_maintain",
        "Increase": "Score_increase",
    }
    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        for action in ACTION_ORDER:
            ax.plot(
                df.Minute,
                df[score_columns[action]],
                color=ACTION_COLORS[action],
                linewidth=1.35,
                label=action,
            )
        ax.axhline(1 / 3, color="#777777", linestyle="--", linewidth=0.8)
        _mark_unobserved_tail(ax)
        ax.set_xlim(9.5, 55.5)
        ax.set_xticks(np.arange(10, 56, 5))
        ax.set_xlabel("Decision time (min)")
        ax.set_ylabel("Softmax action score")
        ax.set_title("Relative action scores of the surrogate DQN policy", fontweight="bold")
        ax.legend(ncol=3, loc="upper center")
        _grid_y(ax)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def surrogate_risk_figure(decision_csv, out_base):
    df = pd.read_csv(decision_csv).sort_values("Minute")
    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.plot(
            df.Minute,
            df.Surrogate_future_risk,
            color="#292929",
            linewidth=1.25,
            zorder=1,
        )
        for action in ACTION_ORDER:
            d = df[df.Action_en == action]
            ax.scatter(
                d.Minute,
                d.Surrogate_future_risk,
                s=22,
                color=ACTION_COLORS[action],
                edgecolor="white",
                linewidth=0.4,
                label=action,
                zorder=3,
            )
        _mark_unobserved_tail(ax)
        ax.set_xlim(9.5, 55.5)
        ax.set_xticks(np.arange(10, 56, 5))
        ax.set_xlabel("Decision time (min)")
        ax.set_ylabel("Surrogate future-risk score")
        ax.set_title("Prediction-derived surrogate risk used by the demo", fontweight="bold")
        ax.legend(ncol=3, loc="upper right")
        _grid_y(ax)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def state_change_heatmap_figure(decision_csv, out_base):
    df = pd.read_csv(decision_csv).sort_values("Minute")
    features = ("Na", "K", "Ca", "TEMP", "FR")
    matrix = np.vstack(
        [df[f"{feature}_predicted_5min_change"].to_numpy(float) for feature in features]
    )
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_sd = matrix.std(axis=1, keepdims=True)
    standardized = np.divide(
        matrix - row_mean,
        row_sd,
        out=np.zeros_like(matrix),
        where=row_sd > 0,
    )
    limit = max(1.0, float(np.nanmax(np.abs(standardized))))
    action_codes = df.Action_en.map({name: i for i, name in enumerate(ACTION_ORDER)}).to_numpy(int)
    action_cmap = ListedColormap([ACTION_COLORS[name] for name in ACTION_ORDER])
    action_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], action_cmap.N)
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 3.8),
            gridspec_kw={"height_ratios": [0.45, 3.2]},
            sharex=True,
        )
        extent = (df.Minute.min() - 0.5, df.Minute.max() + 0.5, 0, 1)
        axes[0].imshow(
            action_codes[np.newaxis, :],
            aspect="auto",
            cmap=action_cmap,
            norm=action_norm,
            extent=extent,
        )
        axes[0].set_yticks([])
        axes[0].set_ylabel("Action", rotation=0, ha="right", va="center")
        axes[0].spines["left"].set_visible(False)
        image = axes[1].imshow(
            standardized,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            extent=(df.Minute.min() - 0.5, df.Minute.max() + 0.5, len(features) - 0.5, -0.5),
        )
        axes[1].set_yticks(np.arange(len(features)))
        axes[1].set_yticklabels([TARGET_LABELS.get(feature, feature) for feature in features])
        axes[1].set_xticks(np.arange(10, 56, 5))
        axes[1].set_xlabel("Decision time (min)")
        axes[1].set_ylabel("Predicted variable")
        colorbar = fig.colorbar(image, ax=axes[1], pad=0.015, fraction=0.035)
        colorbar.set_label("Within-variable standardized predicted 5-min change")
        axes[1].axvline(50.5, color="#555555", linestyle="--", linewidth=0.8)
        axes[1].text(55.2, -0.26, "Incomplete outcome", ha="right", va="center", fontsize=6, color="#555555")
        fig.suptitle(
            "Forecast-state changes presented to the surrogate decision demo",
            fontweight="bold",
            y=0.995,
        )
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.035, 1, 0.96), h_pad=0.25)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def dqn_reward_figure(reward_csv, out_base):
    df = pd.read_csv(reward_csv)
    smooth = df.Reward.rolling(20, min_periods=20).mean()
    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.plot(df.Episode, df.Reward, color="#7B9FC2", linewidth=0.7, alpha=0.5, label="Episode reward")
        ax.plot(df.Episode, smooth, color="#174A7E", linewidth=1.7, label="20-episode mean")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Surrogate reward")
        ax.set_title("Surrogate DQN training convergence", fontweight="bold")
        ax.legend(loc="lower right")
        _grid_y(ax)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def decision_summary_figure(decision_csv, out_base):
    df = pd.read_csv(decision_csv)
    rng = np.random.default_rng(42)
    with plt.rc_context(_PUB_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
        counts = df.Action_en.value_counts().reindex(ACTION_ORDER, fill_value=0)
        bars = axes[0].bar(
            np.arange(len(ACTION_ORDER)),
            counts.values,
            color=[ACTION_COLORS[name] for name in ACTION_ORDER],
            width=0.65,
        )
        axes[0].set_xticks(np.arange(len(ACTION_ORDER)))
        axes[0].set_xticklabels(ACTION_ORDER)
        axes[0].set_ylabel("Number of decision minutes")
        axes[0].set_title("Action frequency", fontweight="bold")
        axes[0].bar_label(bars, padding=2, fontsize=6.5)
        _grid_y(axes[0])
        _panel_label(axes[0], 0)

        for index, action in enumerate(ACTION_ORDER):
            values = df.loc[df.Action_en == action, "Policy_score"].to_numpy(float)
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.035, size=len(values))
            axes[1].scatter(
                np.full(len(values), index) + jitter,
                values,
                s=20,
                color=ACTION_COLORS[action],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
            )
            axes[1].plot(
                [index - 0.16, index + 0.16],
                [np.median(values), np.median(values)],
                color="#222222",
                linewidth=1.4,
            )
        axes[1].axhline(1 / 3, color="#777777", linestyle="--", linewidth=0.8)
        axes[1].set_xticks(np.arange(len(ACTION_ORDER)))
        axes[1].set_xticklabels(ACTION_ORDER)
        axes[1].set_ylabel("Selected-action score")
        axes[1].set_title("Score distribution", fontweight="bold")
        _grid_y(axes[1])
        _panel_label(axes[1], 1)
        fig.suptitle("Summary of surrogate policy decisions", fontweight="bold", y=0.995)
        _decision_footer(fig)
        fig.tight_layout(rect=(0, 0.035, 1, 0.96), w_pad=1.8)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)
