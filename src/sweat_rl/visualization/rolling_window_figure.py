"""Publication-style view of independent rolling forecast windows."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from sweat_rl.visualization.figures import _PUB_STYLE, _save_figure_bundle


TARGETS = ("Na", "K", "Ca", "FR")
TARGET_LABELS = {
    "Na": r"Na$^+$",
    "K": r"K$^+$",
    "Ca": r"Ca$^{2+}$",
    "FR": "FR",
}

CYCLING_COLOR = "#DCEFF7"
REST_COLOR = "#FFF2A8"
RAW_COLOR = "#BBC1C7"
MEASURED_COLOR = "#1F1F1F"
ORIGIN_MARKER_COLOR = "#C15C2E"


def _shade_protocol(ax, protocol, alpha=0.25):
    for stage in protocol:
        color = CYCLING_COLOR if stage["Stage"] == "Cycling" else REST_COLOR
        ax.axvspan(
            stage["Start_min"],
            stage["End_min"],
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )


def rolling_window_source_table(processed_csv, detail_csv, displayed_origins):
    """Return clean source data for the forecast segments shown in the figure."""
    measured = pd.read_csv(processed_csv).sort_values("Minute")
    detail = pd.read_csv(detail_csv)
    displayed = detail[
        detail["Origin"].astype(int).isin([int(value) for value in displayed_origins])
        & detail["Target"].isin(TARGETS)
    ].copy()
    displayed = displayed.sort_values(["Target", "Origin", "Horizon"])
    return measured, displayed


def _draw_protocol_strip(ax, protocol):
    ax.set_ylim(0, 1)
    for stage in protocol:
        color = CYCLING_COLOR if stage["Stage"] == "Cycling" else REST_COLOR
        text_color = "#40515B" if stage["Stage"] == "Cycling" else "#756A25"
        ax.axvspan(stage["Start_min"], stage["End_min"], color=color, linewidth=0)
        ax.text(
            (stage["Start_min"] + stage["End_min"]) / 2,
            0.5,
            stage["Stage"],
            ha="center",
            va="center",
            fontsize=6.3,
            fontweight="bold",
            color=text_color,
        )
    ax.set_xlim(6, 55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(
        -0.080,
        0.88,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        5.75,
        0.5,
        "Day 4\nprotocol",
        ha="right",
        va="center",
        fontsize=6.2,
        color="#4B555C",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_window_schematic(ax, input_length=5, horizon=5):
    origins = (10, 11, 12)
    y_positions = (2, 1, 0)
    input_color = "#D8DDE3"
    forecast_color = "#8EB8DA"
    for origin, y in zip(origins, y_positions):
        input_start = origin - input_length + 1
        forecast_start = origin + 1
        ax.add_patch(
            Rectangle(
                (input_start - 0.45, y - 0.31),
                input_length - 0.10,
                0.62,
                facecolor=input_color,
                edgecolor="#838B92",
                linewidth=0.55,
            )
        )
        ax.add_patch(
            Rectangle(
                (forecast_start - 0.45, y - 0.31),
                horizon - 0.10,
                0.62,
                facecolor=forecast_color,
                edgecolor="#3977A8",
                linewidth=0.65,
            )
        )
        ax.axvline(origin + 0.5, ymin=(y + 0.19) / 3, ymax=(y + 0.81) / 3,
                   color="#2E6C9E", linewidth=0.75)
        ax.text(input_start + 1.8, y, "5-min input", ha="center", va="center", fontsize=5.7)
        ax.text(forecast_start + 1.8, y, "5-min forecast", ha="center", va="center",
                fontsize=5.7, color="#174E78", fontweight="bold")
        ax.text(input_start - 0.75, y, f"Origin {origin}", ha="right", va="center", fontsize=5.8)
        ax.plot(origin, y + 0.36, marker="v", markersize=3.3, color=ORIGIN_MARKER_COLOR,
                clip_on=False)

    ax.annotate(
        "slide forward by 1 min",
        xy=(16.0, 0.03),
        xytext=(16.0, 1.96),
        arrowprops={"arrowstyle": "-|>", "color": ORIGIN_MARKER_COLOR, "lw": 0.8},
        ha="center",
        va="center",
        fontsize=5.8,
        color=ORIGIN_MARKER_COLOR,
    )
    ax.set_xlim(5.2, 18.4)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks(np.arange(6, 19, 2))
    ax.set_xlabel("Exercise time (min)", labelpad=1)
    ax.set_yticks([])
    ax.set_ylabel("b", rotation=0, fontweight="bold", fontsize=8, labelpad=8)
    ax.set_title(
        "Sequential windows preserve temporal order: history → forecast → update with new measurement",
        fontsize=7.2,
        loc="left",
        pad=3,
    )
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#8E969C")


def rolling_window_forecast_figure(
    measured,
    displayed,
    out_base,
    protocol,
    displayed_origins,
):
    """Show the sliding mechanism and independent five-minute predictions."""
    origin_min = min(displayed_origins)
    origin_max = max(displayed_origins)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "forecast_origin", ["#A8CBE7", "#1D5F98"]
    )
    norm = mpl.colors.Normalize(vmin=origin_min, vmax=origin_max)

    with plt.rc_context(_PUB_STYLE):
        fig = plt.figure(figsize=(8.0, 6.0))
        grid = fig.add_gridspec(
            5,
            2,
            height_ratios=[0.30, 0.92, 0.16, 1.40, 1.40],
            hspace=0.40,
            wspace=0.23,
        )
        protocol_ax = fig.add_subplot(grid[0, :])
        schematic_ax = fig.add_subplot(grid[1, :])
        legend_ax = fig.add_subplot(grid[2, :])
        signal_axes = [
            fig.add_subplot(grid[3, 0]),
            fig.add_subplot(grid[3, 1]),
            fig.add_subplot(grid[4, 0]),
            fig.add_subplot(grid[4, 1]),
        ]

        _draw_protocol_strip(protocol_ax, protocol)
        _draw_window_schematic(schematic_ax)

        minute = measured["Minute"].to_numpy(float)
        all_origins = sorted(displayed["Origin"].dropna().astype(int).unique())
        for panel_index, (ax, target) in enumerate(zip(signal_axes, TARGETS)):
            _shade_protocol(ax, protocol, alpha=0.20)
            raw = measured[f"{target}_raw"].to_numpy(float)
            actual = measured[target].to_numpy(float)
            ax.plot(minute, raw, color=RAW_COLOR, linewidth=0.55, alpha=0.70, zorder=1)
            ax.plot(minute, actual, color=MEASURED_COLOR, linewidth=1.15, zorder=3)

            # Every one-minute rolling origin is retained as a separate faint
            # trajectory; selected five-minute origins are then highlighted.
            for origin in all_origins:
                segment = displayed[
                    (displayed["Target"] == target)
                    & (displayed["Origin"].astype(int) == int(origin))
                ].sort_values("Horizon")
                if segment.empty:
                    continue
                origin_match = measured[measured["Minute"] == origin]
                if origin_match.empty:
                    continue
                x = np.r_[float(origin), segment["Predicted_minute"].to_numpy(float)]
                y = np.r_[float(origin_match[target].iloc[0]), segment["Predicted"].to_numpy(float)]
                ax.plot(x, y, color="#6FA6CC", linewidth=0.55, alpha=0.16, zorder=2)

            for origin in displayed_origins:
                segment = displayed[
                    (displayed["Target"] == target)
                    & (displayed["Origin"].astype(int) == int(origin))
                ].sort_values("Horizon")
                if segment.empty:
                    continue
                origin_match = measured[measured["Minute"] == origin]
                if origin_match.empty:
                    continue
                x = np.r_[float(origin), segment["Predicted_minute"].to_numpy(float)]
                y = np.r_[float(origin_match[target].iloc[0]), segment["Predicted"].to_numpy(float)]
                sd = np.r_[0.0, segment["Ensemble_SD"].to_numpy(float)]
                color = cmap(norm(origin))
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.07, linewidth=0, zorder=2)
                ax.plot(x, y, color=color, linewidth=1.05, marker="o", markersize=1.9,
                        alpha=0.95, zorder=4)
                ax.scatter(
                    [origin],
                    [origin_match[target].iloc[0]],
                    marker="v",
                    s=12,
                    color=ORIGIN_MARKER_COLOR,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=5,
                )

            ax.axvline(10, color="#666666", linestyle="--", linewidth=0.65, zorder=2)
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(6, 55.5)
            ax.set_xticks(np.arange(10, 56, 10))
            ax.set_title(TARGET_LABELS[target], fontsize=8, fontweight="bold", loc="left", pad=2)
            ax.set_ylabel("Processed signal value")
            if panel_index >= 2:
                ax.set_xlabel("Exercise time (min)")
            else:
                ax.tick_params(axis="x", labelbottom=False)

        signal_axes[0].text(
            -0.16,
            1.05,
            "c",
            transform=signal_axes[0].transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        legend_handles = [
            Line2D([0], [0], color=RAW_COLOR, linewidth=1.0, label="Raw measurement"),
            Line2D([0], [0], color=MEASURED_COLOR, linewidth=1.3,
                   label="Causally denoised measurement"),
            Line2D([0], [0], color="#6FA6CC", linewidth=1.0, alpha=0.35,
                   label="All 1-min rolling origins"),
            Line2D([0], [0], color="#377EAE", linewidth=1.2, marker="o", markersize=3,
                   label="Every fifth origin highlighted"),
        ]
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=4,
            fontsize=6.1,
        )
        legend_ax.set_axis_off()

        colorbar_ax = fig.add_axes([0.785, 0.935, 0.16, 0.015])
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_ticks([origin_min, (origin_min + origin_max) / 2, origin_max])
        colorbar.set_label("Highlighted forecast origin (min)", fontsize=5.5, labelpad=1)
        colorbar.ax.tick_params(labelsize=5.2, length=2, pad=1)

        fig.suptitle(
            "Rolling-window forecasting: each origin generates an independent 5-min trajectory",
            x=0.08,
            y=0.984,
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )
        fig.text(
            0.01,
            0.008,
            "All one-minute rolling origins are shown faintly; every fifth origin is highlighted for readability.",
            ha="left",
            va="bottom",
            fontsize=5.3,
            color="#555555",
        )
        fig.text(
            0.99,
            0.008,
            "Draft protocol boundaries; confirm against the experiment log",
            ha="right",
            va="bottom",
            fontsize=5.2,
            color="#8A5A42",
        )
        fig.subplots_adjust(left=0.09, right=0.97, top=0.915, bottom=0.075)
        _save_figure_bundle(fig, Path(out_base))
        plt.close(fig)


def rolling_window_schedule(start_origin=10, end_origin=50, input_length=5, horizon=5):
    """Create a complete machine-readable description of the displayed windows."""
    rows = []
    for origin in range(start_origin, end_origin + 1):
        rows.append(
            {
                "Origin_min": origin,
                "Input_start_min": origin - input_length + 1,
                "Input_end_min": origin,
                "Forecast_start_min": origin + 1,
                "Forecast_end_min": origin + horizon,
            }
        )
    return pd.DataFrame(rows)
