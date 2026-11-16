"""Prediction-to-decision project schematic for the sweat RL workflow."""

from pathlib import Path

import numpy as np

from sweat_rl.visualization.figures import _PUB_STYLE, _save_figure_bundle

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#263238"
MUTED = "#60727B"
PREDICTION_BLUE = "#3478B5"
SENSOR_TEAL = "#3A9D8F"
PREPROCESS_BLUE = "#74A9CF"
FORECAST_PURPLE = "#7867A7"
STATE_GOLD = "#C28A24"
DQN_CORAL = "#B85D50"
ACTION_COLORS = {
    "Reduce": "#3775BA",
    "Maintain": "#858585",
    "Increase": "#E28E2C",
}


def _rounded_box(ax, x, y, w, h, face, edge, radius=0.018, lw=1.0, zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, color=INK, style="-", lw=1.35, rad=0.0, zorder=3):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _label(ax, x, y, text, size=7, weight="normal", color=INK, ha="center", va="center"):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=5,
    )


def _sensor_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#EAF6F3", SENSOR_TEAL)
    _label(ax, x + w / 2, y + h - 0.035, "Multimodal sensing", 7.5, "bold")
    channel_labels = ["Na+", "K+", "Ca2+", "TEMP", "FR", "HR"]
    channel_colors = ["#4C8BC5", "#55A7A0", "#8B78B4", "#D98B4B", "#C85A55", "#687A86"]
    for i, (name, color) in enumerate(zip(channel_labels, channel_colors)):
        row, col = divmod(i, 3)
        cx = x + 0.031 + col * (w - 0.06) / 2
        cy = y + h - 0.085 - row * 0.052
        ax.add_patch(Circle((cx, cy), 0.009, facecolor=color, edgecolor="white", linewidth=0.5, zorder=4))
        _label(ax, cx + 0.014, cy, name, 6.1, ha="left")
    _label(ax, x + w / 2, y + 0.026, "1-min aligned signals", 5.8, color=MUTED)


def _preprocess_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#EDF4FA", PREPROCESS_BLUE)
    _label(ax, x + w / 2, y + h - 0.035, "Causal preprocessing", 7.5, "bold")
    xx = np.linspace(x + 0.025, x + w - 0.025, 45)
    phase = np.linspace(0, 2.7 * np.pi, len(xx))
    raw = y + 0.102 + 0.022 * np.sin(phase) + 0.010 * np.sin(4.6 * phase)
    smooth = y + 0.102 + 0.020 * np.sin(phase)
    ax.plot(xx, raw, color="#9BAAB2", linewidth=0.7, alpha=0.75, zorder=4)
    ax.plot(xx, smooth, color=PREDICTION_BLUE, linewidth=1.5, zorder=5)
    _label(ax, x + w / 2, y + 0.058, "Holt denoising", 6.1, "bold", color=PREDICTION_BLUE)
    _label(ax, x + w / 2, y + 0.027, "Raw channels retained", 5.7, color=MUTED)


def _lstm_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#E9F1F8", PREDICTION_BLUE)
    _label(ax, x + w / 2, y + h - 0.035, "LSTM ensemble", 7.5, "bold")
    cell_y = y + 0.088
    centers = np.linspace(x + 0.04, x + w - 0.04, 4)
    for i, cx in enumerate(centers):
        _rounded_box(ax, cx - 0.014, cell_y - 0.019, 0.028, 0.038, "white", PREDICTION_BLUE, radius=0.006, lw=0.8, zorder=4)
        if i < len(centers) - 1:
            _arrow(ax, (cx + 0.016, cell_y), (centers[i + 1] - 0.016, cell_y), PREDICTION_BLUE, lw=0.8, zorder=4)
    _label(ax, x + w / 2, y + 0.052, "Past 5 min → future 5 min", 6.0, "bold", color=PREDICTION_BLUE)
    _label(ax, x + w / 2, y + 0.026, "5 seeds + uncertainty", 5.7, color=MUTED)


def _forecast_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#F1EEF8", FORECAST_PURPLE)
    _label(ax, x + w / 2, y + h - 0.035, "Continuous rolling forecast", 7.5, "bold")
    xx = np.linspace(x + 0.025, x + w - 0.025, 40)
    phase = np.linspace(0, 2.2 * np.pi, len(xx))
    actual = y + 0.095 + 0.029 * np.sin(phase) + 0.008 * np.cos(2.2 * phase)
    predicted = y + 0.096 + 0.026 * np.sin(phase - 0.18)
    sd = 0.009 + 0.002 * np.cos(phase)
    ax.fill_between(xx, predicted - sd, predicted + sd, color="#A99BC8", alpha=0.28, linewidth=0, zorder=3)
    ax.plot(xx, actual, color=INK, linewidth=1.05, zorder=4)
    ax.plot(xx, predicted, color=FORECAST_PURPLE, linewidth=1.45, zorder=5)
    _label(ax, x + w / 2, y + 0.047, "Na+ · K+ · Ca2+ · TEMP · FR", 5.8, "bold", color=FORECAST_PURPLE)
    _label(ax, x + w / 2, y + 0.022, "Minute 10–55", 5.7, color=MUTED)


def _state_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#FFF3D8", STATE_GOLD)
    _label(ax, x + w / 2, y + h - 0.035, "Prediction-derived RL state", 7.3, "bold")
    items = [
        "Current Na/K/Ca/TEMP/FR",
        "Predicted Δ · mean · slope",
        "Ensemble uncertainty",
        "Observed cumulative trend",
    ]
    for i, item in enumerate(items):
        yy = y + h - 0.074 - i * 0.031
        ax.add_patch(Circle((x + 0.024, yy), 0.0045, facecolor=STATE_GOLD, edgecolor="none", zorder=4))
        _label(ax, x + 0.035, yy, item, 5.75, ha="left")


def _dqn_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#F8EAE7", DQN_CORAL)
    _label(ax, x + w / 2, y + h - 0.035, "Surrogate DQN policy", 7.3, "bold")
    layer_x = [x + 0.035, x + w / 2, x + w - 0.035]
    node_sets = [3, 4, 3]
    layer_nodes = []
    for lx, count in zip(layer_x, node_sets):
        ys = np.linspace(y + 0.045, y + h - 0.074, count)
        nodes = [(lx, yy) for yy in ys]
        layer_nodes.append(nodes)
    for left, right in zip(layer_nodes[:-1], layer_nodes[1:]):
        for p1 in left:
            for p2 in right:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#D7A8A1", linewidth=0.4, zorder=3)
    for nodes in layer_nodes:
        for cx, cy in nodes:
            ax.add_patch(Circle((cx, cy), 0.006, facecolor="white", edgecolor=DQN_CORAL, linewidth=0.8, zorder=4))


def _action_block(ax, x, y, w, h):
    _rounded_box(ax, x, y, w, h, "#F5F6F6", "#84939B")
    _label(ax, x + w / 2, y + h - 0.034, "Action recommendation", 7.3, "bold")
    for i, name in enumerate(("Reduce", "Maintain", "Increase")):
        yy = y + h - 0.075 - i * 0.041
        _rounded_box(ax, x + 0.021, yy - 0.014, w - 0.042, 0.028, ACTION_COLORS[name], ACTION_COLORS[name], radius=0.012, lw=0.6, zorder=4)
        _label(ax, x + w / 2, yy, name, 5.9, "bold", color="white")


def project_prediction_decision_schematic(out_base):
    """Draw the end-to-end prediction and surrogate-decision architecture."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        _rounded_box(ax, 0.018, 0.515, 0.964, 0.415, "#F8FBFD", "#D8E3EA", radius=0.022, lw=0.8, zorder=0)
        _rounded_box(ax, 0.268, 0.075, 0.714, 0.355, "#FFFBF5", "#E9DDCA", radius=0.022, lw=0.8, zorder=0)

        _label(ax, 0.035, 0.905, "a", 8.5, "bold", ha="left")
        _label(ax, 0.057, 0.905, "Prediction layer", 8.5, "bold", ha="left")
        _label(ax, 0.287, 0.405, "b", 8.5, "bold", ha="left")
        _label(ax, 0.309, 0.405, "Decision layer", 8.5, "bold", ha="left")

        top_y, top_h = 0.61, 0.225
        sensor = (0.045, top_y, 0.165, top_h)
        preprocess = (0.255, top_y, 0.165, top_h)
        lstm = (0.465, top_y, 0.175, top_h)
        forecast = (0.685, top_y, 0.255, top_h)

        _sensor_block(ax, *sensor)
        _preprocess_block(ax, *preprocess)
        _lstm_block(ax, *lstm)
        _forecast_block(ax, *forecast)

        center_y = top_y + top_h / 2
        _arrow(ax, (sensor[0] + sensor[2] + 0.006, center_y), (preprocess[0] - 0.006, center_y))
        _arrow(ax, (preprocess[0] + preprocess[2] + 0.006, center_y), (lstm[0] - 0.006, center_y))
        _arrow(ax, (lstm[0] + lstm[2] + 0.006, center_y), (forecast[0] - 0.006, center_y))

        _rounded_box(ax, 0.481, 0.855, 0.143, 0.042, "white", "#A6B7C1", radius=0.015, lw=0.7, zorder=2)
        _label(ax, 0.5525, 0.876, "Day 1–3: CV + training", 5.7, "bold", color=MUTED)
        _arrow(ax, (0.5525, 0.853), (0.5525, top_y + top_h + 0.005), color="#8EA1AC", lw=0.8)

        _rounded_box(ax, 0.724, 0.545, 0.177, 0.042, "white", "#A6B7C1", radius=0.015, lw=0.7, zorder=2)
        _label(ax, 0.8125, 0.566, "Day 4: independent rolling test", 5.7, "bold", color=MUTED)

        state = (0.30, 0.155, 0.225, 0.195)
        dqn = (0.57, 0.155, 0.16, 0.195)
        actions = (0.775, 0.155, 0.17, 0.195)
        _state_block(ax, *state)
        _dqn_block(ax, *dqn)
        _action_block(ax, *actions)

        _arrow(
            ax,
            (forecast[0] + forecast[2] / 2, top_y - 0.004),
            (state[0] + state[2] / 2, state[1] + state[3] + 0.004),
            color=FORECAST_PURPLE,
            lw=1.5,
            rad=0.14,
        )
        _label(ax, 0.65, 0.465, "Forecast features", 5.8, "bold", color=FORECAST_PURPLE)
        _arrow(ax, (state[0] + state[2] + 0.006, 0.252), (dqn[0] - 0.006, 0.252), color=STATE_GOLD)
        _arrow(ax, (dqn[0] + dqn[2] + 0.006, 0.252), (actions[0] - 0.006, 0.252), color=DQN_CORAL)

        _arrow(
            ax,
            (actions[0] + actions[2] / 2, actions[1] - 0.006),
            (state[0] + state[2] / 2, state[1] - 0.006),
            color="#A66A62",
            style=(0, (3, 2)),
            lw=1.0,
            rad=-0.22,
            zorder=2,
        )
        _label(ax, 0.64, 0.088, "Surrogate action effect → next state", 5.8, "bold", color="#8E5B54")

        _rounded_box(ax, 0.035, 0.112, 0.205, 0.255, "#F7F8F8", "#C8D1D5", radius=0.018, lw=0.7, zorder=1)
        _label(ax, 0.1375, 0.333, "Model outputs", 7.2, "bold")
        outputs = [
            "Rolling trajectories",
            "R² · MAE · RMSE",
            "Residuals + uncertainty",
            "Decision timeline",
            "Policy scores + reward",
        ]
        for i, item in enumerate(outputs):
            yy = 0.293 - i * 0.039
            ax.add_patch(Circle((0.065, yy), 0.0045, facecolor=PREDICTION_BLUE if i < 3 else DQN_CORAL, edgecolor="none", zorder=4))
            _label(ax, 0.078, yy, item, 5.8, ha="left")

        _label(
            ax,
            0.5,
            0.025,
            "Simulation-only decision layer — human closed-loop validation requires intervention–response data",
            6.2,
            "bold",
            color="#9A4F45",
        )

        fig.suptitle(
            "Wearable sweat sensing → rolling prediction → surrogate decision support",
            fontsize=10,
            fontweight="bold",
            color=INK,
            y=0.985,
        )
        fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.965))
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def compact_project_schematic(out_base):
    """Draw a compact double-column workflow figure for the main manuscript."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 2.55))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Restrained background bands distinguish prediction from decision
        # without adding another panel or a redundant legend.
        _rounded_box(ax, 0.018, 0.235, 0.635, 0.54, "#F8FBFD", "#D9E5EB", radius=0.022, lw=0.75, zorder=0)
        _rounded_box(ax, 0.67, 0.235, 0.312, 0.54, "#FFFAF4", "#EADCC7", radius=0.022, lw=0.75, zorder=0)
        _label(ax, 0.035, 0.745, "Prediction", 6.2, "bold", color=PREDICTION_BLUE, ha="left")
        _label(ax, 0.687, 0.745, "Decision support", 6.2, "bold", color=DQN_CORAL, ha="left")

        y, h = 0.325, 0.33
        blocks = [
            (0.035, 0.125, "#EAF6F3", SENSOR_TEAL),
            (0.195, 0.125, "#EDF4FA", PREPROCESS_BLUE),
            (0.355, 0.125, "#E9F1F8", PREDICTION_BLUE),
            (0.515, 0.125, "#F1EEF8", FORECAST_PURPLE),
            (0.687, 0.125, "#FFF3D8", STATE_GOLD),
            (0.847, 0.125, "#F8EAE7", DQN_CORAL),
        ]
        for x, w, face, edge in blocks:
            _rounded_box(ax, x, y, w, h, face, edge, radius=0.016, lw=0.95, zorder=2)

        centers = [x + w / 2 for x, w, _, _ in blocks]
        titles = [
            "Wearable\nsensing",
            "Causal\npreprocessing",
            "LSTM\nensemble",
            "Rolling\nforecast",
            "Prediction-\nderived state",
            "Surrogate\nDQN policy",
        ]
        for center, title in zip(centers, titles):
            _label(ax, center, y + h - 0.055, title, 6.8, "bold")

        # Multimodal sensing icon and labels.
        signal_colors = ["#4C8BC5", "#55A7A0", "#8B78B4", "#D98B4B", "#C85A55", "#687A86"]
        signal_names = ["Na+", "K+", "Ca2+", "TEMP", "FR", "HR"]
        for i, (name, color) in enumerate(zip(signal_names, signal_colors)):
            row, col = divmod(i, 3)
            cx = blocks[0][0] + 0.022 + col * 0.039
            cy = y + 0.145 - row * 0.055
            ax.add_patch(Circle((cx, cy), 0.006, facecolor=color, edgecolor="none", zorder=4))
            _label(ax, cx + 0.009, cy, name, 4.9, ha="left")

        # Causal filtering icon.
        xx = np.linspace(blocks[1][0] + 0.016, blocks[1][0] + blocks[1][1] - 0.016, 36)
        phase = np.linspace(0, 2.4 * np.pi, len(xx))
        raw = y + 0.13 + 0.025 * np.sin(phase) + 0.010 * np.sin(4 * phase)
        smooth = y + 0.13 + 0.022 * np.sin(phase)
        ax.plot(xx, raw, color="#AAB7BE", linewidth=0.65, zorder=4)
        ax.plot(xx, smooth, color=PREDICTION_BLUE, linewidth=1.35, zorder=5)
        _label(ax, centers[1], y + 0.058, "1-min alignment + denoising", 4.9, color=MUTED)

        # Compact LSTM cell sequence.
        lstm_x = blocks[2][0]
        cell_centers = np.linspace(lstm_x + 0.025, lstm_x + blocks[2][1] - 0.025, 4)
        for i, cx in enumerate(cell_centers):
            _rounded_box(ax, cx - 0.012, y + 0.105, 0.024, 0.041, "white", PREDICTION_BLUE, radius=0.005, lw=0.75, zorder=4)
            if i < 3:
                _arrow(ax, (cx + 0.013, y + 0.1255), (cell_centers[i + 1] - 0.013, y + 0.1255), PREDICTION_BLUE, lw=0.65, zorder=4)
        _label(ax, centers[2], y + 0.058, "Past 5 min → future 5 min", 4.9, color=MUTED)

        # Forecast trajectory and uncertainty.
        forecast_x = blocks[3][0]
        xx = np.linspace(forecast_x + 0.016, forecast_x + blocks[3][1] - 0.016, 36)
        phase = np.linspace(0, 2.2 * np.pi, len(xx))
        actual = y + 0.13 + 0.025 * np.sin(phase) + 0.006 * np.cos(2 * phase)
        pred = y + 0.132 + 0.022 * np.sin(phase - 0.15)
        ax.fill_between(xx, pred - 0.008, pred + 0.008, color="#AC9ECC", alpha=0.3, linewidth=0, zorder=3)
        ax.plot(xx, actual, color=INK, linewidth=0.9, zorder=4)
        ax.plot(xx, pred, color=FORECAST_PURPLE, linewidth=1.25, zorder=5)
        _label(ax, centers[3], y + 0.058, "Minute 10–55 + uncertainty", 4.9, color=MUTED)

        # State features are intentionally summarized to remain readable at
        # final double-column size.
        state_lines = ["Current Na/K/Ca/TEMP/FR", "Predicted Δ · slope", "Uncertainty + trend"]
        for i, line in enumerate(state_lines):
            yy = y + 0.145 - i * 0.043
            ax.add_patch(Circle((blocks[4][0] + 0.017, yy), 0.004, facecolor=STATE_GOLD, edgecolor="none", zorder=4))
            _label(ax, blocks[4][0] + 0.026, yy, line, 4.8, ha="left")

        # DQN action outputs use color and direct labels; no separate legend.
        for i, name in enumerate(("Reduce", "Maintain", "Increase")):
            yy = y + 0.151 - i * 0.046
            _rounded_box(
                ax,
                blocks[5][0] + 0.016,
                yy - 0.014,
                blocks[5][1] - 0.032,
                0.028,
                ACTION_COLORS[name],
                ACTION_COLORS[name],
                radius=0.010,
                lw=0.5,
                zorder=4,
            )
            _label(ax, centers[5], yy, name, 4.9, "bold", color="white")

        # Linear arrows preserve the reading order and avoid the oversized
        # feedback arc used in the detailed supplementary schematic.
        for i in range(len(blocks) - 1):
            x1 = blocks[i][0] + blocks[i][1] + 0.004
            x2 = blocks[i + 1][0] - 0.004
            _arrow(ax, (x1, y + h / 2), (x2, y + h / 2), color=INK, lw=1.0)

        # Train/test labels are kept outside the boxes to avoid implying data
        # leakage between model selection and the held-out Day 4 evaluation.
        _rounded_box(ax, 0.361, 0.795, 0.113, 0.055, "white", "#A8B7C0", radius=0.014, lw=0.65, zorder=2)
        _label(ax, 0.4175, 0.823, "Day 1–3: CV + training", 4.9, "bold", color=MUTED)
        _arrow(ax, (0.4175, 0.792), (centers[2], y + h + 0.004), color="#8EA1AC", lw=0.7)

        _rounded_box(ax, 0.512, 0.795, 0.131, 0.055, "white", "#A8B7C0", radius=0.014, lw=0.65, zorder=2)
        _label(ax, 0.5775, 0.823, "Day 4: independent test", 4.9, "bold", color=MUTED)
        _arrow(ax, (0.5775, 0.792), (centers[3], y + h + 0.004), color="#8EA1AC", lw=0.7)

        _label(
            ax,
            0.5,
            0.145,
            "Simulation-only policy demonstration; human closed-loop validation requires intervention–response data",
            5.4,
            "bold",
            color="#9A4F45",
        )
        fig.tight_layout(pad=0.25)
        _save_figure_bundle(fig, out_base)
        plt.close(fig)


def _save_exact_figure_bundle(fig, out_base):
    """Save without tight cropping so the requested canvas ratio is exact."""
    out_base = Path(out_base)
    fig.savefig(out_base.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out_base.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        out_base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def four_by_three_project_schematic(out_base):
    """Draw an exact 4:3 manuscript schematic using a two-column snake flow."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_STYLE):
        # 7.2 × 5.4 inches = 182.9 × 137.2 mm and exactly 4:3.
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # The upper band contains the predictive model; the lower band makes
        # the simulation-only decision layer visually distinct.
        _rounded_box(ax, 0.045, 0.355, 0.91, 0.585, "#F8FBFD", "#D9E5EB", radius=0.024, lw=0.8, zorder=0)
        _rounded_box(ax, 0.045, 0.065, 0.91, 0.245, "#FFFAF4", "#EADCC7", radius=0.024, lw=0.8, zorder=0)
        _label(ax, 0.065, 0.91, "Prediction", 8.0, "bold", color=PREDICTION_BLUE, ha="left")
        _label(ax, 0.065, 0.282, "Decision support", 8.0, "bold", color=DQN_CORAL, ha="left")

        x_left, x_right = 0.095, 0.555
        width, height = 0.35, 0.19
        y_top, y_middle, y_bottom = 0.665, 0.405, 0.105
        positions = {
            "sensing": (x_left, y_top, width, height),
            "preprocess": (x_right, y_top, width, height),
            "lstm": (x_right, y_middle, width, height),
            "forecast": (x_left, y_middle, width, height),
            "state": (x_left, y_bottom, width, 0.15),
            "dqn": (x_right, y_bottom, width, 0.15),
        }
        styles = {
            "sensing": ("#EAF6F3", SENSOR_TEAL),
            "preprocess": ("#EDF4FA", PREPROCESS_BLUE),
            "lstm": ("#E9F1F8", PREDICTION_BLUE),
            "forecast": ("#F1EEF8", FORECAST_PURPLE),
            "state": ("#FFF3D8", STATE_GOLD),
            "dqn": ("#F8EAE7", DQN_CORAL),
        }
        for key, position in positions.items():
            face, edge = styles[key]
            _rounded_box(ax, *position, face, edge, radius=0.018, lw=1.0, zorder=2)

        # Numbering reinforces the snake reading order at 4:3 aspect ratio.
        sequence = ["sensing", "preprocess", "lstm", "forecast", "state", "dqn"]
        for index, key in enumerate(sequence, start=1):
            x, y, w, h = positions[key]
            color = styles[key][1]
            ax.add_patch(Circle((x + 0.026, y + h - 0.028), 0.016, facecolor=color, edgecolor="white", linewidth=0.7, zorder=6))
            ax.text(
                x + 0.026,
                y + h - 0.028,
                str(index),
                fontsize=6.0,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=7,
            )

        # 1. Wearable sensing.
        x, y, w, h = positions["sensing"]
        _label(ax, x + w / 2, y + h - 0.038, "Wearable multimodal sensing", 7.5, "bold")
        names = ["Na+", "K+", "Ca2+", "TEMP", "FR", "HR"]
        colors = ["#4C8BC5", "#55A7A0", "#8B78B4", "#D98B4B", "#C85A55", "#687A86"]
        for i, (name, color) in enumerate(zip(names, colors)):
            row, col = divmod(i, 3)
            cx = x + 0.075 + col * 0.105
            cy = y + 0.095 - row * 0.045
            ax.add_patch(Circle((cx, cy), 0.008, facecolor=color, edgecolor="white", linewidth=0.4, zorder=4))
            _label(ax, cx + 0.013, cy, name, 5.7, ha="left")
        _label(ax, x + w / 2, y + 0.018, "Minute-resolved signals", 5.5, color=MUTED)

        # 2. Causal preprocessing.
        x, y, w, h = positions["preprocess"]
        _label(ax, x + w / 2, y + h - 0.038, "Causal preprocessing", 7.5, "bold")
        xx = np.linspace(x + 0.055, x + w - 0.055, 46)
        phase = np.linspace(0, 2.7 * np.pi, len(xx))
        raw = y + 0.083 + 0.023 * np.sin(phase) + 0.009 * np.sin(4.2 * phase)
        smooth = y + 0.083 + 0.021 * np.sin(phase)
        ax.plot(xx, raw, color="#AAB7BE", linewidth=0.7, zorder=4)
        ax.plot(xx, smooth, color=PREDICTION_BLUE, linewidth=1.45, zorder=5)
        _label(ax, x + w / 2, y + 0.027, "1-min alignment · Holt denoising · raw retained", 5.5, color=MUTED)

        # 3. LSTM ensemble and training split.
        x, y, w, h = positions["lstm"]
        _label(ax, x + w / 2, y + h - 0.038, "LSTM ensemble", 7.5, "bold")
        cells = np.linspace(x + 0.09, x + w - 0.09, 5)
        for i, cx in enumerate(cells):
            _rounded_box(ax, cx - 0.018, y + 0.085, 0.036, 0.042, "white", PREDICTION_BLUE, radius=0.005, lw=0.75, zorder=4)
            if i < len(cells) - 1:
                _arrow(ax, (cx + 0.019, y + 0.106), (cells[i + 1] - 0.019, y + 0.106), PREDICTION_BLUE, lw=0.65, zorder=4)
        _label(ax, x + w / 2, y + 0.058, "Past 5 min → future 5 min · five seeds", 5.6, color=MUTED)
        _rounded_box(ax, x + 0.092, y + 0.012, w - 0.184, 0.026, "white", "#A8B7C0", radius=0.010, lw=0.6, zorder=3)
        _label(ax, x + w / 2, y + 0.025, "Day 1–3: CV + training", 5.2, "bold", color=MUTED)

        # 4. Held-out rolling forecast.
        x, y, w, h = positions["forecast"]
        _label(ax, x + w / 2, y + h - 0.038, "Continuous rolling forecast", 7.5, "bold")
        xx = np.linspace(x + 0.055, x + w - 0.055, 46)
        phase = np.linspace(0, 2.3 * np.pi, len(xx))
        actual = y + 0.108 + 0.025 * np.sin(phase) + 0.005 * np.cos(2 * phase)
        pred = y + 0.110 + 0.022 * np.sin(phase - 0.17)
        ax.fill_between(xx, pred - 0.008, pred + 0.008, color="#AC9ECC", alpha=0.3, linewidth=0, zorder=3)
        ax.plot(xx, actual, color=INK, linewidth=0.95, zorder=4)
        ax.plot(xx, pred, color=FORECAST_PURPLE, linewidth=1.35, zorder=5)
        _label(ax, x + w / 2, y + 0.058, "Minute 10–55 · prediction + uncertainty", 5.6, color=MUTED)
        _rounded_box(ax, x + 0.092, y + 0.012, w - 0.184, 0.026, "white", "#A8B7C0", radius=0.010, lw=0.6, zorder=3)
        _label(ax, x + w / 2, y + 0.025, "Day 4: independent test", 5.2, "bold", color=MUTED)

        # 5. Prediction-derived state.
        x, y, w, h = positions["state"]
        _label(ax, x + w / 2, y + h - 0.034, "Prediction-derived RL state", 7.3, "bold")
        state_items = ["Current Na/K/Ca/TEMP/FR", "Predicted Δ · mean · slope", "Uncertainty + observed trend"]
        for i, item in enumerate(state_items):
            yy = y + 0.075 - i * 0.031
            ax.add_patch(Circle((x + 0.055, yy), 0.0045, facecolor=STATE_GOLD, edgecolor="none", zorder=4))
            _label(ax, x + 0.068, yy, item, 5.6, ha="left")

        # 6. Surrogate DQN outputs.
        x, y, w, h = positions["dqn"]
        _label(ax, x + w / 2, y + h - 0.034, "Surrogate DQN policy", 7.3, "bold")
        for i, name in enumerate(("Reduce", "Maintain", "Increase")):
            yy = y + 0.073 - i * 0.031
            _rounded_box(ax, x + 0.075, yy - 0.011, w - 0.15, 0.022, ACTION_COLORS[name], ACTION_COLORS[name], radius=0.009, lw=0.5, zorder=4)
            _label(ax, x + w / 2, yy, name, 5.4, "bold", color="white")

        # Snake-flow arrows: left-to-right, down, right-to-left, down, then
        # left-to-right. The numbered nodes remove any directional ambiguity.
        _arrow(ax, (x_left + width + 0.005, y_top + height / 2), (x_right - 0.005, y_top + height / 2), color=INK, lw=1.2)
        _arrow(ax, (x_right + width / 2, y_top - 0.006), (x_right + width / 2, y_middle + height + 0.006), color=INK, lw=1.2)
        _arrow(ax, (x_right - 0.005, y_middle + height / 2), (x_left + width + 0.005, y_middle + height / 2), color=INK, lw=1.2)
        _arrow(ax, (x_left + width / 2, y_middle - 0.006), (x_left + width / 2, y_bottom + positions["state"][3] + 0.006), color=INK, lw=1.2)
        _arrow(ax, (x_left + width + 0.005, y_bottom + positions["state"][3] / 2), (x_right - 0.005, y_bottom + positions["dqn"][3] / 2), color=INK, lw=1.2)

        _label(
            ax,
            0.5,
            0.026,
            "Simulation-only policy demonstration; human closed-loop validation requires intervention–response data",
            5.8,
            "bold",
            color="#9A4F45",
        )
        _save_exact_figure_bundle(fig, out_base)
        plt.close(fig)


def lstm_dqn_network_schematic_detailed(out_base):
    """Draw the implemented LSTM-to-DQN network and surrogate training loop."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_STYLE):
        # Exact 4:3 canvas; vector exports retain editable text.
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Two visually distinct method stages plus a compact reward pathway.
        _rounded_box(
            ax, 0.035, 0.515, 0.93, 0.425,
            "#F7FBFD", "#D4E3EB", radius=0.024, lw=0.85, zorder=0,
        )
        _rounded_box(
            ax, 0.035, 0.115, 0.93, 0.345,
            "#FFFBF5", "#E8DBC7", radius=0.024, lw=0.85, zorder=0,
        )
        _label(ax, 0.055, 0.91, "a", 8.5, "bold", ha="left")
        _label(ax, 0.078, 0.91, "Five-minute LSTM forecasting", 8.5, "bold", ha="left")
        _label(ax, 0.055, 0.43, "b", 8.5, "bold", ha="left")
        _label(ax, 0.078, 0.43, "Surrogate DQN decision policy", 8.5, "bold", ha="left")

        # ------------------------------------------------------------------
        # Panel a: actual LSTM topology used by DecisionLSTM.
        # ------------------------------------------------------------------
        input_box = (0.065, 0.61, 0.17, 0.23)
        lstm_box = (0.285, 0.61, 0.22, 0.23)
        heads_box = (0.555, 0.61, 0.18, 0.23)
        ensemble_box = (0.785, 0.61, 0.145, 0.23)
        for position, face, edge in (
            (input_box, "#EAF6F3", SENSOR_TEAL),
            (lstm_box, "#E9F1F8", PREDICTION_BLUE),
            (heads_box, "#F1EEF8", FORECAST_PURPLE),
            (ensemble_box, "#EDF4FA", PREPROCESS_BLUE),
        ):
            _rounded_box(ax, *position, face, edge, radius=0.017, lw=1.0, zorder=2)

        x, y, w, h = input_box
        _label(ax, x + w / 2, y + h - 0.035, "Input window", 7.2, "bold")
        channel_colors = ["#4C8BC5", "#55A7A0", "#8B78B4", "#D98B4B", "#C85A55"]
        for row in range(5):
            for col, color in enumerate(channel_colors):
                ax.add_patch(Rectangle(
                    (x + 0.038 + col * 0.019, y + 0.072 + row * 0.020),
                    0.014, 0.014, facecolor=color, edgecolor="white",
                    linewidth=0.35, zorder=4,
                ))
        _label(ax, x + w / 2, y + 0.052, "t−4, …, t", 5.5, "bold", color=MUTED)
        _label(ax, x + w / 2, y + 0.027, "signals ± HR ± time", 5.3, color=MUTED)

        x, y, w, h = lstm_box
        _label(ax, x + w / 2, y + h - 0.035, "LSTM encoder", 7.2, "bold")
        cell_centers = np.linspace(x + 0.047, x + w - 0.047, 5)
        cell_y = y + 0.118
        for index, cx in enumerate(cell_centers):
            _rounded_box(
                ax, cx - 0.017, cell_y - 0.025, 0.034, 0.05,
                "white", PREDICTION_BLUE, radius=0.006, lw=0.8, zorder=4,
            )
            _label(ax, cx, cell_y, f"h{index + 1}", 4.8, "bold", color=PREDICTION_BLUE)
            if index < len(cell_centers) - 1:
                _arrow(
                    ax, (cx + 0.018, cell_y),
                    (cell_centers[index + 1] - 0.018, cell_y),
                    PREDICTION_BLUE, lw=0.7, zorder=4,
                )
        _label(ax, x + w / 2, y + 0.052, "last hidden state", 5.5, "bold", color=MUTED)
        _label(ax, x + w / 2, y + 0.027, "hidden size = 8 or 16", 5.3, color=MUTED)

        x, y, w, h = heads_box
        _label(ax, x + w / 2, y + h - 0.035, "Output heads", 7.2, "bold")
        _rounded_box(
            ax, x + 0.025, y + 0.105, w - 0.05, 0.052,
            "white", FORECAST_PURPLE, radius=0.011, lw=0.75, zorder=4,
        )
        _label(ax, x + w / 2, y + 0.131, "Linear(h, 5)", 5.8, "bold", color=FORECAST_PURPLE)
        _label(ax, x + w / 2, y + 0.086, "trajectory: t+1 … t+5", 5.2, color=MUTED)
        _rounded_box(
            ax, x + 0.025, y + 0.022, w - 0.05, 0.042,
            "white", "#9A8ABD", radius=0.010, lw=0.7, zorder=4,
        )
        _label(ax, x + w / 2, y + 0.043, "auxiliary Linear(h, 3)", 5.2, "bold", color="#6D5C98")

        x, y, w, h = ensemble_box
        _label(ax, x + w / 2, y + h - 0.035, "Ensemble", 7.2, "bold")
        seed_y = np.linspace(y + 0.075, y + 0.145, 5)
        for seed, yy in zip((101, 202, 303, 404, 505), seed_y):
            ax.add_patch(Circle(
                (x + 0.037, yy), 0.007, facecolor="white",
                edgecolor=PREDICTION_BLUE, linewidth=0.8, zorder=4,
            ))
            _label(ax, x + 0.054, yy, f"seed {seed}", 4.7, ha="left", color=MUTED)
        _label(ax, x + w / 2, y + 0.047, "mean trajectory", 5.3, "bold", color=PREDICTION_BLUE)
        _label(ax, x + w / 2, y + 0.025, "+ endpoint SD", 5.1, color=MUTED)

        top_center_y = input_box[1] + input_box[3] / 2
        _arrow(ax, (0.241, top_center_y), (0.279, top_center_y), color=INK, lw=1.2)
        _arrow(ax, (0.511, top_center_y), (0.549, top_center_y), color=INK, lw=1.2)
        _arrow(ax, (0.741, top_center_y), (0.779, top_center_y), color=INK, lw=1.2)
        _label(
            ax, 0.5, 0.555,
            "One model per target: Na+ · K+ · Ca2+ · TEMP · FR",
            6.1, "bold", color=PREDICTION_BLUE,
        )

        # Explicitly connect the rolling ensemble forecast to the RL state.
        # The routed line sits in the narrow inter-panel gutter.
        ax.plot([0.857, 0.857], [0.605, 0.487], color=FORECAST_PURPLE, linewidth=1.15, zorder=3)
        ax.plot([0.857, 0.192], [0.487, 0.487], color=FORECAST_PURPLE, linewidth=1.15, zorder=3)
        _arrow(ax, (0.192, 0.487), (0.192, 0.371), color=FORECAST_PURPLE, lw=1.15)
        _label(ax, 0.525, 0.497, "rolling forecast features", 5.0, "bold", color=FORECAST_PURPLE)

        # ------------------------------------------------------------------
        # Panel b: 30 policy features -> 64 -> 32 -> three Q values.
        # ------------------------------------------------------------------
        state_box = (0.065, 0.205, 0.255, 0.16)
        dqn_box = (0.385, 0.205, 0.285, 0.16)
        action_box = (0.735, 0.205, 0.195, 0.16)
        _rounded_box(ax, *state_box, "#FFF3D8", STATE_GOLD, radius=0.017, lw=1.0, zorder=2)
        _rounded_box(ax, *dqn_box, "#F8EAE7", DQN_CORAL, radius=0.017, lw=1.0, zorder=2)
        _rounded_box(ax, *action_box, "#F5F6F6", "#84939B", radius=0.017, lw=1.0, zorder=2)

        x, y, w, h = state_box
        _label(ax, x + w / 2, y + h - 0.031, r"RL state  $s_t \in \mathbb{R}^{30}$", 7.0, "bold")
        state_items = [
            "Current Na/K/Ca/TEMP/FR",
            "Predicted end/mean Δ + slope",
            "Endpoint SD + observed trend",
        ]
        for index, item in enumerate(state_items):
            yy = y + 0.083 - index * 0.027
            ax.add_patch(Circle((x + 0.027, yy), 0.004, facecolor=STATE_GOLD, edgecolor="none", zorder=4))
            _label(ax, x + 0.038, yy, item, 5.2, ha="left")

        x, y, w, h = dqn_box
        _label(ax, x + w / 2, y + h - 0.031, "Q-network", 7.0, "bold")
        layer_x = [x + 0.04, x + 0.115, x + 0.195, x + 0.255]
        layer_counts = [5, 6, 5, 3]
        layer_nodes = []
        for lx, count in zip(layer_x, layer_counts):
            ys = np.linspace(y + 0.035, y + 0.105, count)
            layer_nodes.append([(lx, yy) for yy in ys])
        for left, right in zip(layer_nodes[:-1], layer_nodes[1:]):
            for p1 in left:
                for p2 in right:
                    ax.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]],
                        color="#DAB1AA", linewidth=0.28, alpha=0.75, zorder=3,
                    )
        for nodes in layer_nodes:
            for cx, cy in nodes:
                ax.add_patch(Circle(
                    (cx, cy), 0.0045, facecolor="white",
                    edgecolor=DQN_CORAL, linewidth=0.65, zorder=4,
                ))
        for lx, text_value in zip(layer_x, ("30", "64", "32", "3 Q")):
            _label(ax, lx, y + 0.017, text_value, 4.6, "bold", color=MUTED)

        x, y, w, h = action_box
        _label(ax, x + w / 2, y + h - 0.031, r"$\arg\max_a Q(s_t,a)$", 6.8, "bold")
        for index, name in enumerate(("Reduce", "Maintain", "Increase")):
            yy = y + 0.086 - index * 0.028
            _rounded_box(
                ax, x + 0.033, yy - 0.010, w - 0.066, 0.020,
                ACTION_COLORS[name], ACTION_COLORS[name],
                radius=0.008, lw=0.45, zorder=4,
            )
            _label(ax, x + w / 2, yy, name, 5.0, "bold", color="white")

        _arrow(ax, (0.326, 0.285), (0.379, 0.285), color=INK, lw=1.2)
        _arrow(ax, (0.676, 0.285), (0.729, 0.285), color=INK, lw=1.2)

        # Reward and replay-learning route. Dashed arrows separate training
        # feedback from the solid inference path above.
        reward_box = (0.17, 0.045, 0.49, 0.095)
        _rounded_box(ax, *reward_box, "#FFF7F5", "#C67A70", radius=0.015, lw=0.85, zorder=2)
        _label(ax, 0.415, 0.116, "Surrogate reward", 6.4, "bold", color="#9A4F45")
        _label(
            ax, 0.415, 0.086,
            r"$r_t$ = HR-zone reward − future-risk − 0.10 × action switching",
            5.5, "bold", color=INK,
        )
        _label(
            ax, 0.415, 0.059,
            "Replay buffer · Bellman target · γ = 0.95 · 300 episodes",
            5.1, color=MUTED,
        )
        _arrow(
            ax, (0.833, 0.199), (0.66, 0.092),
            color="#A86359", style=(0, (3, 2)), lw=1.0, rad=-0.12,
        )
        _arrow(
            ax, (0.17, 0.092), (0.49, 0.199),
            color="#A86359", style=(0, (3, 2)), lw=1.0, rad=0.12,
        )
        _label(ax, 0.79, 0.126, "action", 4.8, "bold", color="#9A4F45")
        _label(ax, 0.315, 0.164, "TD update", 4.8, "bold", color="#9A4F45")

        _label(
            ax, 0.845, 0.065,
            "Simulation only\n(no validated human\naction–response model)",
            5.2, "bold", color="#9A4F45",
        )

        _save_exact_figure_bundle(fig, out_base)
        plt.close(fig)


def lstm_dqn_network_schematic(out_base):
    """Draw a concise, single-row LSTM-to-DQN manuscript schematic."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUB_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        _label(
            ax, 0.5, 0.845,
            "LSTM forecasting and DQN decision pipeline",
            10.5, "bold", color=INK,
        )
        _label(
            ax, 0.5, 0.792,
            "Five-minute multimodal sensing is translated into a surrogate intensity recommendation",
            6.2, color=MUTED,
        )

        y, h = 0.34, 0.34
        blocks = {
            "input": (0.035, y, 0.155, h),
            "lstm": (0.225, y, 0.165, h),
            "forecast": (0.425, y, 0.165, h),
            "state": (0.625, y, 0.155, h),
            "dqn": (0.815, y, 0.15, h),
        }
        styles = {
            "input": ("#EAF6F3", SENSOR_TEAL),
            "lstm": ("#E9F1F8", PREDICTION_BLUE),
            "forecast": ("#F1EEF8", FORECAST_PURPLE),
            "state": ("#FFF3D8", STATE_GOLD),
            "dqn": ("#F8EAE7", DQN_CORAL),
        }
        for key, position in blocks.items():
            face, edge = styles[key]
            _rounded_box(ax, *position, face, edge, radius=0.018, lw=1.0, zorder=2)

        # Input window.
        x, y0, w, h0 = blocks["input"]
        _label(ax, x + w / 2, y0 + h0 - 0.050, "Input signals", 7.0, "bold")
        colors = ["#4C8BC5", "#55A7A0", "#8B78B4", "#D98B4B", "#C85A55", "#687A86"]
        for row in range(5):
            for col, color in enumerate(colors):
                ax.add_patch(Rectangle(
                    (x + 0.027 + col * 0.0175, y0 + 0.135 + row * 0.021),
                    0.0125, 0.0145, facecolor=color, edgecolor="white",
                    linewidth=0.3, zorder=4,
                ))
        _label(ax, x + w / 2, y0 + 0.096, "past 5 min", 5.5, "bold", color=MUTED)
        _label(ax, x + w / 2, y0 + 0.054, "Na+ · K+ · Ca2+", 5.1, color=MUTED)
        _label(ax, x + w / 2, y0 + 0.030, "TEMP · FR · HR", 5.1, color=MUTED)

        # LSTM ensemble.
        x, y0, w, h0 = blocks["lstm"]
        _label(ax, x + w / 2, y0 + h0 - 0.050, "LSTM ensemble", 7.0, "bold")
        cells = np.linspace(x + 0.045, x + w - 0.045, 3)
        cy = y0 + 0.183
        for index, cx in enumerate(cells):
            _rounded_box(
                ax, cx - 0.020, cy - 0.030, 0.040, 0.060,
                "white", PREDICTION_BLUE, radius=0.007, lw=0.8, zorder=4,
            )
            if index < len(cells) - 1:
                _arrow(
                    ax, (cx + 0.021, cy), (cells[index + 1] - 0.021, cy),
                    color=PREDICTION_BLUE, lw=0.7, zorder=4,
                )
        _label(ax, x + w / 2, y0 + 0.104, "hidden = 8/16", 5.4, "bold", color=MUTED)
        _label(ax, x + w / 2, y0 + 0.057, "5 independent seeds", 5.2, color=MUTED)
        _label(ax, x + w / 2, y0 + 0.031, "one model per target", 5.0, color=MUTED)

        # Five-minute forecast.
        x, y0, w, h0 = blocks["forecast"]
        _label(ax, x + w / 2, y0 + h0 - 0.050, "5-min forecast", 7.0, "bold")
        xx = np.linspace(x + 0.027, x + w - 0.027, 35)
        phase = np.linspace(0, 1.7 * np.pi, len(xx))
        pred = y0 + 0.166 + 0.035 * np.sin(phase - 0.2) + 0.025 * np.linspace(0, 1, len(xx))
        sd = 0.013 + 0.003 * np.cos(phase)
        ax.fill_between(xx, pred - sd, pred + sd, color="#A99BC8", alpha=0.32, linewidth=0, zorder=3)
        ax.plot(xx, pred, color=FORECAST_PURPLE, linewidth=1.55, zorder=4)
        for point_x, point_y in zip(xx[::7], pred[::7]):
            ax.add_patch(Circle((point_x, point_y), 0.004, facecolor="white", edgecolor=FORECAST_PURPLE, linewidth=0.7, zorder=5))
        _label(ax, x + w / 2, y0 + 0.091, "t+1  …  t+5", 5.5, "bold", color=MUTED)
        _label(ax, x + w / 2, y0 + 0.050, "ensemble mean ± SD", 5.2, color=MUTED)

        # Compact prediction-derived state.
        x, y0, w, h0 = blocks["state"]
        _label(ax, x + w / 2, y0 + h0 - 0.050, "RL state", 7.0, "bold")
        _label(ax, x + w / 2, y0 + h0 - 0.082, r"$s_t \in \mathbb{R}^{30}$", 5.4, "bold", color=STATE_GOLD)
        items = ["Current values", "Predicted change", "Slope", "Uncertainty"]
        for index, item in enumerate(items):
            yy = y0 + 0.188 - index * 0.040
            ax.add_patch(Circle((x + 0.025, yy), 0.0045, facecolor=STATE_GOLD, edgecolor="none", zorder=4))
            _label(ax, x + 0.038, yy, item, 5.2, ha="left")

        # DQN output; architecture is stated once rather than drawing every edge.
        x, y0, w, h0 = blocks["dqn"]
        _label(ax, x + w / 2, y0 + h0 - 0.050, "DQN policy", 7.0, "bold")
        _label(ax, x + w / 2, y0 + h0 - 0.084, "30 → 64 → 32 → 3", 5.2, "bold", color=DQN_CORAL)
        for index, name in enumerate(("Reduce", "Maintain", "Increase")):
            yy = y0 + 0.185 - index * 0.052
            _rounded_box(
                ax, x + 0.022, yy - 0.017, w - 0.044, 0.034,
                ACTION_COLORS[name], ACTION_COLORS[name],
                radius=0.011, lw=0.5, zorder=4,
            )
            _label(ax, x + w / 2, yy, name, 5.3, "bold", color="white")

        # A strictly left-to-right inference path.
        center_y = 0.51
        ordered = [blocks[key] for key in ("input", "lstm", "forecast", "state", "dqn")]
        for left, right in zip(ordered[:-1], ordered[1:]):
            _arrow(
                ax,
                (left[0] + left[2] + 0.006, center_y),
                (right[0] - 0.006, center_y),
                color=INK, lw=1.25,
            )

        _label(
            ax, 0.5, 0.245,
            "DQN training: 300 episodes · reward combines HR range, forecast risk and action stability",
            5.6, color=MUTED,
        )
        _label(
            ax, 0.5, 0.195,
            "Surrogate decision demonstration — not a validated human closed-loop controller",
            5.8, "bold", color="#9A4F45",
        )

        _save_exact_figure_bundle(fig, out_base)
        plt.close(fig)
