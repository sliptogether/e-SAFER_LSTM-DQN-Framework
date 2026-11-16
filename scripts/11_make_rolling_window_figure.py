"""Create a rolling-window schematic with independent Day 4 forecasts."""

import argparse

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.rolling_window_figure import (
    rolling_window_forecast_figure,
    rolling_window_schedule,
    rolling_window_source_table,
)


# Draft intervals interpreted from the supplied Day 4 timeline image.
DRAFT_PROTOCOL = [
    {"Start_min": 6, "End_min": 20, "Stage": "Cycling"},
    {"Start_min": 20, "End_min": 24, "Stage": "Rest"},
    {"Start_min": 24, "End_min": 42, "Stage": "Cycling"},
    {"Start_min": 42, "End_min": 47, "Stage": "Rest"},
    {"Start_min": 47, "End_min": 55, "Stage": "Cycling"},
]


def main(config):
    cfg = load_config(config)
    output_root = project_path(cfg, cfg["paths"]["output_dir"])
    processed_csv = output_root / "processed" / "Day4.csv"
    detail_csv = output_root / "evaluation" / "day4_rolling_forecasts.csv"
    figure_dir = ensure_dir(output_root / "prediction_decision_figures")
    source_dir = ensure_dir(figure_dir / "source_data")
    out_base = figure_dir / "Fig20_rolling_window_independent_forecasts_DRAFT"

    displayed_origins = list(range(10, 51, 5))
    all_origins = list(range(10, 51))
    measured, displayed = rolling_window_source_table(
        processed_csv,
        detail_csv,
        all_origins,
    )
    displayed.to_csv(
        source_dir / "Fig20_all_independent_forecasts_10to50_DRAFT.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_window_schedule().to_csv(
        source_dir / "Fig20_all_rolling_windows_10to50.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_window_forecast_figure(
        measured,
        displayed,
        out_base,
        DRAFT_PROTOCOL,
        displayed_origins,
    )
    print(out_base)
    print("NOTE: the model advances every minute; the main figure displays every fifth origin.")
    print("WARNING: verify the draft exercise-stage boundaries against the experiment log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
