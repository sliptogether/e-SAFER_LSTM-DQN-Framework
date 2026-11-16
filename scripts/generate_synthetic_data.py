"""Generate synthetic example workbooks; no experimental data are read."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def synthetic_signals(day, rng):
    """Return analytic signals with independent noise at 10-second intervals.

    Magnitudes are illustrative. These signals do not simulate a validated
    physiological model and must not be reported as experimental observations.
    """
    seconds = np.arange(10, 55 * 60 + 1, 10, dtype=float)
    minute = seconds / 60.0
    phase = 0.12 * day
    wave = np.sin(minute / 8.0 + phase)
    slow = np.sin(minute / 19.0 + phase)
    signals = {
        "Na": 35.0 + 0.12 * minute + 2.1 * wave + 0.3 * day + rng.normal(0, 0.25, len(minute)),
        "K": 4.5 + 0.012 * minute + 0.32 * wave + 0.04 * day + rng.normal(0, 0.04, len(minute)),
        "Ca": 0.7 + 0.002 * minute + 0.07 * slow + 0.01 * day + rng.normal(0, 0.01, len(minute)),
        "TEMP": 32.5 + 0.025 * minute + 0.22 * wave + 0.05 * day + rng.normal(0, 0.025, len(minute)),
        "FR": 0.8 + 0.007 * minute + 0.10 * wave + 0.02 * day + rng.normal(0, 0.015, len(minute)),
        "HR": 100.0 + 0.65 * minute + 8.0 * wave + 1.2 * day + rng.normal(0, 1.0, len(minute)),
    }
    return seconds, signals


def generate(output_dir, seed=2026, force=False):
    output = Path(output_dir).expanduser().resolve()
    names = [f"Day{day}.xlsx" for day in range(1, 5)] + ["SYNTHETIC_DATA.json"]
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing example files: "
            + ", ".join(path.name for path in existing)
            + ". Use --force to regenerate them."
        )
    # This config supplies sheet names only; raw workbooks are never accessed.
    with (ROOT / "configs" / "denoised_5min.yaml").open(encoding="utf-8") as stream:
        sheet_map = yaml.safe_load(stream)["data"]["sheet_map"]
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    for day in range(1, 5):
        seconds, signals = synthetic_signals(day, rng)
        path = output / f"Day{day}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for feature, values in signals.items():
                pd.DataFrame({"Time_s": seconds, "Synthetic_value": values}).to_excel(
                    writer, sheet_name=sheet_map[f"Day{day}"][feature], index=False,
                )
            writer.book.properties.creator = "e-SAFER synthetic example generator"
            writer.book.properties.title = f"SYNTHETIC Day {day} example; not experimental data"
            writer.book.properties.description = "Analytic signals plus seeded noise; demonstration only."
        print(path, flush=True)
    marker = {
        "synthetic": True,
        "seed": int(seed),
        "generator": "scripts/generate_synthetic_data.py",
        "files": names[:4],
        "interval_seconds": 10,
        "duration_seconds": 3300,
        "features": ["Na", "K", "Ca", "TEMP", "FR", "HR"],
        "warning": "Entirely synthetic analytic signals. Not participant data, paper results, or a validated physiological simulation.",
    }
    (output / "SYNTHETIC_DATA.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="Replace existing generated example files.")
    args = parser.parse_args()
    try:
        generate(args.output_dir, args.seed, args.force)
    except FileExistsError as error:
        parser.exit(2, f"Error: {error}\n")
