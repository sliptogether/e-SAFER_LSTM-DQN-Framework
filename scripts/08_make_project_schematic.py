import argparse

import pandas as pd

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.visualization.project_schematic import (
    compact_project_schematic,
    four_by_three_project_schematic,
    lstm_dqn_network_schematic,
    project_prediction_decision_schematic,
)


def main(config):
    cfg = load_config(config)
    out_dir = project_path(cfg, cfg["paths"]["output_dir"], "prediction_decision_figures")
    detailed_base = out_dir / "Fig13_project_prediction_decision_schematic"
    compact_base = out_dir / "Fig14_project_framework_compact"
    four_by_three_base = out_dir / "Fig15_project_framework_4x3"
    network_base = out_dir / "Fig19_LSTM_DQN_network_schematic"
    project_prediction_decision_schematic(detailed_base)
    compact_project_schematic(compact_base)
    four_by_three_project_schematic(four_by_three_base)
    lstm_dqn_network_schematic(network_base)
    manifest_path = out_dir / "figure_manifest.csv"
    rows = [
        {
            "Figure": out_base.name,
            "PNG": f"{out_base.name}.png",
            "PDF": f"{out_base.name}.pdf",
            "SVG": f"{out_base.name}.svg",
            "TIFF": f"{out_base.name}.tiff",
        }
        for out_base in (detailed_base, compact_base, four_by_three_base, network_base)
    ]
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[~manifest.Figure.isin([row["Figure"] for row in rows])]
        manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
        manifest = manifest.sort_values("Figure")
    else:
        manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    print(detailed_base)
    print(compact_base)
    print(four_by_three_base)
    print(network_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/denoised_5min.yaml")
    main(parser.parse_args().config)
