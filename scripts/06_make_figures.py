import argparse

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.utils import ensure_dir
from sweat_rl.visualization.figures import (
    prediction_grid, performance_heatmap, rl_state_plot, reward_plot,
    lstm_loss_plot, rolling_prediction_figures
)

def main(config):
    cfg=load_config(config)
    out=ensure_dir(project_path(cfg,cfg["paths"]["output_dir"],"figures"))
    ev=project_path(cfg,cfg["paths"]["output_dir"],"evaluation")
    rl=project_path(cfg,cfg["paths"]["output_dir"],"rl")

    figure_cfg=cfg.get("figures",{})
    targets=figure_cfg.get("prediction_targets",["Na","K","Ca","FR"])
    origin=int(figure_cfg.get("prediction_origin",20))
    prediction_grid(
        ev/"day4_trajectory.csv",
        out/f"Fig_prediction_all_ions_origin{origin}",
        targets=targets,origin=origin
    )
    rolling_csv=ev/"day4_rolling_consensus.csv"
    if rolling_csv.exists():
        rolling_prediction_figures(
            rolling_csv,out,targets=targets,
            horizon=int(cfg["data"]["forecast_horizon_min"])
        )
    performance_heatmap(ev/"metrics.json",out/"Fig_performance_heatmap.png")
    rl_state_plot(rl/"rl_state.csv",out/"Fig_prediction_to_RL.png",30)

    if (rl/"dqn_rewards.csv").exists():
        reward_plot(rl/"dqn_rewards.csv",out/"Fig_DQN_reward.png")

    history=project_path(cfg,cfg["paths"]["output_dir"],"training_history")
    if history.exists():
        lstm_loss_plot(history,out/"Fig_LSTM_loss_curves")

    print(out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/denoised_5min.yaml")
    main(p.parse_args().config)
