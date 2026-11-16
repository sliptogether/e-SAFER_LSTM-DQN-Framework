import argparse
import numpy as np

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.data import load_processed_all
from sweat_rl.training.lstm_pipeline import arrays_from_days
from sweat_rl.rl.state_builder import build_rl_state
from sweat_rl.utils import load_json, ensure_dir

def main(config):
    cfg=load_config(config)
    features=cfg["data"]["features"]
    days=load_processed_all(project_path(cfg,cfg["paths"]["output_dir"],"processed"))
    arrays=arrays_from_days(days,features)
    evaluation=project_path(cfg,cfg["paths"]["output_dir"],"evaluation")
    use_rolling=bool(cfg["rl_state"].get("use_rolling_predictions",False))
    if use_rolling:
        prediction_path=evaluation/"rolling_ensemble_predictions.json"
        start=int(cfg["rl_state"].get("decision_start_minute",10))
        end=int(cfg["rl_state"].get("decision_end_minute",55))
        origins=list(range(start,end+1))
    else:
        prediction_path=evaluation/"ensemble_predictions.json"
        origins=cfg["data"]["forecast_origins"]
    raw=load_json(prediction_path)

    pred={}
    for f in features:
        pred[f]={}
        for o,v in raw[f].items():
            pred[f][int(o)]={
                "trajectory_mean":np.asarray(v["trajectory_mean"],float),
                "trajectory_sd":np.asarray(v["trajectory_sd"],float)
            }

    df=build_rl_state(
        arrays,4,features,origins,pred,
        bool(cfg["rl_state"]["include_observed_cumulative_slope"])
    )
    out=ensure_dir(project_path(cfg,cfg["paths"]["output_dir"],"rl"))
    df.to_csv(out/"rl_state.csv",index=False)
    print(df.to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/denoised_5min.yaml")
    main(p.parse_args().config)
