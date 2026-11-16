import argparse
import numpy as np
import pandas as pd
import torch

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.data import load_processed_all
from sweat_rl.models.decision_lstm import DecisionLSTM
from sweat_rl.training.lstm_pipeline import arrays_from_days, ModelSpec, predict_origin
from sweat_rl.metrics import regression_metrics
from sweat_rl.rolling import aggregate_rolling_predictions
from sweat_rl.utils import ensure_dir, save_json

def load_bundle(path,H):
    # These checkpoints are generated locally by 02_train_lstm.py and include
    # NumPy scaler state in addition to tensor weights. PyTorch 2.6+ defaults
    # to weights_only=True, which rejects that trusted non-tensor metadata.
    c=torch.load(path,map_location="cpu",weights_only=False)
    s=c["spec"]
    sp=ModelSpec(
        mode=s["mode"],use_all=bool(s["use_all"]),
        hidden=int(s["hidden"]),
        summary_loss_weight=float(s["summary_loss_weight"]),
        include_hr=bool(s.get("include_hr",False)),
        include_time=bool(s.get("include_time",False)),
        trend_skip=bool(s.get("trend_skip",False)),
        persistence_blend=float(s.get("persistence_blend",0.0))
    )
    sc={k:np.asarray(v) for k,v in c["scalers"].items()}
    nin=len(sc["feature_mean"]) if sp.use_all else 1
    if sp.mode=="abs_rel":
        nin*=2
    if sp.include_hr:
        nin+=2 if sp.mode=="abs_rel" else 1
    if sp.include_time:
        nin+=1
    m=DecisionLSTM(nin,sp.hidden,H)
    m.load_state_dict(c["state_dict"])
    m.eval()
    return m,sc,sp

def main(config):
    cfg=load_config(config)
    features=cfg["data"]["features"]
    H=int(cfg["data"]["forecast_horizon_min"])
    origins=cfg["data"]["forecast_origins"]
    m0=int(cfg["data"]["minute_start"])
    mend=int(cfg["data"]["minute_end"])

    days=load_processed_all(project_path(cfg,cfg["paths"]["output_dir"],"processed"))
    arrays=arrays_from_days(days,features)
    model_dir=project_path(cfg,cfg["paths"]["output_dir"],"models")
    out=ensure_dir(project_path(cfg,cfg["paths"]["output_dir"],"evaluation"))

    rows=[]
    metrics={}
    raw_metrics={}
    pred_json={}
    rolling_rows=[]
    rolling_pred_json={}
    rolling_cfg=cfg.get("rolling_forecast",{})
    rolling_enabled=bool(rolling_cfg.get("enabled",False))
    rolling_start=int(rolling_cfg.get("start_minute",10))
    rolling_end_origin=int(rolling_cfg.get("end_origin_minute",mend-1))

    for j,f in enumerate(features):
        members=[
            load_bundle(model_dir/f/f"seed_{seed}.pt",H)
            for seed in cfg["model"]["ensemble_seeds"]
        ]
        ya,pa,da=[],[],[]
        pred_json[f]={}
        rolling_pred_json[f]={}

        for origin in origins:
            preds=[]
            for m,sc,sp in members:
                p,_=predict_origin(m,sc,arrays,4,j,sp,cfg,origin)
                preds.append(p)
            P=np.stack(preds)
            mean=P.mean(0)
            sd=P.std(0)
            pred_json[f][int(origin)]={
                "trajectory_mean":mean,
                "trajectory_sd":sd
            }

            oi=origin-m0
            h=min(H,mend-origin)
            actual=arrays[4]["features"][oi+1:oi+1+h,j]
            pe=mean[:h]
            ya.extend(actual);pa.extend(pe)

            cur=arrays[4]["features"][oi,j]
            da.append(np.sign(actual[-1]-cur)==np.sign(pe[-1]-cur))

            for k in range(H):
                minute=origin+k+1
                av=None
                er=None
                if minute<=mend:
                    av=float(arrays[4]["features"][minute-m0,j])
                    er=float(mean[k]-av)
                rows.append({
                    "Origin":origin,"Horizon":k+1,
                    "Predicted_minute":minute,"Target":f,
                    "Predicted":float(mean[k]),
                    "Ensemble_SD":float(sd[k]),
                    "Actual":av,"Error":er
                })

        if rolling_enabled:
            for origin in range(rolling_start,rolling_end_origin+1):
                predictions=[]
                for model,scalers,spec in members:
                    prediction,_=predict_origin(
                        model,scalers,arrays,4,j,spec,cfg,origin
                    )
                    predictions.append(prediction)
                stacked=np.stack(predictions)
                mean=stacked.mean(0)
                sd=stacked.std(0)
                rolling_pred_json[f][int(origin)]={
                    "trajectory_mean":mean,
                    "trajectory_sd":sd
                }
                for k in range(H):
                    minute=origin+k+1
                    if minute > mend:
                        continue
                    actual=float(arrays[4]["features"][minute-m0,j])
                    rolling_rows.append({
                        "Origin":origin,"Horizon":k+1,
                        "Predicted_minute":minute,"Target":f,
                        "Predicted":float(mean[k]),
                        "Ensemble_SD":float(sd[k]),"Actual":actual
                    })

        met=regression_metrics(np.asarray(ya),np.asarray(pa))
        met["direction_accuracy"]=float(np.mean(da))
        metrics[f]=met

        raw_column=f"{f}_raw"
        if raw_column in days[4].columns:
            raw_actual=[]
            raw_predicted=[]
            for row in rows:
                if row["Target"] == f and row["Actual"] is not None:
                    raw_actual.append(float(
                        days[4].loc[
                            days[4]["Minute"] == row["Predicted_minute"],
                            raw_column
                        ].iloc[0]
                    ))
                    raw_predicted.append(row["Predicted"])
            raw_metrics[f]=regression_metrics(raw_actual,raw_predicted)

    pd.DataFrame(rows).to_csv(out/"day4_trajectory.csv",index=False)
    save_json(metrics,out/"metrics.json")
    if raw_metrics:
        save_json(raw_metrics,out/"metrics_against_raw.json")
    save_json(pred_json,out/"ensemble_predictions.json")
    if rolling_rows:
        rolling_detail=pd.DataFrame(rolling_rows)
        rolling_detail.to_csv(out/"day4_rolling_forecasts.csv",index=False)
        start_index=rolling_start-m0
        start_values={
            feature:float(arrays[4]["features"][start_index,j])
            for j,feature in enumerate(features)
        }
        consensus=aggregate_rolling_predictions(
            rolling_detail,rolling_start,start_values
        )
        consensus.to_csv(out/"day4_rolling_consensus.csv",index=False)
        rolling_metrics={}
        for feature in features:
            subset=consensus[
                (consensus.Target==feature)&consensus.Predicted.notna()
            ]
            rolling_metrics[feature]=regression_metrics(
                subset.Actual,subset.Predicted
            )
        save_json(rolling_metrics,out/"rolling_metrics.json")
        save_json(
            rolling_pred_json,out/"rolling_ensemble_predictions.json"
        )
    print(metrics)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/denoised_5min.yaml")
    main(p.parse_args().config)
