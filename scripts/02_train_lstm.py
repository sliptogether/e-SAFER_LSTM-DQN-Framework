import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.data import load_processed_all
from sweat_rl.training.lstm_pipeline import (
    arrays_from_days, stable_thresholds, select_spec, train_one,
    select_training_epochs, select_persistence_blend
)
from sweat_rl.utils import ensure_dir, save_json

def main(config):
    cfg=load_config(config)
    features=cfg["data"]["features"]
    days=load_processed_all(project_path(cfg,cfg["paths"]["output_dir"],"processed"))
    arrays=arrays_from_days(days,features)

    thresholds=stable_thresholds(
        arrays,[1,2,3],features,
        float(cfg["selection"]["stable_threshold_multiplier"])
    )

    out=ensure_dir(project_path(cfg,cfg["paths"]["output_dir"],"models"))
    save_json(thresholds,out/"stable_thresholds.json")
    cv={}
    training_summary={}
    history_root=ensure_dir(project_path(
        cfg,cfg["paths"]["output_dir"],"training_history"
    ))
    validation_day=int(
        cfg["model"].get("early_stopping",{}).get("validation_day",3)
    )

    for j,f in enumerate(features):
        best=select_spec(arrays,features,j,cfg,thresholds[f])[0]
        sp=best["spec"]
        blend=select_persistence_blend(
            arrays,features,j,sp,cfg,thresholds[f],
            cfg["model"].get("candidate_persistence_blend",[0.0])
        )[0]
        sp.persistence_blend=float(blend["weight"])
        cv[f]={
            "score":blend["score"],
            "trajectory_r2":blend["trajectory_r2"],
            "trend_accuracy":blend["trend_acc"],
            "summary_nrmse":blend["summary_nrmse"],
            "folds":blend["folds"],
            "spec":{
                "mode":sp.mode,"use_all":sp.use_all,
                "hidden":sp.hidden,
                "summary_loss_weight":sp.summary_loss_weight,
                "include_hr":sp.include_hr,
                "include_time":sp.include_time,
                "trend_skip":sp.trend_skip,
                "persistence_blend":sp.persistence_blend
            }
        }

        td=ensure_dir(out/f)
        hd=ensure_dir(history_root/f)
        training_summary[f]={}
        for seed in cfg["model"]["ensemble_seeds"]:
            seed=int(seed)
            best_epoch,selection_history,meta=select_training_epochs(
                arrays,[1,2,3],validation_day,features,j,sp,cfg,seed
            )
            model,sc,final_history=train_one(
                arrays,[1,2,3],features,j,sp,cfg,seed,
                epochs=best_epoch,return_history=True
            )
            torch.save({
                "state_dict":model.state_dict(),
                "spec":cv[f]["spec"],
                "scalers":{k:np.asarray(v) for k,v in sc.items()},
                "selected_epoch":best_epoch,
                "epoch_selection":meta
            },td/f"seed_{seed}.pt")

            rows=[]
            for row in selection_history:
                rows.append({
                    "phase":"epoch_selection","target":f,"seed":seed,
                    "validation_day":validation_day,
                    "selected_epoch":best_epoch,**row
                })
            for row in final_history:
                rows.append({
                    "phase":"final_fit","target":f,"seed":seed,
                    "validation_day":validation_day,
                    "selected_epoch":best_epoch,
                    "epoch":row["epoch"],
                    "train_trajectory_loss":row["trajectory_loss"],
                    "train_summary_loss":row["summary_loss"],
                    "train_total_loss":row["total_loss"],
                    "validation_trajectory_loss":np.nan,
                    "validation_summary_loss":np.nan,
                    "validation_total_loss":np.nan
                })
            pd.DataFrame(rows).to_csv(hd/f"seed_{seed}_loss.csv",index=False)
            training_summary[f][str(seed)]=meta
        print(f,cv[f])

    save_json(cv,out/"cv_summary.json")
    save_json(training_summary,history_root/"epoch_selection.json")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/denoised_5min.yaml")
    main(p.parse_args().config)
