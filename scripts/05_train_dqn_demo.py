import argparse
import numpy as np
import pandas as pd
import torch

import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.rl.env import SurrogateExerciseEnv
from sweat_rl.rl.dqn import infer_policy, train_dqn
from sweat_rl.utils import ensure_dir

def main(config):
    cfg=load_config(config)
    horizon=int(cfg["data"]["forecast_horizon_min"])
    last_observed_minute=int(cfg["data"]["minute_end"])
    observed_column=f"Full_{horizon}min_outcome_observed"
    df=pd.read_csv(project_path(cfg,cfg["paths"]["output_dir"],"rl","rl_state.csv"))
    cols=[c for c in df.columns if c!="Origin_min"]
    policy_cols=[c for c in cols if c!="HR_current"]
    env=SurrogateExerciseEnv(
        df[cols].to_numpy(float), cols, cfg, observation_columns=policy_cols
    )
    q,rewards=train_dqn(env,cfg)
    actions,q_values,scores=infer_policy(q,env.observations)

    out=ensure_dir(project_path(cfg,cfg["paths"]["output_dir"],"rl"))
    torch.save({
        "state_dict":q.state_dict(),"columns":policy_cols,
        "reward_state_columns":cols,
        "state_mean":env.mean,"state_std":env.std,
        "actions":["reduce","maintain","increase"],
        "warning":"surrogate demo only; not validated exercise prescription"
    },out/"dqn_demo.pt")
    pd.DataFrame({
        "Episode":range(1,len(rewards)+1),
        "Reward":rewards
    }).to_csv(out/"dqn_rewards.csv",index=False)

    labels_en=["Reduce","Maintain","Increase"]
    labels_zh=["降低强度","保持强度","提高强度"]
    decision=pd.DataFrame({
        "Minute":df["Origin_min"].astype(int),
        "Action_id":actions.astype(int),
        "Action_en":[labels_en[int(a)] for a in actions],
        "Action_zh":[labels_zh[int(a)] for a in actions],
        "Policy_score":scores.max(1),
        "Score_reduce":scores[:,0],
        "Score_maintain":scores[:,1],
        "Score_increase":scores[:,2],
        "Q_reduce":q_values[:,0],
        "Q_maintain":q_values[:,1],
        "Q_increase":q_values[:,2],
        "Surrogate_future_risk":[env.risk(s) for s in env.z],
        "HR_demo_zone":[env.hr_zone(s) for s in env.z],
        observed_column:df["Origin_min"].to_numpy(int)+horizon<=last_observed_minute,
        "Status":"simulation_only"
    })
    for feature in ["Na","K","Ca","TEMP","FR"]:
        column=next(
            (c for c in cols if c.startswith(f"{feature}_pred_end_delta")),
            None
        )
        if column is not None:
            decision[f"{feature}_predicted_{horizon}min_change"]=df[column]
    decision.to_csv(out/"decision_timeline.csv",index=False)
    segment_id=decision.Action_id.ne(decision.Action_id.shift()).cumsum()
    segments=(
        decision.assign(Segment=segment_id)
        .groupby("Segment",as_index=False)
        .agg(
            Start_min=("Minute","min"),End_min=("Minute","max"),
            Action_en=("Action_en","first"),Action_zh=("Action_zh","first"),
            Mean_policy_score=("Policy_score","mean"),
            Mean_surrogate_risk=("Surrogate_future_risk","mean"),
            **{observed_column:(observed_column,"all")}
        )
        .drop(columns="Segment")
    )
    segments.to_csv(out/"decision_segments.csv",index=False)

    print("DQN demo complete.")
    print(decision[[
        "Minute","Action_zh","Policy_score","Surrogate_future_risk",
        observed_column
    ]].to_string(index=False))
    print("WARNING: surrogate demo, not validated human closed-loop control.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/denoised_5min.yaml")
    main(p.parse_args().config)
