from dataclasses import dataclass, replace
import itertools
import numpy as np
import torch
import torch.optim as optim

from sweat_rl.models.decision_lstm import DecisionLSTM
from sweat_rl.metrics import regression_metrics, trend_label
from sweat_rl.utils import set_seed

@dataclass
class ModelSpec:
    mode: str
    use_all: bool
    hidden: int
    summary_loss_weight: float
    include_hr: bool = False
    include_time: bool = False
    trend_skip: bool = False
    persistence_blend: float = 0.0


def blend_with_persistence(trajectory, current, weight):
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("persistence blend weight must be in [0, 1]")
    trajectory = np.asarray(trajectory, dtype=float)
    return (1.0 - weight) * trajectory + weight * float(current)

def arrays_from_days(days, features):
    return {
        d: {
            "features": df[features].to_numpy(float),
            "hr": df["HR"].to_numpy(float),
            "minutes": df["Minute"].to_numpy(int)
        }
        for d, df in days.items()
    }

def feature_stats(arrays, train_days):
    x = np.vstack([arrays[d]["features"] for d in train_days])
    return x.mean(0), x.std(0) + 1e-8

def transform_input(
    x, mean, std, target_idx, mode, use_all,
    hr=None, hr_mean=0.0, hr_std=1.0, include_hr=False,
    minutes=None, minute_start=0, minute_end=1, include_time=False
):
    if use_all:
        az = (x-mean)/std
        rz = (x-x[-1])/std
    else:
        v = x[:, target_idx:target_idx+1]
        az = (v-mean[target_idx])/std[target_idx]
        rz = (v-v[-1])/std[target_idx]

    if mode == "rel":
        out=rz
    elif mode == "abs":
        out=az
    elif mode == "abs_rel":
        out=np.concatenate([az,rz],axis=1)
    else:
        raise ValueError(mode)

    context=[]
    if include_hr:
        if hr is None:
            raise ValueError("HR context requested but HR values were not provided")
        hz=((np.asarray(hr,float)-hr_mean)/hr_std)[:,None]
        hrz=((np.asarray(hr,float)-float(hr[-1]))/hr_std)[:,None]
        context.append(hrz if mode=="rel" else hz)
        if mode=="abs_rel":
            context.append(hrz)
    if include_time:
        if minutes is None:
            raise ValueError("Time context requested but minute values were not provided")
        scale=max(float(minute_end)-float(minute_start),1.0)
        context.append(((np.asarray(minutes,float)-minute_start)/scale)[:,None])
    if context:
        out=np.concatenate([out,*context],axis=1)
    return out

def target_from_origin(mat, origin, target_idx, cfg):
    m0 = int(cfg["data"]["minute_start"])
    H = int(cfg["data"]["forecast_horizon_min"])
    oi = origin - m0
    cur = mat[oi, target_idx]
    fut = mat[oi+1:oi+1+H, target_idx]
    if len(fut) != H:
        raise ValueError("Incomplete forecast horizon")
    delta = fut-cur
    t = np.arange(1, H+1, dtype=float)
    summary = np.array([
        delta[-1],
        delta.mean(),
        np.polyfit(t, fut, 1)[0]
    ], dtype=float)
    return delta, summary

def local_trend_delta(mat, origin, target_idx, cfg):
    m0=int(cfg["data"]["minute_start"])
    L=int(cfg["data"]["input_length_min"])
    H=int(cfg["data"]["forecast_horizon_min"])
    oi=origin-m0
    past=mat[oi-L+1:oi+1,target_idx]
    if len(past)<2:
        return np.zeros(H,dtype=float)
    slope=np.polyfit(np.arange(len(past),dtype=float),past,1)[0]
    return slope*np.arange(1,H+1,dtype=float)

def make_dataset(arrays, train_days, features, target_idx, spec, cfg):
    mean, std = feature_stats(arrays, train_days)
    hr_values=np.concatenate([arrays[d]["hr"] for d in train_days])
    hr_mean=float(hr_values.mean())
    hr_std=float(hr_values.std()+1e-8)
    L = int(cfg["data"]["input_length_min"])
    H = int(cfg["data"]["forecast_horizon_min"])
    m0 = int(cfg["data"]["minute_start"])
    start = int(cfg["data"]["prediction_start_min"])
    last_origin = int(cfg["data"]["minute_end"]) - H

    X, YT, YS = [], [], []
    for d in train_days:
        mat = arrays[d]["features"]
        for origin in range(start, last_origin+1):
            oi = origin-m0
            x = mat[oi-L+1:oi+1]
            if len(x) != L:
                continue
            yt, ys = target_from_origin(mat, origin, target_idx, cfg)
            if spec.trend_skip:
                yt=yt-local_trend_delta(mat,origin,target_idx,cfg)
            hr=arrays[d]["hr"][oi-L+1:oi+1]
            minutes=arrays[d]["minutes"][oi-L+1:oi+1]
            X.append(transform_input(
                x,mean,std,target_idx,spec.mode,spec.use_all,
                hr=hr,hr_mean=hr_mean,hr_std=hr_std,
                include_hr=spec.include_hr,minutes=minutes,
                minute_start=cfg["data"]["minute_start"],
                minute_end=cfg["data"]["minute_end"],
                include_time=spec.include_time
            ))
            YT.append(yt)
            YS.append(ys)

    X = np.asarray(X, np.float32)
    YT = np.asarray(YT, np.float32)
    YS = np.asarray(YS, np.float32)

    tm, ts = YT.mean(0), YT.std(0)+1e-8
    sm, ss = YS.mean(0), YS.std(0)+1e-8
    scalers = {
        "feature_mean":mean, "feature_std":std,
        "hr_mean":np.asarray(hr_mean),"hr_std":np.asarray(hr_std),
        "traj_mean":tm, "traj_std":ts,
        "summary_mean":sm, "summary_std":ss
    }
    return X, (YT-tm)/ts, (YS-sm)/ss, scalers

def make_dataset_with_scalers(
    arrays, days, features, target_idx, spec, cfg, scalers
):
    L = int(cfg["data"]["input_length_min"])
    H = int(cfg["data"]["forecast_horizon_min"])
    m0 = int(cfg["data"]["minute_start"])
    start = int(cfg["data"]["prediction_start_min"])
    last_origin = int(cfg["data"]["minute_end"]) - H

    X, YT, YS = [], [], []
    for d in days:
        mat = arrays[d]["features"]
        for origin in range(start, last_origin+1):
            oi = origin-m0
            x = mat[oi-L+1:oi+1]
            if len(x) != L:
                continue
            yt, ys = target_from_origin(mat, origin, target_idx, cfg)
            if spec.trend_skip:
                yt=yt-local_trend_delta(mat,origin,target_idx,cfg)
            hr=arrays[d]["hr"][oi-L+1:oi+1]
            minutes=arrays[d]["minutes"][oi-L+1:oi+1]
            X.append(transform_input(
                x, scalers["feature_mean"], scalers["feature_std"],
                target_idx,spec.mode,spec.use_all,
                hr=hr,hr_mean=float(scalers["hr_mean"]),
                hr_std=float(scalers["hr_std"]),include_hr=spec.include_hr,
                minutes=minutes,minute_start=cfg["data"]["minute_start"],
                minute_end=cfg["data"]["minute_end"],
                include_time=spec.include_time
            ))
            YT.append(yt)
            YS.append(ys)

    X = np.asarray(X, np.float32)
    YT = np.asarray(YT, np.float32)
    YS = np.asarray(YS, np.float32)
    YT = (YT-scalers["traj_mean"])/scalers["traj_std"]
    YS = (YS-scalers["summary_mean"])/scalers["summary_std"]
    return X, YT.astype(np.float32), YS.astype(np.float32)

def loss_components(pt, yt, ps, ys, summary_weight):
    trajectory = torch.mean((pt-yt)**2)
    summary = torch.mean((ps-ys)**2)
    total = trajectory + float(summary_weight)*summary
    return trajectory, summary, total

def evaluate_loss(model, xt, yt, ys, summary_weight):
    model.eval()
    with torch.no_grad():
        pt, ps = model(xt)
        trajectory, summary, total = loss_components(
            pt, yt, ps, ys, summary_weight
        )
    return {
        "trajectory_loss":float(trajectory.item()),
        "summary_loss":float(summary.item()),
        "total_loss":float(total.item())
    }

def train_one(
    arrays, train_days, features, target_idx, spec, cfg, seed,
    epochs=None, return_history=False
):
    set_seed(seed)
    X, YT, YS, scalers = make_dataset(
        arrays, train_days, features, target_idx, spec, cfg
    )

    model = DecisionLSTM(
        X.shape[-1], spec.hidden,
        int(cfg["data"]["forecast_horizon_min"])
    )
    opt = optim.Adam(
        model.parameters(),
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"])
    )
    xt = torch.tensor(X)
    yt = torch.tensor(YT)
    ys = torch.tensor(YS)
    lam = float(spec.summary_loss_weight)
    n_epochs = int(cfg["model"]["epochs"] if epochs is None else epochs)
    history = []

    for epoch in range(1,n_epochs+1):
        model.train()
        opt.zero_grad()
        pt, ps = model(xt)
        _, _, total = loss_components(pt,yt,ps,ys,lam)
        total.backward()
        opt.step()

        if return_history:
            row=evaluate_loss(model,xt,yt,ys,lam)
            row["epoch"]=epoch
            history.append(row)

    if return_history:
        return model, scalers, history
    return model, scalers

def select_training_epochs(
    arrays, all_train_days, validation_day, features, target_idx,
    spec, cfg, seed
):
    calibration_days=[d for d in all_train_days if d != validation_day]
    if not calibration_days or validation_day not in all_train_days:
        raise ValueError("validation_day must be one of all_train_days")

    set_seed(seed)
    X,YT,YS,scalers=make_dataset(
        arrays,calibration_days,features,target_idx,spec,cfg
    )
    VX,VYT,VYS=make_dataset_with_scalers(
        arrays,[validation_day],features,target_idx,spec,cfg,scalers
    )
    model=DecisionLSTM(
        X.shape[-1],spec.hidden,
        int(cfg["data"]["forecast_horizon_min"])
    )
    opt=optim.Adam(
        model.parameters(),
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"])
    )
    xt,yt,ys=torch.tensor(X),torch.tensor(YT),torch.tensor(YS)
    vxt,vyt,vys=torch.tensor(VX),torch.tensor(VYT),torch.tensor(VYS)
    lam=float(spec.summary_loss_weight)

    es=cfg["model"].get("early_stopping",{})
    enabled=bool(es.get("enabled",True))
    patience=max(1,int(es.get("patience",5)))
    min_delta=float(es.get("min_delta",1e-4))
    min_epochs=max(1,int(es.get("min_epochs",5)))
    monitor=str(es.get("monitor","total_loss"))
    if monitor not in {"trajectory_loss","summary_loss","total_loss"}:
        raise ValueError(f"Unsupported early-stopping monitor: {monitor}")
    max_epochs=int(cfg["model"]["epochs"])

    best_epoch=min(min_epochs,max_epochs)
    best_loss=float("inf")
    stale=0
    history=[]

    for epoch in range(1,max_epochs+1):
        model.train()
        opt.zero_grad()
        pt,ps=model(xt)
        _,_,total=loss_components(pt,yt,ps,ys,lam)
        total.backward()
        opt.step()

        train_metrics=evaluate_loss(model,xt,yt,ys,lam)
        validation_metrics=evaluate_loss(model,vxt,vyt,vys,lam)
        history.append({
            "epoch":epoch,
            "train_trajectory_loss":train_metrics["trajectory_loss"],
            "train_summary_loss":train_metrics["summary_loss"],
            "train_total_loss":train_metrics["total_loss"],
            "validation_trajectory_loss":validation_metrics["trajectory_loss"],
            "validation_summary_loss":validation_metrics["summary_loss"],
            "validation_total_loss":validation_metrics["total_loss"]
        })

        if epoch < min_epochs:
            continue
        current=validation_metrics[monitor]
        if current < best_loss-min_delta:
            best_loss=current
            best_epoch=epoch
            stale=0
        else:
            stale+=1
        if enabled and stale >= patience:
            break

    if not enabled:
        best_epoch=max_epochs
        best_loss=history[-1][f"validation_{monitor}"]

    return best_epoch,history,{
        "validation_day":int(validation_day),
        "stopped_epoch":int(history[-1]["epoch"]),
        "selected_epoch":int(best_epoch),
        "monitor":monitor,
        "best_validation_monitor_loss":float(best_loss)
    }

def predict_origin(model, scalers, arrays, day, target_idx, spec, cfg, origin):
    mat = arrays[day]["features"]
    m0 = int(cfg["data"]["minute_start"])
    L = int(cfg["data"]["input_length_min"])
    oi = origin-m0
    x = mat[oi-L+1:oi+1]
    cur = mat[oi, target_idx]
    hr=arrays[day]["hr"][oi-L+1:oi+1]
    minutes=arrays[day]["minutes"][oi-L+1:oi+1]

    xx = transform_input(
        x, scalers["feature_mean"], scalers["feature_std"],
        target_idx,spec.mode,spec.use_all,
        hr=hr,hr_mean=float(scalers.get("hr_mean",0.0)),
        hr_std=float(scalers.get("hr_std",1.0)),include_hr=spec.include_hr,
        minutes=minutes,minute_start=cfg["data"]["minute_start"],
        minute_end=cfg["data"]["minute_end"],
        include_time=spec.include_time
    )

    model.eval()
    with torch.no_grad():
        pt, ps = model(torch.tensor(xx[None], dtype=torch.float32))

    traj_delta = pt.numpy()[0]*scalers["traj_std"] + scalers["traj_mean"]
    if spec.trend_skip:
        traj_delta=traj_delta+local_trend_delta(
            mat,origin,target_idx,cfg
        )
    trajectory=blend_with_persistence(
        cur+traj_delta,cur,spec.persistence_blend
    )
    t=np.arange(1,len(traj_delta)+1,dtype=float)
    summary=np.asarray([
        trajectory[-1]-cur,(trajectory-cur).mean(),
        np.polyfit(t,trajectory,1)[0]
    ],dtype=float)
    return trajectory, summary

def stable_thresholds(arrays, train_days, features, multiplier):
    out = {}
    for j, f in enumerate(features):
        diffs = np.concatenate([
            np.diff(arrays[d]["features"][:, j]) for d in train_days
        ])
        out[f] = float(multiplier*np.median(np.abs(diffs)))
    return out

def evaluate_spec(arrays, heldout, train_days, features, target_idx, spec, cfg, eps, seed):
    model, scalers = train_one(
        arrays, train_days, features, target_idx, spec, cfg, seed
    )
    H = int(cfg["data"]["forecast_horizon_min"])
    m0=int(cfg["data"]["minute_start"])
    mend=int(cfg["data"]["minute_end"])
    origins=[int(x) for x in cfg["data"]["forecast_origins"]]

    yt,yp,lt,lp=[],[],[],[]
    trajectory_actual,trajectory_predicted=[],[]
    for origin in origins:
        trajectory,ps=predict_origin(
            model, scalers, arrays, heldout, target_idx, spec, cfg, origin
        )
        oi=origin-m0
        h=min(H,mend-origin)
        current=arrays[heldout]["features"][oi,target_idx]
        actual=arrays[heldout]["features"][oi+1:oi+1+h,target_idx]
        predicted=trajectory[:h]
        trajectory_actual.extend(actual)
        trajectory_predicted.extend(predicted)

        true_delta=actual-current
        t=np.arange(1,h+1,dtype=float)
        true_summary=np.asarray([
            true_delta[-1],true_delta.mean(),np.polyfit(t,actual,1)[0]
        ])
        pred_delta=predicted-current
        pred_summary=np.asarray([
            pred_delta[-1],pred_delta.mean(),np.polyfit(t,predicted,1)[0]
        ])
        yt.append(true_summary)
        yp.append(pred_summary)
        lt.append(trend_label(true_summary[0],eps))
        lp.append(trend_label(pred_summary[0],eps))

    yt = np.asarray(yt)
    yp = np.asarray(yp)
    scale = yt.std(0)+1e-8
    nrmse = float(np.sqrt(np.mean(((yp-yt)/scale)**2)))
    acc = float(np.mean(np.asarray(lt)==np.asarray(lp)))
    trajectory_r2=regression_metrics(
        trajectory_actual,trajectory_predicted
    )["r2"]
    return acc,nrmse,trajectory_r2

def select_spec(arrays, features, target_idx, cfg, eps):
    candidates = []
    for mode,use_all,hidden,lam,include_hr,include_time,trend_skip in itertools.product(
        cfg["model"]["candidate_modes"],
        cfg["model"]["candidate_use_all_features"],
        cfg["model"]["candidate_hidden"],
        cfg["model"]["candidate_summary_loss_weight"],
        cfg["model"].get("candidate_include_hr",[False]),
        cfg["model"].get("candidate_include_time",[False]),
        cfg["model"].get("candidate_trend_skip",[False])
    ):
        spec = ModelSpec(
            mode=str(mode),
            use_all=bool(use_all),
            hidden=int(hidden),
            summary_loss_weight=float(lam),
            include_hr=bool(include_hr),include_time=bool(include_time),
            trend_skip=bool(trend_skip)
        )
        folds = []
        for heldout in [1,2,3]:
            train_days = [d for d in [1,2,3] if d != heldout]
            acc,nrmse,trajectory_r2=evaluate_spec(
                arrays, heldout, train_days, features, target_idx,
                spec, cfg, eps,
                int(cfg["project"]["seed"]) + heldout + 100*target_idx
            )
            folds.append({
                "trend_acc":acc,"summary_nrmse":nrmse,
                "trajectory_r2":trajectory_r2
            })

        acc = float(np.mean([x["trend_acc"] for x in folds]))
        nrmse = float(np.mean([x["summary_nrmse"] for x in folds]))
        trajectory_r2=float(np.mean([x["trajectory_r2"] for x in folds]))
        score=(
            float(cfg["selection"].get("trajectory_r2_weight",1.0))*trajectory_r2
            +float(cfg["selection"].get("trend_accuracy_weight",0.0))*acc
            -float(cfg["selection"]["summary_nrmse_weight"])*nrmse
        )

        candidates.append({
            "spec":spec, "score":score,
            "trend_acc":acc, "summary_nrmse":nrmse,
            "trajectory_r2":trajectory_r2,
            "folds":folds
        })

    candidates.sort(
        key=lambda x:(
            x["score"],x["trajectory_r2"],x["trend_acc"],-x["summary_nrmse"]
        ),
        reverse=True
    )
    return candidates


def select_persistence_blend(
    arrays, features, target_idx, spec, cfg, eps, candidate_weights
):
    """Select a conservative LSTM/persistence blend using Day1-Day3 only."""
    base_spec=replace(spec,persistence_blend=0.0)
    fold_records=[]
    origins=[int(x) for x in cfg["data"]["forecast_origins"]]
    H=int(cfg["data"]["forecast_horizon_min"])
    m0=int(cfg["data"]["minute_start"])
    mend=int(cfg["data"]["minute_end"])

    for heldout in [1,2,3]:
        train_days=[d for d in [1,2,3] if d != heldout]
        model,scalers=train_one(
            arrays,train_days,features,target_idx,base_spec,cfg,
            int(cfg["project"]["seed"])+heldout+100*target_idx
        )
        records=[]
        for origin in origins:
            predicted,_=predict_origin(
                model,scalers,arrays,heldout,target_idx,base_spec,cfg,origin
            )
            oi=origin-m0
            h=min(H,mend-origin)
            current=float(arrays[heldout]["features"][oi,target_idx])
            actual=np.asarray(
                arrays[heldout]["features"][oi+1:oi+1+h,target_idx],
                dtype=float
            )
            records.append((current,actual,np.asarray(predicted[:h],float)))
        fold_records.append(records)

    candidates=[]
    for weight in candidate_weights:
        weight=float(weight)
        folds=[]
        for records in fold_records:
            actual_all=[]
            predicted_all=[]
            true_summaries=[]
            predicted_summaries=[]
            true_labels=[]
            predicted_labels=[]
            for current,actual,lstm_predicted in records:
                predicted=blend_with_persistence(
                    lstm_predicted,current,weight
                )
                actual_all.extend(actual)
                predicted_all.extend(predicted)
                t=np.arange(1,len(actual)+1,dtype=float)
                true_summary=np.asarray([
                    actual[-1]-current,(actual-current).mean(),
                    np.polyfit(t,actual,1)[0]
                ])
                predicted_summary=np.asarray([
                    predicted[-1]-current,(predicted-current).mean(),
                    np.polyfit(t,predicted,1)[0]
                ])
                true_summaries.append(true_summary)
                predicted_summaries.append(predicted_summary)
                true_labels.append(trend_label(true_summary[0],eps))
                predicted_labels.append(trend_label(predicted_summary[0],eps))
            yt=np.asarray(true_summaries)
            yp=np.asarray(predicted_summaries)
            scale=yt.std(0)+1e-8
            folds.append({
                "trend_acc":float(np.mean(
                    np.asarray(true_labels)==np.asarray(predicted_labels)
                )),
                "summary_nrmse":float(np.sqrt(np.mean(((yp-yt)/scale)**2))),
                "trajectory_r2":regression_metrics(
                    actual_all,predicted_all
                )["r2"]
            })
        acc=float(np.mean([x["trend_acc"] for x in folds]))
        nrmse=float(np.mean([x["summary_nrmse"] for x in folds]))
        trajectory_r2=float(np.mean([x["trajectory_r2"] for x in folds]))
        score=(
            float(cfg["selection"].get("trajectory_r2_weight",1.0))*trajectory_r2
            +float(cfg["selection"].get("trend_accuracy_weight",0.0))*acc
            -float(cfg["selection"]["summary_nrmse_weight"])*nrmse
        )
        candidates.append({
            "weight":weight,"score":score,"trajectory_r2":trajectory_r2,
            "trend_acc":acc,"summary_nrmse":nrmse,"folds":folds
        })
    candidates.sort(
        key=lambda x:(x["score"],x["trajectory_r2"],-x["weight"]),
        reverse=True
    )
    return candidates
