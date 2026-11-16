import numpy as np
import pandas as pd


def aggregate_rolling_predictions(rows, start_minute, start_values):
    """Aggregate overlapping causal forecasts into one continuous trajectory.

    Total uncertainty combines the within-origin ensemble variance and the
    between-origin variance from forecasts issued at different lead times.
    """
    df=pd.DataFrame(rows).copy()
    required={
        "Target","Origin","Horizon","Predicted_minute","Predicted",
        "Ensemble_SD","Actual"
    }
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"Missing rolling forecast columns: {sorted(missing)}")

    records=[]
    for target,value in start_values.items():
        records.append({
            "Target":target,"Minute":int(start_minute),
            "Predicted":np.nan,"Consensus_SD":np.nan,
            "Actual":float(value),"Num_forecasts":0,
            "Min_lead":np.nan,"Max_lead":np.nan
        })

    for (target,minute),group in df.groupby(
        ["Target","Predicted_minute"],sort=True
    ):
        predicted=group["Predicted"].to_numpy(float)
        ensemble_sd=group["Ensemble_SD"].to_numpy(float)
        total_variance=float(np.mean(ensemble_sd**2)+np.var(predicted))
        actual=group["Actual"].dropna()
        records.append({
            "Target":target,"Minute":int(minute),
            "Predicted":float(predicted.mean()),
            "Consensus_SD":float(np.sqrt(max(total_variance,0.0))),
            "Actual":float(actual.iloc[0]) if len(actual) else np.nan,
            "Num_forecasts":int(len(group)),
            "Min_lead":int(group["Horizon"].min()),
            "Max_lead":int(group["Horizon"].max())
        })
    return pd.DataFrame(records).sort_values(["Target","Minute"])
