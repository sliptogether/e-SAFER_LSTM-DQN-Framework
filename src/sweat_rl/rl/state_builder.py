import numpy as np
import pandas as pd
from sweat_rl.metrics import slope

def build_rl_state(arrays, day, features, origins, predictions, observed_slope=True):
    rows = []
    mat = arrays[day]["features"]
    hr = arrays[day]["hr"]
    m0 = int(arrays[day]["minutes"][0])

    for origin in origins:
        oi = origin-m0
        row = {"Origin_min":origin, "HR_current":float(hr[oi])}

        for j, f in enumerate(features):
            cur = float(mat[oi, j])
            pred = np.asarray(predictions[f][origin]["trajectory_mean"], float)
            sd = np.asarray(predictions[f][origin]["trajectory_sd"], float)
            t = np.arange(1, len(pred)+1, dtype=float)
            horizon=len(pred)

            row[f"{f}_current"] = cur
            row[f"{f}_pred_end_delta{horizon}"] = float(pred[-1]-cur)
            row[f"{f}_pred_mean_delta{horizon}"] = float(pred.mean()-cur)
            row[f"{f}_pred_slope{horizon}"] = float(np.polyfit(t, pred, 1)[0])
            row[f"{f}_endpoint_sd"] = float(sd[-1])

            if observed_slope:
                row[f"{f}_observed_cumulative_slope"] = slope(mat[:oi+1, j])

        rows.append(row)

    return pd.DataFrame(rows)
