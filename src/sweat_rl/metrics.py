import numpy as np

def regression_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mse = float(np.mean((p-y)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p-y)))
    den = np.sum((y-y.mean())**2)
    r2 = float(1-np.sum((p-y)**2)/den) if den > 0 else float("nan")
    return {"r2":r2, "mae":mae, "mse":mse, "rmse":rmse}

def slope(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    t = np.arange(len(v), dtype=float)
    return float(np.polyfit(t, v, 1)[0])

def trend_label(delta, eps):
    if delta > eps:
        return 1
    if delta < -eps:
        return -1
    return 0
