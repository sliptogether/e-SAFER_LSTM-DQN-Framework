import numpy as np
import pandas as pd


def causal_denoise(values, spec):
    """Denoise a 1-D signal without using future samples.

    Supported methods are ``none``, ``ema``, ``median_ema`` and ``holt``.
    Holt filtering keeps a causal local level and trend, which reduces the lag
    introduced by a plain moving average.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("causal_denoise expects a one-dimensional signal")
    if len(values) == 0:
        return values.copy()

    spec = {} if spec is None else dict(spec)
    method = str(spec.get("method", "none")).lower()
    if method == "none":
        return values.copy()

    alpha = float(spec.get("alpha", 0.3))
    if not 0.0 < alpha <= 1.0:
        raise ValueError("denoise alpha must be in (0, 1]")

    if method == "ema":
        return (
            pd.Series(values)
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy(float)
        )

    if method == "median_ema":
        window = int(spec.get("median_window", 3))
        if window < 1 or window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer")
        return (
            pd.Series(values)
            .rolling(window=window, min_periods=1)
            .median()
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy(float)
        )

    if method == "holt":
        beta = float(spec.get("beta", 0.1))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("denoise beta must be in [0, 1]")
        out = np.empty_like(values)
        level = float(values[0])
        trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
        out[0] = level
        for i in range(1, len(values)):
            previous_level = level
            level = alpha * float(values[i]) + (1.0 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1.0 - beta) * trend
            out[i] = level
        return out

    raise ValueError(f"Unsupported denoise method: {method}")

def read_signal(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    if df.shape[1] < 2:
        raise ValueError(f"{sheet}: need time and value columns")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    arr = np.c_[x[mask].to_numpy(float), y[mask].to_numpy(float)]
    if len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{sheet}: expected finite numeric time/value pairs")
    if np.any(np.diff(arr[:, 0]) < 0):
        raise ValueError(f"{sheet}: time values must be sorted in ascending order")
    return arr

def minute_average(arr, minute):
    # Minute m = mean over ((m-1)*60, m*60].
    lo = (minute - 1) * 60.0
    hi = minute * 60.0
    mask = (arr[:, 0] > lo) & (arr[:, 0] <= hi)
    if mask.any():
        return float(arr[mask, 1].mean())

    tt = np.arange(lo + 10, hi + 1, 10, dtype=float)
    return float(np.interp(tt, arr[:, 0], arr[:, 1]).mean())

def preprocess_day(path, day_key, cfg):
    names = list(cfg["data"]["features"]) + ["HR"]
    mapping = cfg["data"]["sheet_map"][day_key]
    minutes = np.arange(
        int(cfg["data"]["minute_start"]),
        int(cfg["data"]["minute_end"]) + 1
    )
    out = {"Minute": minutes}
    for name in names:
        arr = read_signal(path, mapping[name])
        out[name] = [minute_average(arr, int(m)) for m in minutes]

    # Filtering is applied after minute aggregation. Each filtered value at
    # minute t depends only on samples at or before t. Raw feature values are
    # retained so performance against the unfiltered measurements can also be
    # reported.
    denoise = cfg["data"].get("denoise", {})
    for name, spec in denoise.items():
        if name not in out:
            raise ValueError(f"Denoise target is not available: {name}")
        raw = np.asarray(out[name], dtype=float)
        if bool(spec.get("keep_raw", True)):
            out[f"{name}_raw"] = raw.copy()
        out[name] = causal_denoise(raw, spec)
    return pd.DataFrame(out)

def load_processed_all(processed_dir):
    from pathlib import Path
    p = Path(processed_dir)
    return {d: pd.read_csv(p / f"Day{d}.csv") for d in [1,2,3,4]}
