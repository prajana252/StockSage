"""Transparent statistical forecasters (not machine learning). All use only data up to the last observed day."""
import numpy as np
import pandas as pd

MIN_HISTORY = 56  # days needed for the weekday profile


def _future_index(s: pd.Series, horizon: int):
    return pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)


def forecast_wabl(series: pd.Series, horizon: int, alpha: float = 0.6):
    """Weekday-Adjusted Blended Level: level = alpha*mean7 + (1-alpha)*mean28, times a shrunken weekday index.
    Returns (forecast array, daily sigma, method label)."""
    s = series.astype(float)
    n = len(s)
    if n == 0:
        return np.zeros(horizon), 0.0, "no history (zero forecast)"
    if n < 28:
        return np.full(horizon, s.mean()), float(s.std(ddof=0)), "fallback: mean of short history (<28d)"
    level = alpha * s.iloc[-7:].mean() + (1 - alpha) * s.iloc[-28:].mean()
    idx, method = np.ones(7), "blended level (no weekday profile, <56d)"
    if n >= MIN_HISTORY:
        last = s.iloc[-MIN_HISTORY:]; overall = last.mean()
        if overall > 0:
            raw = last.groupby(last.index.dayofweek).mean().reindex(range(7)).fillna(overall) / overall
            idx = 0.7 * raw.to_numpy() + 0.3  # shrink toward 1 to limit noise
        method = "WABL (weekday-adjusted blended level)"
    f = level * idx[_future_index(s, horizon).dayofweek]
    return f, float(s.iloc[-28:].std(ddof=0)), method


def forecast_ma28(series: pd.Series, horizon: int):
    s = series.astype(float)
    return np.full(horizon, s.iloc[-28:].mean() if len(s) else 0.0)


def forecast_seasonal_naive(series: pd.Series, horizon: int):
    s = series.astype(float).to_numpy(); n = len(s)
    if n < 7: return np.full(horizon, s.mean() if n else 0.0)
    return np.array([s[n - 1 + h - 7 * int(np.ceil(h / 7))] for h in range(1, horizon + 1)])


def mae(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(a - p))) if len(a) else float("nan")


def wape(actual, pred) -> float:
    """Weighted absolute percentage error; NaN (not inf) when total actual demand is zero."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    tot = a.sum()
    return float(np.abs(a - p).sum() / tot) if tot > 0 else float("nan")
