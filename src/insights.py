"""Small analysis helpers used by the notebook to show the evidence behind the model."""
import numpy as np

from disaggregate import deconvolve, R_EV


def ev_event_shapes(X, ev_mask, jump=4.0, horizon=12):
    """Normalised decay after isolated EV starts: (x[t+k] - x[t-1]) / (x[t] - x[t-1])."""
    shapes, amps, starts = [], [], []
    for i in np.where(ev_mask)[0]:
        x = X[i]; d = np.diff(x)
        for t in np.where(d > jump)[0] + 1:
            if t < 2 or t + horizon >= len(x) or np.any(d[t:t + horizon] > 1):
                continue
            base = x[t - 1]
            shapes.append((x[t:t + horizon] - base) / (x[t] - base))
    for i in np.where(ev_mask)[0]:
        u = deconvolve(X[i])
        s = np.where(u > 1.5)[0]
        amps.append(u[s]); starts.append(s)
    return np.array(shapes), np.concatenate(amps), starts


def acf(x, lags):
    x = x - x.mean()
    return np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])


def residual_acf(X, rows, hour, weekend, lags):
    """ACF after removing each household's own hour x weekend mean profile."""
    out = []
    for i in rows:
        x = X[i]
        prof = np.zeros((2, 24))
        for w in (0, 1):
            for h in range(24):
                prof[w, h] = x[(hour == h) & (weekend == w)].mean()
        out.append(acf(x - prof[weekend, hour], lags))
    return np.array(out)


def heating_excess_daily(X, heat_rows, pure_rows):
    """Daily mean of (avg heating household - avg pure household)."""
    exc = X[heat_rows].mean(0) - X[pure_rows].mean(0)
    return exc.reshape(-1, 24).mean(1)


def dip_runs(r, level, drop, min_len):
    """Runs of at least `min_len` hours where r (reading minus heating and EV) stays more than `drop`
    below the household's always-on level. Returns a list of (household, start, length)."""
    out = []
    for i, row in enumerate(r < level[:, None] - drop):
        b = np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0])).reshape(-1, 2)
        out += [(i, s, e - s) for s, e in b if e - s >= min_len]
    return out


def leaderboard(path='../results/leaderboard.csv'):
    import pandas as pd
    log = pd.read_csv(path)
    log['n'] = log.file.str[:2].astype(int)
    log['kind'] = np.where(log.file.str.contains('probe'), 'probe', 'model')
    return log


def energy_mix(pred, ts):
    """Total energy (MWh) per component and month across all households."""
    months = np.asarray(ts.month)
    return {c: np.array([pred[c][:, months == m].sum() / 1000 for m in np.unique(months)])
            for c in pred}, np.unique(months)
