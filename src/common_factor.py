"""Common daily heating factor.

All heating households share one random daily deviation from k * HDD * profile: within a day the
deviation is (almost) perfectly correlated hour to hour, it breaks at midnight, it is independent from
day to day and unrelated to temperature, and its size scales with how much heating is used.

Per household it hides under the always-on / other noise (sd ~0.25), but averaged over the ~85 heating
households it is visible, so it can be estimated and added back to heating:

    G_t      = sum_i k_i dev_it / sum_i k_i^2        (dev = reading - heating - EV - own hour profile)
    G_hat_d  = Wiener filter of day d (+ 6 h of each neighbouring day; signal covariance from split
               halves, rank 3, scaled by that day's heating level)
    heating_it += k_i * G_hat_t
"""
import numpy as np


def deviations(r, cell, ok):
    """r minus each household's own (weekend, hour) mean, NaN where not usable."""
    prof = np.stack([np.bincount(cell, u * o, 48) / np.maximum(np.bincount(cell, o.astype(float), 48), 1)
                     for u, o in zip(r, ok)])
    return np.where(ok, r - prof[:, cell], np.nan)


def weighted_factor(dev, k):
    """Cross-household least-squares factor per hour: dev_it ~ k_i * G_t."""
    w = np.where(np.isnan(dev), 0.0, k[:, None])
    return np.nansum(dev * k[:, None], 0) / np.maximum((w ** 2).sum(0), 1e-9)


def estimate(X, heating, ev, cell, ev_thr=0.05, **wiener):
    """Returns (G_hat [T], k [n]) with k = 0 for households without heating.
    wiener: ctx / rank of daily_wiener (v5 used ctx=0, rank=24: a plain 24 h filter)."""
    n, T = X.shape
    H = np.where(heating.max(1) > 0)[0]
    k = np.zeros(n)
    k[H] = heating[H].sum(1) / heating[H].sum(1).mean()
    dev = deviations(X - heating - ev, cell, ev < ev_thr)[H]
    lam = (heating[H] / k[H][:, None]).mean(0).reshape(-1, 24).mean(1)          # heating level per day
    return daily_wiener(dev, k[H], lam, **wiener), k


def daily_wiener(dev, k, lam=None, ctx=6, rank=3):
    """Factor G_t of dev_it ~ k_i G_t, denoised day by day with a Wiener filter.
    lam: relative size of the factor per day (signal covariance scales with lam^2).
    ctx: hours of the neighbouring days used as context; rank: signal covariance kept to its top
    eigenvectors (both chosen by split-half cross-validation: +26% captured vs a plain 24 h filter)."""
    G = weighted_factor(dev, k)
    A = weighted_factor(dev[::2], k[::2])
    B = weighted_factor(dev[1::2], k[1::2])
    nd = len(G) // 24
    lam = np.ones(nd) if lam is None else lam / np.sqrt((lam ** 2).mean())

    def blocks(x):                               # day d with ctx hours on each side
        xp = np.r_[np.zeros(ctx), x, np.zeros(ctx)]
        return np.stack([xp[d * 24: d * 24 + 24 + 2 * ctx] for d in range(nd)])
    Ab, Bb, Gb = blocks(A), blocks(B), blocks(G)
    # signal covariance per unit lam (cross-half moments), noise from the half difference
    S = sum(np.outer(a, b) + np.outer(b, a) for a, b in zip(Ab, Bb)) / 2 / (lam ** 2).sum()
    w, U = np.linalg.eigh((S + S.T) / 2)
    w = np.maximum(w, 0)
    w[:-rank] = 0
    S = (U * w) @ U.T
    N_full = np.cov((Ab - Bb).T) / 4
    out = np.empty((nd, 24))
    for d in range(nd):
        Sd = lam[d] ** 2 * S
        out[d] = (Sd @ np.linalg.solve(Sd + N_full, Gb[d]))[ctx:ctx + 24]
    return out.ravel()
