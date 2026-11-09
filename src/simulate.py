"""A simulator calibrated to everything learned about the real data.

It produces data *with known ground truth*, so model variants can be scored offline before
spending a Kaggle submission, and real-vs-simulated statistics reveal what the simulator
(and therefore the model) is still missing.

Every household keeps its real type and fitted parameters; only the randomness is redrawn.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from data import calendar
import disaggregate as dis

OTHER_PHI = 0.90


def clipped_mean(m, s):
    """E[max(0, m + z)], z ~ N(0, s^2)."""
    return m * norm.cdf(m / s) + s * norm.pdf(m / s)


def clipped_var(m, s):
    a = m / s
    ex2 = (m ** 2 + s ** 2) * norm.cdf(a) + m * s * norm.pdf(a)
    return ex2 - clipped_mean(m, s) ** 2


def invert_mean(mu, s):
    """m such that E[max(0, m + z)] = mu."""
    mu = max(mu, 1e-4)
    return brentq(lambda m: clipped_mean(m, s) - mu, -10 * s, 10 * s + mu)


def calibrate_other(other, hour, weekend, target_extra_var=0.0):
    """Per household: latent sd s and per-(weekend, hour) latent mean m so that the clipped AR process
    reproduces the household's mean profile and total variance of 'other'."""
    n = other.shape[0]
    cell = weekend * 24 + hour
    mu = np.stack([np.bincount(cell, other[i], 48) / np.bincount(cell, minlength=48) for i in range(n)])
    params = []
    for i in range(n):
        target = other[i].var() - target_extra_var
        def gap(s):
            m = np.array([invert_mean(v, s) for v in mu[i]])
            w = np.bincount(cell, minlength=48) / len(cell)
            return (w * clipped_var(m, s)).sum() + (w * (mu[i] - (w * mu[i]).sum()) ** 2).sum() - target
        lo, hi = 0.02, 3.0
        s = brentq(gap, lo, hi) if gap(lo) < 0 < gap(hi) else (hi if gap(hi) < 0 else lo)
        params.append((s, np.array([invert_mean(v, s) for v in mu[i]])))
    return params


def ar1(n, T, phi, sigma, rng):
    e = rng.standard_normal((n, T)) * sigma * np.sqrt(1 - phi ** 2)
    z = np.zeros((n, T)); z[:, 0] = rng.standard_normal(n) * sigma
    for t in range(1, T):
        z[:, t] = phi * z[:, t - 1] + e[:, t]
    return z


def simulate(fit, temp, ts, seed=0, tail_rel=0.035, ao_phi=dis.AO_PHI, ao_sigma=dis.AO_SIGMA,
             slow_phi=0.96, slow_sigma=0.05, slow_up=1.0, meter_sigma=dis.METER_SIGMA, other_params=None):
    """fit: a disaggregate.Result on the real data (supplies household types and parameters)."""
    rng = np.random.default_rng(seed)
    n, T = fit.X.shape
    hour, weekend, _, hdd = calendar(ts, temp)
    kernel = fit.kernel
    truth = {}

    # always-on = level + fast fluctuation (hours) + slow drift (days); the slow drift reproduces the
    # long multi-hour dips below the level seen in the real data
    slow = ar1(n, T, slow_phi, slow_sigma, rng) if slow_sigma > 0 else np.zeros((n, T))
    slow = np.where(slow < 0, slow, slow_up * slow)          # upward drift scaled (likelihood: ~0.75)
    truth['always_on'] = np.maximum(fit.a[:, None] + ar1(n, T, ao_phi, ao_sigma, rng) + slow, 0)
    truth['heating'] = np.where(fit.has_heat[:, None], fit.k[:, None] * hdd * fit.hp[weekend, hour], 0.0)

    ev = np.zeros((n, T))
    tail = kernel.copy(); tail[0] = 0
    for i in np.where(fit.has_ev)[0]:
        starts, A = fit.events[i]
        for s, a in zip(starts, A):
            a = max(a, 0)
            L = min(len(kernel), T - s)
            dev = rng.standard_normal() * tail_rel * a
            ev[i, s:s + L] += a * kernel[:L] + dev * tail[:L]
    truth['ev'] = np.maximum(ev, 0)

    if other_params is None:
        other_params = calibrate_other(fit.pred['other'], hour, weekend)
    cell = weekend * 24 + hour
    z = ar1(n, T, OTHER_PHI, 1.0, rng)
    other = np.zeros((n, T))
    for i, (s, m) in enumerate(other_params):
        other[i] = np.maximum(m[cell] + s * z[i], 0)
    truth['other'] = other

    total = sum(truth.values())
    X = np.round(np.maximum(total + rng.standard_normal((n, T)) * meter_sigma, 0), 4)
    return X, truth, other_params


def score(pred, truth):
    """Kaggle metric: MSE averaged over the 4 columns and all rows; also per component."""
    per = {c: float(((pred[c] - truth[c]) ** 2).mean()) for c in truth}
    return sum(per.values()) / 4, per
