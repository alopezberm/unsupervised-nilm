"""Grid hidden Markov model for the always-on / other split.

Hidden state per hour (two independent AR(1) chains, discretised on a grid):
    w : always-on drift          always_on = max(0, level + w)
    z : latent 'other' activity  other     = max(0, m[weekend, hour] + s * z)
Observation:
    residual = reading - heating - ev = always_on + other + noise,  noise ~ N(0, sigma_e^2)
    (readings of exactly 0 are treated as censored: the meter clips at 0)

Forward-backward gives the exact posterior mean of always-on and other under this model, which
removes meter noise and tracks the slow always-on drift at the same time.
"""
import numpy as np
from scipy.stats import norm


def ar1_transition(grid, phi, sd, df=None):
    """Row-stochastic transition matrix of an AR(1) with marginal sd `sd` on `grid`.
    df: Student-t innovations (heavier-tailed jumps) instead of Gaussian."""
    step = sd * np.sqrt(1 - phi ** 2)
    u = (grid[None, :] - phi * grid[:, None]) / step
    if df is None:
        P = norm.pdf(u)
    else:
        from scipy.stats import t as student_t
        P = student_t.pdf(u * np.sqrt(df / (df - 2)), df)      # unit-variance t
    return P / P.sum(1, keepdims=True)


def stationary_of(P, iters=500):
    p = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(iters):
        p = p @ P
    return p / p.sum()


def stationary(grid, sd):
    p = norm.pdf(grid / sd)
    return p / p.sum()


def other_params_from_moments(resid, level, cell, var_ao, sigma_e, mask=None):
    """Latent 'other' parameters (s, m[48]) by the method of moments, per household.

    mean over a (weekend, hour) cell of the residual = level + E[other]  ->  E[other] per cell
    total variance of the residual = var(always-on) + var(other) + sigma_e^2  ->  var(other)
    mask: hours to use (e.g. away from EV sessions, where EV-fit errors inflate the variance).
    """
    from simulate import clipped_var, invert_mean
    from scipy.optimize import brentq
    n = resid.shape[0]
    mask = np.ones_like(resid, bool) if mask is None else mask
    s_out, m_out = np.zeros(n), np.zeros((n, 48))
    for i in range(n):
        r, c = resid[i, mask[i]], cell[mask[i]]
        cnt = np.bincount(c, minlength=48)
        mu = np.maximum(np.bincount(c, r, 48) / np.maximum(cnt, 1) - level[i], 2e-3)
        w = cnt / cnt.sum()
        target = max(r.var() - var_ao - sigma_e ** 2, 1e-3)
        between = (w * (mu - (w * mu).sum()) ** 2).sum()

        def gap(sd):
            m = np.array([invert_mean(v, sd) for v in mu])
            return (w * clipped_var(m, sd)).sum() + between - target
        lo, hi = 0.02, 3.0
        sd = brentq(gap, lo, hi) if gap(lo) < 0 < gap(hi) else (hi if gap(hi) < 0 else lo)
        s_out[i], m_out[i] = sd, [invert_mean(v, sd) for v in mu]
    return s_out, m_out


def drift_moments(sd, up):
    """Mean and variance of g(w) = w if w < 0 else up * w, w ~ N(0, sd^2)."""
    c = sd / np.sqrt(2 * np.pi)
    mean = (up - 1) * c
    return mean, sd ** 2 / 2 * (1 + up ** 2) - mean ** 2


def em_block(resid, zero, mask, mean_ao, cell, var_ao, meter_sigma, iters, **hmm_kw):
    """Full EM-style fit for a block of households (moments -> HMM posterior -> new mean always-on),
    so that blocks can run in parallel processes. With an asymmetric drift the HMM 'level' (zero-drift
    point) differs from the mean always-on by the mean of the drift."""
    g_mean, _ = drift_moments(hmm_kw.get('w_sd', 0.05), hmm_kw.get('up', 1.0))
    for _ in range(iters):
        s, m = other_params_from_moments(resid, mean_ao, cell, var_ao, meter_sigma, mask)
        ao, ot = posterior(resid, zero, mean_ao - g_mean, m, s, cell, **hmm_kw)
        mean_ao = ao.mean(1)
    return ao, ot, mean_ao, s, m


def posterior(resid, zero, level, m_cell, s, cell, w_phi=0.96, w_sd=0.05, z_phi=0.90,
              sigma_e=0.046, n_w=31, n_z=40, chunk=5, progress=None, up=1.0, loglik_only=False,
              z_df=None, s_mult=1.0, z_map=None):
    """resid, zero: [n, T]; level: [n]; m_cell: [n, 48]; s: [n]; cell: [T] (weekend*24 + hour).

    up: scale of upward drift (1 = symmetric drift, 0 = always-on can only dip below its level).
    Returns E[always_on], E[other] with shape [n, T], or the per-household log-likelihood if
    loglik_only (forward pass only, used to compare model settings on the real data).
    """
    n, T = resid.shape
    wg = np.linspace(-4 * w_sd, 4 * w_sd, n_w)
    zg = np.linspace(-4, 4, n_z)
    zv = zg if z_map is None else z_map(zg)          # latent value on the grid (z_map: skewed 'other')
    Pw = ar1_transition(wg, w_phi, w_sd).astype(np.float32)
    Pz = ar1_transition(zg, z_phi, 1.0, z_df).astype(np.float32)
    pw0 = stationary(wg, w_sd)
    pz0 = stationary(zg, 1.0) if z_df is None else stationary_of(Pz.astype(np.float64))
    s = s * s_mult
    E_ao = np.zeros((n, T)); E_ot = np.zeros((n, T)); ll = np.zeros(n)
    drift = np.where(wg < 0, wg, up * wg)
    log_norm = -np.log(sigma_e * np.sqrt(2 * np.pi))

    PwT, PzT = Pw.T.copy(), Pz.T.copy()
    for c0 in range(0, n, chunk):
        idx = np.arange(c0, min(c0 + chunk, n)); k = len(idx)
        ao = np.maximum(level[idx, None] + drift[None, :], 0).astype(np.float32)       # [k, W]
        # 'other' value on the z grid for every hour: [T, k, Z]
        OT = np.maximum(m_cell[idx][:, cell].T[:, :, None] + s[idx][None, :, None] * zv[None, None, :], 0)
        OT = OT.astype(np.float32)
        store = not loglik_only
        alpha = np.empty((T if store else 1, k, n_w, n_z), np.float32)
        emis = np.empty((T if store else 1, k, n_w, n_z), np.float32)
        a = np.broadcast_to((pw0[:, None] * pz0[None, :]).astype(np.float32), (k, n_w, n_z)).copy()
        for t in range(T):
            d = (resid[idx, t][:, None, None] - ao[:, :, None] - OT[t][:, None, :]) / sigma_e
            e = np.exp(-0.5 * d * d)
            zt = zero[idx, t]
            if zt.any():                                   # reading clipped at 0: P(true reading <= 0)
                e[zt] = norm.cdf(d[zt])
            e = np.maximum(e, 1e-30)
            if store:
                emis[t] = e
            if t:
                a = np.matmul(np.matmul(PwT[None], a), Pz)
            a = a * e
            norm_t = a.sum((1, 2), keepdims=True)
            ll[idx] += np.log(norm_t[:, 0, 0].astype(np.float64)) + np.where(zt, 0.0, log_norm)
            a /= norm_t
            np.maximum(a, 1e-30, out=a)                    # keep clear of denormals (very slow)
            if store:
                alpha[t] = a
        if loglik_only:
            continue
        b = np.ones((k, n_w, n_z), np.float32)
        for t in range(T - 1, -1, -1):
            g = alpha[t] * b
            g /= g.sum((1, 2), keepdims=True)
            E_ao[idx, t] = (g.sum(2) * ao).sum(1)
            E_ot[idx, t] = (g.sum(1) * OT[t]).sum(1)
            if t:
                b = np.matmul(np.matmul(Pw[None], b * emis[t]), PzT)
                b /= b.sum((1, 2), keepdims=True)
                np.maximum(b, 1e-30, out=b)
        if progress:
            progress(f'HMM households {idx[-1] + 1}/{n}')
    return ll if loglik_only else (E_ao, E_ot)
