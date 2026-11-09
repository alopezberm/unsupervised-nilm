"""Variants of the grid HMM (hmm.py) used for the last-day likelihood tests and model v7.

posterior_t takes the 'other' mean m_t, its scale s_t and the meter-noise sd sig_t per hour ([n, T]), so that
time-varying settings can be compared by the log-likelihood of the real readings:
  * 'other' scale per (weekend, hour) cell                      -> worse
  * 'other' mean per day of week x hour, per household          -> +1141 on 24 households, but mostly overfitting
  * same, shared by all households                              -> +29
  * meter noise sd^2 = (0.8 sigma_e)^2 + (4-7 % of the level)^2 -> +300 to +370 (v7, submission 30: no gain)
"""
import numpy as np
from scipy.stats import norm
import hmm


def posterior_t(resid, zero, level, m_t, s_t, sig_t, w_phi=0.91, w_sd=0.055, z_phi=0.90, up=0.75,
                n_w=31, n_z=40, loglik_only=False, **_):
    n, T = resid.shape
    wg = np.linspace(-4 * w_sd, 4 * w_sd, n_w); zg = np.linspace(-4, 4, n_z)
    Pw = hmm.ar1_transition(wg, w_phi, w_sd).astype(np.float32)
    Pz = hmm.ar1_transition(zg, z_phi, 1.0).astype(np.float32)
    pw0, pz0 = hmm.stationary(wg, w_sd), hmm.stationary(zg, 1.0)
    drift = np.where(wg < 0, wg, up * wg)
    E_ao = np.zeros((n, T)); E_ot = np.zeros((n, T)); ll = np.zeros(n)
    PwT, PzT = Pw.T.copy(), Pz.T.copy()
    for i in range(n):
        ao = np.maximum(level[i] + drift, 0).astype(np.float32)                          # [W]
        OT = np.maximum(m_t[i][:, None] + s_t[i][:, None] * zg[None, :], 0).astype(np.float32)   # [T, Z]
        store = not loglik_only
        alpha = np.empty((T if store else 1, n_w, n_z), np.float32); emis = np.empty_like(alpha)
        a = (pw0[:, None] * pz0[None, :]).astype(np.float32)
        for t in range(T):
            d = (resid[i, t] - ao[:, None] - OT[t][None, :]) / sig_t[i, t]
            e = norm.cdf(d) if zero[i, t] else np.exp(-0.5 * d * d)
            e = np.maximum(e, 1e-30).astype(np.float32)
            if store: emis[t] = e
            if t: a = PwT @ a @ Pz
            a = a * e
            nt = a.sum()
            ll[i] += np.log(nt) + (0.0 if zero[i, t] else -np.log(sig_t[i, t] * np.sqrt(2 * np.pi)))
            a /= nt; np.maximum(a, 1e-30, out=a)
            if store: alpha[t] = a
        if loglik_only: continue
        b = np.ones((n_w, n_z), np.float32)
        for t in range(T - 1, -1, -1):
            g = alpha[t] * b; g /= g.sum()
            E_ao[i, t] = (g.sum(1) * ao).sum(); E_ot[i, t] = (g.sum(0) * OT[t]).sum()
            if t:
                b = Pw @ (b * emis[t]) @ PzT; b /= b.sum(); np.maximum(b, 1e-30, out=b)
    return ll if loglik_only else (E_ao, E_ot)


def other_params_cells(resid, level, cells, ncell, var_ao, sigma_e, mask, per_cell_s):
    """m per cell (and s per cell if per_cell_s) by moments of the clipped Gaussian. Returns m_t, s_t [n, T]."""
    from simulate import clipped_var, invert_mean
    from scipy.optimize import brentq
    n, T = resid.shape
    m_t = np.zeros((n, T)); s_t = np.zeros((n, T))
    s0, _ = hmm.other_params_from_moments(resid, level, cells % 48 if ncell == 48 else (cells % 168 // 24 >= 5) * 24 + cells % 24,
                                          var_ao, sigma_e, mask)
    for i in range(n):
        r, c = resid[i, mask[i]], cells[mask[i]]
        cnt = np.bincount(c, minlength=ncell)
        mu = np.maximum(np.bincount(c, r, ncell) / np.maximum(cnt, 1) - level[i], 2e-3)
        m = np.zeros(ncell); s = np.full(ncell, s0[i])
        if per_cell_s:
            v = np.bincount(c, r * r, ncell) / np.maximum(cnt, 1) - (mu + level[i]) ** 2
            for j in range(ncell):
                target = max(v[j] - var_ao - sigma_e ** 2, 1e-4)
                gap = lambda sd: clipped_var(invert_mean(mu[j], sd), sd) - target
                lo, hi = 0.02, 3.0
                s[j] = brentq(gap, lo, hi) if gap(lo) < 0 < gap(hi) else (hi if gap(hi) < 0 else lo)
        for j in range(ncell):
            m[j] = invert_mean(mu[j], s[j])
        m_t[i] = m[cells]; s_t[i] = s[cells]
    return m_t, s_t
