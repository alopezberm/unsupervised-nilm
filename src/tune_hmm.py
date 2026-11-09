"""Choose the always-on drift model by maximum likelihood on (a subset of) the real households.

For each setting the 'other' parameters are re-estimated by moments, then the HMM forward pass
returns the log-likelihood of the observed readings. Higher = the setting explains the data better.
"""
import itertools
import sys
import time

import numpy as np
from joblib import Parallel, delayed

import data
import disaggregate as dis
import hmm


def drift_moments(sd, up):
    """Mean and variance of g(w) = w if w < 0 else up * w, w ~ N(0, sd^2)."""
    c = sd / np.sqrt(2 * np.pi)
    mean = (up - 1) * c
    second = sd ** 2 / 2 * (1 + up ** 2)
    return mean, second - mean ** 2


def loglik(cfg, resid, zero, mask, mean_ao, cell):
    sd, phi, up, sigma_e, z_phi = cfg[:5]
    extra = cfg[5] if len(cfg) > 5 else {}
    g_mean, g_var = drift_moments(sd, up)
    var_ao = g_var + dis.AO_SIGMA ** 2
    s, m = hmm.other_params_from_moments(resid, mean_ao, cell, var_ao, dis.METER_SIGMA, mask)
    level = mean_ao - g_mean
    ll = hmm.posterior(resid, zero, level, m, s, cell, w_phi=phi, w_sd=sd, z_phi=z_phi, sigma_e=sigma_e,
                       up=up, loglik_only=True, **extra)
    return cfg, ll.sum()


def main(grid, households):
    X, temp, ts = data.load()
    hour, weekend, _, _ = data.calendar(ts, temp)
    cell = weekend * 24 + hour
    v2c = np.load('../data/base_v2c.npz'); v3 = np.load('../data/base_v3.npz')
    idx = households
    resid = (X - v2c['heating'] - v2c['ev'])[idx]
    zero = (X == 0)[idx]
    mask = (v2c['ev'] < 0.3)[idx]
    mean_ao = v3['always_on'][idx].mean(1)
    t0 = time.time()
    out = Parallel(n_jobs=4)(delayed(loglik)(cfg, resid, zero, mask, mean_ao, cell) for cfg in grid)
    out.sort(key=lambda r: -r[1])
    best = out[0][1]
    print(f'{len(grid)} settings on {len(idx)} households in {time.time() - t0:.0f}s', flush=True)
    print('  drift_sd  drift_phi  up    sigma_e  other_phi   delta log-lik vs best', flush=True)
    for cfg, ll in out:
        extra = cfg[5] if len(cfg) > 5 else ''
        print(f'  {cfg[0]:.3f}     {cfg[1]:.3f}     {cfg[2]:.2f}  {cfg[3]:.3f}    {cfg[4]:.2f}       {ll - best:10.1f}   {extra}', flush=True)
    return out


if __name__ == '__main__':
    X, _, _ = data.load()
    ev = X.max(1) > 10
    types = np.load('../data/base_v3.npz')['heating'].max(1) > 0
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.where(g)[0], 6, replace=False)
                           for g in (~ev & ~types, ~ev & types, ev & ~types, ev & types)])
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    if stage == '1':
        grid = list(itertools.product([0.04, 0.05, 0.065], [0.93, 0.96, 0.98], [1.0, 0.5, 0.0], [0.046], [0.90]))
    elif stage == '1b':
        grid = list(itertools.product([0.045, 0.05, 0.055, 0.06], [0.86, 0.90, 0.93], [1.0, 0.75], [0.046], [0.90]))
    elif stage == '4':
        base = (0.055, 0.91, 0.75, 0.036, 0.90)
        grid = [base + (dict(z_df=df, s_mult=sm),) for df in (None, 8, 4) for sm in (0.9, 1.0, 1.1)]
    elif stage == '3':
        grid = list(itertools.product([0.055, 0.06, 0.065], [0.91, 0.93], [0.75], [0.033, 0.036, 0.038], [0.90]))
    else:
        sd, phi, up = map(float, sys.argv[2:5])
        grid = list(itertools.product([sd], [phi], [up], [0.038, 0.046, 0.055], [0.87, 0.90, 0.93]))
    main(grid, np.sort(pick))
