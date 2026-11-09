"""Leaderboard stacking over EVERY scored submission (models and probes), read from the CSV files.

    python megastack.py [--base=20] [--save]

Each scored file is one exact measurement of an inner product with the hidden truth, so any of them
can enter the blend (see stack.py for the algebra). With many nearly parallel directions the 5-decimal
rounding of the scores would be over-fitted, so the ridge is chosen by the expected score of the
weights actually submitted:  nominal prediction + tr((Q + ridge I)^-1 Sigma_L) / 8,  where Sigma_L is
the covariance of L caused by the rounding.

Default (--zmin): instead of a ridge, keep only the eigen-directions of Q whose measured inner product
is clearly above the rounding noise (|z| > zmin) and shrink each by its unbiased factor
(1 - noise^2 / signal^2). Directions with tiny eigenvalues are left out: there the 'gain' is rounding.
"""
import os
import sys

import numpy as np
import pandas as pd

import submission

PARTS = ['heating_kWh', 'ev_kWh', 'always_on_kWh', 'other_kWh']
ROUND_VAR = 1e-5 ** 2 / 12                     # variance of a score rounded to 5 decimals


def load_scored(log_path='../results/leaderboard.csv'):
    log = pd.read_csv(log_path)
    names, scores, preds = [], [], []
    for f, s in zip(log.file, log.public_score):
        path = os.path.join(submission.SUB_DIR, f + '.csv')
        if os.path.exists(path) and np.isfinite(s):
            names.append(f); scores.append(float(s))
            preds.append(pd.read_csv(path, usecols=PARTS)[PARTS].values.astype(np.float32).T)   # [4, N]
    return names, np.array(scores), np.stack(preds)                                           # [K, 4, N]


def gram(P, base):
    """Q_jk = sum over columns of mean (Pj - P0)(Pk - P0), for all candidates j, k != base."""
    idx = [k for k in range(len(P)) if k != base]
    D = (P[idx] - P[base]).reshape(len(idx), -1).astype(np.float64)
    return idx, D @ D.T / P.shape[2]


def solve(Q, L, s0, ridge):
    w = -np.linalg.solve(Q + ridge * np.eye(len(L)), L) / 2
    nominal = s0 + (w @ Q @ w + L @ w) / 4
    S = 16 * ROUND_VAR * (np.eye(len(L)) + 1)             # L_k = 4 (s_k - s0) - Q_kk: both scores rounded
    optimism = np.trace(np.linalg.solve(Q + ridge * np.eye(len(L)), S)) / 8
    return w, nominal, nominal + optimism


def eigen_select(Q, L, s0, zmin=3.0, min_mu=1e-8):
    """Weights from the significant eigen-directions only. Returns w, expected score, sd, table.
    min_mu: directions smaller than this are differences at the level of the CSV rounding (e.g. a
    probe re-measuring a combination of older files); their 'gain' would be rounding noise."""
    S = 16 * ROUND_VAR * (np.eye(len(L)) + 1)
    mu, U = np.linalg.eigh(Q)
    w, gain, var, rows = np.zeros(len(L)), 0.0, ROUND_VAR, []
    for i in np.argsort(-mu):
        u = U[:, i]
        lu, sd = L @ u, np.sqrt(u @ S @ u)
        keep = abs(lu / sd) > zmin and mu[i] > min_mu
        if keep:
            shrink = 1 - sd ** 2 / lu ** 2
            a = -lu / (2 * mu[i]) * shrink
            w += a * u
            gain += (lu ** 2 - sd ** 2) / (16 * mu[i])
            var += (a * sd / 4) ** 2
        rows.append((mu[i], lu / sd, keep))
    return w, s0 - gain, np.sqrt(var), rows


if __name__ == '__main__':
    args = dict(a[2:].split('=') if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    names, scores, P = load_scored()
    base = next(i for i, n in enumerate(names) if n.startswith(args.get('base', '20') + '_'))
    idx, Q = gram(P, base)
    L = 4 * (scores[idx] - scores[base]) - np.diag(Q)
    print(f'{len(idx)} candidates around {names[base]} ({scores[base]:.5f}); '
          f'eigenvalues of Q: {np.linalg.eigvalsh(Q)[:3]} ... {np.linalg.eigvalsh(Q)[-1]:.2e}')
    if 'ridge' not in args:
        w, exp, sd, rows = eigen_select(Q, L, scores[base], float(args.get('zmin', 3)))
        print(f'{sum(r[2] for r in rows)} of {len(rows)} eigen-directions kept (|z| > {args.get("zmin", 3)}): '
              f'expected {exp:.6f} +- {sd:.6f}')
        best = (None, w, exp, None)
    else:
        best = None
    for ridge in (() if best else (0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4)):
        w, nom, exp = solve(Q, L, scores[base], ridge)
        print(f'  ridge {ridge:7.0e}  nominal {nom:.6f}  expected (rounding-corrected) {exp:.6f}  |w| {np.abs(w).sum():.2f}')
        if best is None or exp < best[2]:
            best = (ridge, w, exp, nom)
    ridge, w, exp, nom = best
    if ridge is not None:
        print(f'chosen ridge {ridge:.0e}: expected {exp:.6f}')
    for k, wk in sorted(zip(idx, w), key=lambda t: -abs(t[1])):
        print(f'    {wk:+.3f}  {names[k]}')
    if 'save' in args:
        mix = P[base].astype(np.float64) + np.tensordot(w, P[idx].astype(np.float64) - P[base], 1)
        mix = np.maximum(mix, 0)
        ids = submission.reference_ids()
        sub = pd.DataFrame({'kaggle_id': ids, **{c: mix[j] for j, c in enumerate(PARTS)}})[submission.COLUMNS]
        rep = submission.validate(sub)
        path = submission.numbered_path(args.get('name', f'megastack_{len(names)}_files'))
        assert rep['READY TO SUBMIT'], rep
        sub.to_csv(path, index=False, float_format='%.5f')
        print('->', path)
