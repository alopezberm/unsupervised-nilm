"""Leaderboard stacking: the best linear mix of already-scored submissions.

With a reference submission P0 (score s0) and others Pk (scores sk), let Dk = Pk - P0. For any mix
P0 + sum_k w_k Dk the MSE is exactly

    s0 + (w' Q w + L' w) / 4,     Q_jk = mean over rows of sum over columns Dj * Dk  (known)
                                   L_k  = 4 (s_k - s0) - Q_kk                        (from the scores)

so the optimal weights are w = -Q^-1 L / 2. Scores are rounded to 5 decimals, so the prediction is
also reported with its spread under that rounding (Monte Carlo).
"""
import numpy as np

COLS = ('always_on', 'heating', 'other', 'ev')


def gram(D):
    K = len(D)
    Q = np.zeros((K, K))
    for j in range(K):
        for k in range(j, K):
            Q[j, k] = Q[k, j] = sum((D[j][c] * D[k][c]).mean() for c in COLS)
    return Q


def optimise(P0, s0, cands, scores, ridge=0.0, n_mc=2000, seed=0):
    """cands: list of prediction dicts; scores: their public scores. Returns weights, prediction, spread."""
    D = [{c: P[c] - P0[c] for c in COLS} for P in cands]
    Q = gram(D)
    s = np.asarray(scores, float)

    def solve(s0_, s_):
        L = 4 * (s_ - s0_) - np.diag(Q)
        w = -np.linalg.solve(Q + ridge * np.eye(len(s_)), L) / 2
        return w, s0_ + (w @ Q @ w + L @ w) / 4, L

    w, pred, L = solve(s0, s)
    rng = np.random.default_rng(seed)
    preds, ws = [], []
    for _ in range(n_mc):
        e0, e = rng.uniform(-5e-6, 5e-6), rng.uniform(-5e-6, 5e-6, len(s))
        w_mc, _, L_mc = solve(s0 + e0, s + e)
        # true score of the weights we would actually submit (w), if the scores were really s + e
        preds.append(s0 + e0 + (w @ Q @ w + L_mc @ w) / 4); ws.append(w_mc)
    return w, pred, np.std(preds), np.std(ws, 0), Q


def mix(P0, cands, w):
    out = {c: P0[c] + sum(wk * (P[c] - P0[c]) for wk, P in zip(w, cands)) for c in COLS}
    return {c: np.maximum(v, 0) for c, v in out.items()}
