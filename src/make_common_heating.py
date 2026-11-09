"""Add the common daily heating factor to an existing (cached) prediction, without re-running the HMM.

    python make_common_heating.py stack14 [name]      (16 used the plain 24 h filter: ctx=0, rank=24)

The factor is taken out of always-on and other in the proportion in which those already carry it.
"""
import sys

import numpy as np

import common_factor as cf
import data
import submission

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'stack14'
    name = sys.argv[2] if len(sys.argv) > 2 else f'{base}_common_heating'
    X, temp, ts = data.load()
    hour, weekend, _, _ = data.calendar(ts, temp)
    cell = weekend * 24 + hour
    P = dict(np.load(f'../data/base_{base}.npz'))
    G, k = cf.estimate(X, P['heating'], P['ev'], cell)
    add = np.maximum(P['heating'] + k[:, None] * G, 0) - P['heating']
    H = k > 0
    # how much of the factor each of always-on / other currently carries (regression over heating households)
    ok = (P['ev'] < 0.05)[H]
    f = add[H][ok]
    da = cf.deviations(P['always_on'], cell, P['ev'] < 0.05)[H][ok]
    do = cf.deviations(P['other'], cell, P['ev'] < 0.05)[H][ok]
    ba, bo = (da * f).sum() / (f * f).sum(), (do * f).sum() / (f * f).sum()
    share = ba / (ba + bo)
    print(f'factor sd {G.std():.4f}; carried now by always-on {ba:.2f}, other {bo:.2f} -> always-on share {share:.2f}')
    Q = {c: v.copy() for c, v in P.items()}
    Q['heating'] = P['heating'] + add
    Q['always_on'] = np.maximum(P['always_on'] - share * add, 0)
    Q['other'] = np.maximum(P['other'] - (1 - share) * add, 0)
    N = X.size * 4
    d2 = sum(((Q[c] - P[c]) ** 2).sum() for c in P) / N
    print(f'change size (score added if the factor were NOT heating, base exact): +{d2:.6f}')
    np.savez_compressed(f'../data/base_{name}.npz', **Q)
    path, rep = submission.save(Q, ts, name, total=X)
    print('->', path, 'ready', rep['READY TO SUBMIT'])
