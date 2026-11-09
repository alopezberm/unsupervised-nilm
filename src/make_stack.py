"""Rebuild the leaderboard-stacking submission from cached model outputs and logged scores.

    python make_stack.py v2c v3 v4 v2b v2a      (first name = reference)
    python make_stack.py v2c v3 v4 v2b v2a stack14_cf v5 stack14_heatfactor_v2 v6 --ridge=1e-5   (submission 20)

Needs data/base_<name>.npz for every name (written by run_v3.py / the notebook) and the
public scores in results/leaderboard.csv.
"""
import sys

import numpy as np
import pandas as pd

import data
import stack
import submission

FILES = {'v2a': '06_v2a', 'v2b': '09_v2b', 'v2c': '10_v2c', 'v3': '11_v3', 'v4': '13_v4',
         'stack14_cf': '16_stack14_common_heating', 'v5': '17_v5',
         'stack14_heatfactor_v2': '18_stack14_heatfactor_v2', 'v6': '19_v6'}


def scores():
    log = pd.read_csv('../results/leaderboard.csv').set_index('file')
    return {k: float(log.loc[f, 'public_score']) for k, f in FILES.items() if f in log.index}


def build(names, ridge=0.0):
    sc = scores()
    load = lambda n: dict(np.load(f'../data/base_{n}.npz'))
    P0, cands = load(names[0]), [load(n) for n in names[1:]]
    w, pred, sd, _, _ = stack.optimise(P0, sc[names[0]], cands, [sc[n] for n in names[1:]], ridge=ridge)
    return stack.mix(P0, cands, w), w, pred


if __name__ == '__main__':
    ridge = float(next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--ridge=')), 0))
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or ['v2c', 'v3', 'v4', 'v2b', 'v2a']
    mixed, w, pred = build(names, ridge)
    X, temp, ts = data.load()
    name = 'stack_' + '_'.join(names) if len(names) <= 5 else f'stack_all_{len(names)}_models'
    np.savez_compressed('../data/base_' + ('stack14' if names == ['v2c', 'v3', 'v4', 'v2b', 'v2a'] else name) + '.npz', **mixed)
    path, rep = submission.save(mixed, ts, name, total=X)
    print(f'weights {dict(zip(names[1:], np.round(w, 3)))}  predicted {pred:.5f}  -> {path}')
