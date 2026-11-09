"""Calibration probes around an already-scored submission (read from its CSV).

    python make_probe.py <base file number> <direction> <delta>

Each probe moves energy between two components (the sum stays the same), so its score measures one
inner product with the truth; megastack.py then uses it as one more direction.

    drift     always-on minus its household mean  -> other      (is the HMM drift too big / too small?)
    evtail    EV hours below 1 kWh (session tails) -> other
    heat_ev   heating of EV + heating households   -> other
    ao_evhh   constant always-on level of EV households -> other (are EV-household levels biased?)
    ao_heathh same for households with heating and no EV
    ao_pure   same for households with neither heating nor EV
    ao_to_ev  EV households, hours with EV running: constant always-on -> EV (is the extra level EV?)
    drift_day     day-to-day part of the drift (daily mean minus its 7-day average) -> other
    drift_within  within-day part of the drift (drift minus its daily mean) -> other
"""
import os
import sys

import numpy as np
import pandas as pd

import submission
from megastack import PARTS

N_HH = 200


def direction(P, name):
    """P: dict part -> [200, T]. Returns dict of changes per part for delta = 1."""
    ao, ev, heat = P['always_on_kWh'], P['ev_kWh'], P['heating_kWh']
    if name == 'drift':
        d = ao - ao.mean(1, keepdims=True)
        return {'always_on_kWh': d, 'other_kWh': -d}
    if name == 'evtail':
        d = np.where((ev > 0) & (ev < 1.0), ev, 0)
        return {'ev_kWh': d, 'other_kWh': -d}
    if name == 'heat_ev':
        evhh = (ev.max(1) > 2)[:, None]
        d = np.where(evhh, heat, 0)
        return {'heating_kWh': d, 'other_kWh': -d}
    if name == 'ao_evhh':
        d = np.where((ev.max(1) > 2)[:, None], 1.0, 0.0) * np.ones_like(ao)
        return {'always_on_kWh': d, 'other_kWh': -d}
    evhh = (ev.max(1) > 2)[:, None]
    heathh = (heat.mean(1) > 0.01)[:, None]
    if name in ('ao_heathh', 'ao_pure'):
        rows = (~evhh & heathh) if name == 'ao_heathh' else (~evhh & ~heathh)
        d = np.where(rows, 1.0, 0.0) * np.ones_like(ao)
        return {'always_on_kWh': d, 'other_kWh': -d}
    if name == 'ao_to_ev':
        d = np.where(evhh & (ev > 0.05), 1.0, 0.0)
        return {'always_on_kWh': -d, 'ev_kWh': d}
    if name in ('drift_day', 'drift_within'):
        drift = (ao - ao.mean(1, keepdims=True)).reshape(N_HH, -1, 24)
        day = drift.mean(2)
        if name == 'drift_within':
            d = (drift - day[:, :, None]).reshape(N_HH, -1)
        else:
            week = np.stack([np.convolve(x, np.ones(7) / 7, 'same') for x in day])
            d = np.repeat(day - week, 24, 1)
        return {'always_on_kWh': d, 'other_kWh': -d}
    raise ValueError(name)


if __name__ == '__main__':
    num, name, delta = sys.argv[1], sys.argv[2], float(sys.argv[3])
    log = pd.read_csv('../results/leaderboard.csv')
    base = log[log.file.str.startswith(num + '_')].iloc[0]
    df = pd.read_csv(os.path.join(submission.SUB_DIR, base.file + '.csv'))
    P = {c: df[c].values.reshape(N_HH, -1) for c in PARTS}
    D = direction(P, name)
    for c, d in D.items():
        df[c] = (P[c] + delta * d).ravel()
    extra = sum(((delta * d) ** 2).sum() for d in D.values()) / (4 * df.shape[0])
    rep = submission.validate(df, allow_negative=True)
    assert rep['READY TO SUBMIT'], rep
    path = submission.numbered_path(f'probe_{name}_{delta:+g}_on_{num}')
    df.to_csv(path, index=False, float_format='%.5f')
    print(f'{path}\n  score if base were exact on this direction: {base.public_score + extra:.5f} '
          f'(base {base.public_score:.5f} + {extra:.6f}); negatives: {(df[PARTS].values < 0).sum()}')
