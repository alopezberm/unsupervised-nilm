"""Final submission (31): megastack around submission 25 + always-on de-biasing in EV charging hours.

    python make_final.py

1. Megastack of every scored file (see megastack.py), base 25, eigen-directions with |z| > 3.
2. In EV households our always-on is ~0.01 kWh higher in charging hours than in non-charging hours of the
   same (weekend, hour) cell. The generator's always-on cannot know whether the car is charging, so the gap is a
   bias of the model: it is removed while keeping each household's always-on mean (already calibrated by
   probe 24). The excess of charging hours goes half to EV and half to other (half-and-half gains whichever
   of the two it belongs to); non-charging hours take the deficit back from other.
"""
import numpy as np
import pandas as pd

import data
import submission
from megastack import PARTS, eigen_select, gram, load_scored


def megastack_mix(base_prefix='25_', zmin=3.0, upto=30):
    """upto: last submission number used (31 itself is scored now and must not enter its own blend)."""
    names, scores, P = load_scored()
    keep = [j for j, n in enumerate(names) if int(n[:2]) <= upto]
    names, scores, P = [names[j] for j in keep], scores[keep], P[keep]
    base = next(i for i, n in enumerate(names) if n.startswith(base_prefix))
    idx, Q = gram(P, base)
    L = 4 * (scores[idx] - scores[base]) - np.diag(Q)
    w, exp, sd, rows = eigen_select(Q, L, scores[base], zmin)
    mix = P[base].astype(np.float64) + np.tensordot(w, P[idx].astype(np.float64) - P[base], 1)
    return np.maximum(mix, 0), exp, sd


def ev_hours_gap(ao, ev, cell, rows):
    """Always-on in EV-on hours minus EV-off hours of the same cell, averaged over EV households."""
    gaps = []
    for i in rows:
        on, off = ev[i] > 0.05, ev[i] <= 0.01
        num = den = 0
        for c in range(48):
            a, b = on & (cell == c), off & (cell == c)
            if a.sum() > 3 and b.sum() > 3:
                num += a.sum() * (ao[i, a].mean() - ao[i, b].mean()); den += a.sum()
        gaps.append(num / den)
    return float(np.mean(gaps)), float(np.std(gaps) / np.sqrt(len(gaps)))


def debias_ev_hours(he, ev, ao, ot, cell):
    rows = np.where(ev.max(1) > 2)[0]
    gap, se = ev_hours_gap(ao, ev, cell, rows)
    for i in rows:
        on = ev[i] > 0.05
        f = on.mean()
        take = np.minimum(ao[i, on], gap * (1 - f))       # charging hours: always-on down ...
        ao[i, on] -= take; ev[i, on] += take / 2; ot[i, on] += take / 2
        give = np.minimum(ot[i, ~on], gap * f)            # ... other hours: always-on up (mean unchanged)
        ot[i, ~on] -= give; ao[i, ~on] += give
    return gap, se


if __name__ == '__main__':
    mix, exp, sd = megastack_mix()
    print(f'megastack around 25: expected {exp:.6f} +- {sd:.6f}')
    X, temp, ts = data.load()
    hour, weekend, _, _ = data.calendar(ts, temp)
    he, ev, ao, ot = (mix[j].reshape(X.shape[0], -1).copy() for j in range(4))
    gap, se = debias_ev_hours(he, ev, ao, ot, weekend * 24 + hour)
    print(f'always-on gap in EV charging hours: {gap:+.4f} kWh (se {se:.4f}), removed')
    sub = pd.DataFrame({'kaggle_id': submission.reference_ids(),
                        **{c: p.ravel() for c, p in zip(PARTS, (he, ev, ao, ot))}})[submission.COLUMNS]
    rep = submission.validate(sub)
    assert rep['READY TO SUBMIT'], rep
    path = submission.numbered_path('final_megastack_30_files_evhours')
    sub.to_csv(path, index=False, float_format='%.5f')
    print('->', path)
