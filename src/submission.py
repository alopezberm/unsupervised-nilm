"""Write and validate Kaggle submissions.

Format (from the competition page):
    kaggle_id,heating_kWh,ev_kWh,always_on_kWh,other_kWh
one row per kaggle_id, in the same order as the data file.
"""
import os
import numpy as np
import pandas as pd

from data import CSV, ROOT

SUB_DIR = os.path.join(ROOT, 'submissions')
COLUMNS = ['kaggle_id', 'heating_kWh', 'ev_kWh', 'always_on_kWh', 'other_kWh']


def reference_ids():
    return pd.read_csv(CSV, usecols=['kaggle_id']).kaggle_id.values


def build(pred, ts):
    """pred: dict component -> array[200, T] in household-major order (same as the data file)."""
    n_hh = pred['ev'].shape[0]
    hh = np.repeat(np.arange(n_hh), len(ts)).astype(str)
    stamp = np.tile(np.asarray(ts.strftime('%Y-%m-%d %H')), n_hh)
    return pd.DataFrame({
        'kaggle_id': np.char.add(np.char.add(hh, '_'), stamp.astype(str)),
        'heating_kWh': pred['heating'].ravel(),
        'ev_kWh': pred['ev'].ravel(),
        'always_on_kWh': pred['always_on'].ravel(),
        'other_kWh': pred['other'].ravel(),
    })


def validate(sub, total=None, allow_negative=False):
    """Hard checks that would make Kaggle reject or mis-score the file. Returns a report dict.

    allow_negative is for leaderboard probes, where unclipped perturbations keep the
    probe algebra exact (MSE itself is fine with negative values).
    """
    ref = reference_ids()
    report = {
        'header matches': list(sub.columns) == COLUMNS,
        'row count': f'{len(sub):,} (expected {len(ref):,})',
        'ids identical and in order': len(sub) == len(ref) and bool((sub.kaggle_id.values == ref).all()),
        'no NaN / inf': bool(np.isfinite(sub[COLUMNS[1:]].values).all()),
        'no negative values': bool((sub[COLUMNS[1:]].values >= 0).all()),
    }
    if total is not None:
        gap = sub[COLUMNS[1:]].sum(1).values - total.ravel()
        report['max |sum of parts - total| (kWh)'] = float(np.abs(gap).max())
    ok = report['header matches'] and report['ids identical and in order'] \
        and report['no NaN / inf'] and (allow_negative or report['no negative values']) and len(sub) == len(ref)
    report['READY TO SUBMIT'] = ok
    return report


def numbered_path(name):
    """'NN_name.csv' where NN is the upload order: reuse the number if the name exists, else next free."""
    os.makedirs(SUB_DIR, exist_ok=True)
    files = [f for f in os.listdir(SUB_DIR) if f.endswith('.csv') and f[:2].isdigit()]
    for f in files:
        if f[3:-4] == name:
            return os.path.join(SUB_DIR, f)
    nxt = max([int(f[:2]) for f in files], default=0) + 1
    return os.path.join(SUB_DIR, f'{nxt:02d}_{name}.csv')


def save(pred, ts, name, total=None, allow_negative=False):
    sub = build(pred, ts)
    report = validate(sub, total, allow_negative)
    if not report['READY TO SUBMIT']:
        raise ValueError(f'Submission failed validation: {report}')
    path = numbered_path(name)
    sub.to_csv(path, index=False, float_format='%.5f')
    return path, report
