"""Data loading, caching and quality checks.

The raw Kaggle file is long-format (one row per household-hour). Everything
downstream works on a dense matrix X[household, hour], which makes per-household
time-series operations (deconvolution, regressions) trivial and fast.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
CSV = os.path.join(DATA_DIR, 'kaggle_clustering_student_version.csv')
CACHE = os.path.join(DATA_DIR, 'cache.npz')


def read_raw():
    return pd.read_csv(CSV, parse_dates=['timestamp'])


def load():
    """Return (X, temp, ts): X[200, T] consumption in kWh, temp[T] in °C, ts DatetimeIndex."""
    if not os.path.exists(CACHE):
        df = read_raw()
        X = df.pivot(index='household_id', columns='timestamp', values='consumption_kWh')
        temp = df[df.household_id == 0].set_index('timestamp').temperature_C.loc[X.columns]
        np.savez(CACHE, X=X.values, temp=temp.values,
                 ts=X.columns.values.astype('datetime64[h]').astype('int64'))
    d = np.load(CACHE)
    return d['X'], d['temp'], pd.DatetimeIndex(pd.to_datetime(d['ts'], unit='h'))


def calendar(ts, temp, tref=20.5):
    """Hour of day, weekend flag, daily-mean temperature and heating degree signal."""
    hour = np.asarray(ts.hour)
    weekend = (np.asarray(ts.dayofweek) >= 5).astype(int)
    t_daily = np.repeat(temp.reshape(-1, 24).mean(1), 24)
    hdd = np.maximum(0, tref - t_daily)
    return hour, weekend, t_daily, hdd


def quality_report(df=None):
    """Sanity checks on the raw file. Returns a one-column DataFrame for display."""
    df = read_raw() if df is None else df
    per_hh = df.groupby('household_id').timestamp.agg(['min', 'max', 'count'])
    full_grid = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq='h')
    temp_spread = df.pivot(index='timestamp', columns='household_id', values='temperature_C').std(1).max()
    expected_ids = df.household_id.astype(str) + '_' + df.timestamp.dt.strftime('%Y-%m-%d %H')
    checks = {
        'rows': len(df),
        'households': df.household_id.nunique(),
        'hours per household': f"{per_hh['count'].min()}–{per_hh['count'].max()}",
        'period': f"{df.timestamp.min():%Y-%m-%d} → {df.timestamp.max():%Y-%m-%d %H:%M}",
        'complete hourly grid (no gaps / DST jumps)': bool((per_hh['count'] == len(full_grid)).all()),
        'missing values': int(df.isna().sum().sum()),
        'duplicate (household, timestamp)': int(df.duplicated(['household_id', 'timestamp']).sum()),
        'kaggle_id == "<household>_<YYYY-MM-DD HH>"': bool((expected_ids == df.kaggle_id).all()),
        'temperature identical across households': bool(temp_spread < 1e-9),
        'negative readings': int((df.consumption_kWh < 0).sum()),
        'exact-zero readings (clipped)': int((df.consumption_kWh == 0).sum()),
        'max reading (kWh)': float(df.consumption_kWh.max()),
    }
    return pd.DataFrame({'value': checks})
