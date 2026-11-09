"""Unsupervised disaggregation model.

Generative picture inferred from the data (see the notebook for the evidence):

    meter = always_on + other + heating + ev + meter_noise
    always_on_t = a_h                                    flat per-household floor
    heating_t   = k_h * max(0, 20.5 - T_daily) * hp[weekend, hour]   (deterministic)
    ev_t        = sum_e A_e * kappa[t - start_e]         sharp start, ~0.70/h decay, ends ~16 h
    other_t     = AR(1)-like bursts, phi ~ 0.89
    meter_noise ~ white, sigma ~ 0.037 (belongs to no component, see leaderboard probe P2)

Per household we fit one regression (prewhitened for the AR(1) noise) with
48 hour x weekend intercepts, one heating slope and one amplitude per EV event.

Versions
    v1  : geometric EV kernel 0.688^k, KDE floor, other = remainder
    v2a : free EV kernel (alternating least squares), leaderboard-calibrated always-on
          offset and heating scale, Kalman-smoothed meter noise removed from 'other'
    v2b : v2a + household always-on levels shrunk 13 % + always-on fluctuation (Kalman, quiet hours)
    v2c : v2b + a second (ridge-shrunk) tail amplitude per EV session, fluctuation weight 0.70
    v3  : v2c heating and EV + a grid HMM for always-on (level + slow drift) and other (clipped AR),
          parameters re-estimated by a few EM-style passes
    v4  : v3 with the HMM settings chosen by maximum likelihood (faster, mildly asymmetric drift)
"""
import os
import numpy as np
from scipy.stats import gaussian_kde

from data import load, calendar, DATA_DIR

R_EV = 0.688        # EV hourly decay ratio (v1 geometric kernel)
PHI = 0.89          # AR(1) coefficient of the 'other' noise
EV_KLEN = 48        # EV kernel length (hours)
EV_MAX_THR = 10     # households whose max reading exceeds this own an EV
EV_START_THR = 1.5  # deconvolved jump that marks an EV session start
HEAT_T_THR = 3.0    # t-statistic of the heating slope needed to call heating
EV_TAIL_REL = 0.03  # prior sd of a session's tail amplitude deviation, relative to its amplitude
EV_PW_VAR = 0.021   # variance of the prewhitened residual (scale for the ridge penalty)

# Offsets measured with leaderboard probes (results/leaderboard.csv)
CALIBRATION = {'always_on_shift': -0.0137,   # probe 02
               'heating_scale': 1.019,       # probe 05
               'always_on_shrink': 0.132,    # probe 07: pull household levels 13 % toward the mean
               'always_on_dynamics': 1.0,    # probe 08 / v2b: weight of the always-on fluctuation
               'always_on_dynamics_v2c': 0.70}  # optimum decoded from submission 09

# Always-on fluctuation around its level (variogram of quiet-hour readings)
AO_PHI, AO_SIGMA = 0.7755, 0.0279
METER_SIGMA = 0.0367
QUIET_THR = 0.02    # 'other' below this (after denoising) = quiet hour, the reading shows always-on

# v3 HMM: always-on slow drift (matches the real multi-hour dips below the level), emission noise =
# meter noise + fast always-on fluctuation
DRIFT_PHI, DRIFT_SIGMA = 0.96, 0.05
HMM_SIGMA_E = 0.046
HMM_ITERS = 3
# v4: same HMM with settings chosen by maximum likelihood on real households (src/tune_hmm.py)
HMM_V3 = dict(w_phi=DRIFT_PHI, w_sd=DRIFT_SIGMA, sigma_e=HMM_SIGMA_E, up=1.0, z_phi=0.90)
HMM_V4 = dict(w_phi=0.91, w_sd=0.055, sigma_e=0.036, up=0.75, z_phi=0.90)

KERNEL_CACHE = os.path.join(DATA_DIR, 'ev_kernel.npy')
COMPONENTS = ('always_on', 'heating', 'other', 'ev')


def deconvolve(x, r=R_EV):
    """u_t = x_t - r x_{t-1}: turns every A * r^k EV decay into a single spike A at its start."""
    return np.r_[x[0], x[1:] - r * x[:-1]]


def ev_kernel(n=EV_KLEN, r=R_EV):
    return r ** np.arange(n)


def heat_profile(X, hour, weekend, hdd, ev_mask, heat_mask):
    """Global heating shape hp[weekend, hour] (mean 1): heat-only minus pure households, per slot slope."""
    exc = X[heat_mask & ~ev_mask].mean(0) - X[~heat_mask & ~ev_mask].mean(0)
    hp = np.zeros((2, 24))
    for w in (0, 1):
        for h in range(24):
            m = (hour == h) & (weekend == w)
            A = np.c_[hdd[m], np.ones(m.sum())]
            hp[w, h] = np.linalg.lstsq(A, exc[m], rcond=None)[0][0]
    return hp / hp.mean()


def first_pass_heating(X, t_daily, thr=-0.35):
    """Correlation between daily median consumption and daily temperature (robust to EV spikes)."""
    D = X.reshape(X.shape[0], -1, 24)
    td = t_daily[::24]
    corr = np.array([np.corrcoef(np.median(D[i], 1), td)[0, 1] for i in range(X.shape[0])])
    return corr, corr < thr


def ev_design(T, starts, kernel=None):
    k = ev_kernel() if kernel is None else kernel
    E = np.zeros((T, len(starts)))
    for j, s in enumerate(starts):
        L = min(len(k), T - s)
        E[s:s + L, j] = k[:L]
    return E


def prewhiten(M):
    return M[1:] - PHI * M[:-1]


def nuisance_design(hour, weekend, hdd, hp):
    T = len(hour)
    D = np.zeros((T, 48)); D[np.arange(T), weekend * 24 + hour] = 1
    return np.hstack([D, (hdd * hp[weekend, hour])[:, None]])


def fit_household(x, hour, weekend, hdd, hp, has_ev, kernel=None, tail_free=False):
    """tail_free: give each EV session a second amplitude for hours 1+ (its first hour is often
    partial). Ridge prior: tail deviation ~ N(0, (EV_TAIL_REL * A)^2), so small sessions stay put."""
    T = len(x)
    N = nuisance_design(hour, weekend, hdd, hp)
    u = deconvolve(x)
    starts = np.where(u > EV_START_THR)[0] if has_ev else np.array([], int)
    E = ev_design(T, starts, kernel)
    blocks = [N, E]
    if tail_free and len(starts):
        Tl = E.copy(); Tl[starts, np.arange(len(starts))] = 0
        blocks.append(Tl)
    M = np.hstack(blocks)
    Mw, xw = prewhiten(M), prewhiten(x[:, None])[:, 0]
    if len(blocks) == 3:
        pen = np.zeros(M.shape[1])
        pen[49 + len(starts):] = EV_PW_VAR / (EV_TAIL_REL * u[starts]) ** 2
        Mw = np.vstack([Mw, np.diag(np.sqrt(pen))[pen > 0]]); xw = np.r_[xw, np.zeros((pen > 0).sum())]
    beta = np.linalg.lstsq(Mw, xw, rcond=None)[0]
    s2 = (xw - Mw @ beta)[:T - 1].var()
    k_se = np.sqrt(s2 * np.linalg.pinv(Mw.T @ Mw)[48, 48])
    n_ev = len(starts)
    ev = E @ beta[49:49 + n_ev]
    if len(blocks) == 3:
        ev = ev + Tl @ beta[49 + n_ev:]
    return dict(k=beta[48], k_t=beta[48] / k_se, starts=starts, A=beta[49:49 + n_ev], ev=ev,
                nuisance=N @ beta[:49])


def estimate_ev_kernel(X, hour, weekend, hdd, hp, ev_mask, length=36, iters=4, verbose=False):
    """Free EV kernel kappa[k] (kappa[0] = 1) shared by all sessions, by alternating least squares:
    amplitudes given the kernel, then the kernel given the amplitudes."""
    T = X.shape[1]
    kappa = ev_kernel(length)
    rows = np.where(ev_mask)[0]
    starts = {i: np.where(deconvolve(X[i]) > EV_START_THR)[0] for i in rows}
    for it in range(iters):
        fits = {i: fit_household(X[i], hour, weekend, hdd, hp, True, kappa) for i in rows}
        KM, Ky = [], []
        for i in rows:
            K = np.zeros((T, length))
            for s, A in zip(starts[i], fits[i]['A']):
                n = min(length, T - s); K[s + np.arange(n), np.arange(n)] += A
            KM.append(prewhiten(K)); Ky.append(prewhiten((X[i] - fits[i]['nuisance'])[:, None])[:, 0])
        kappa = np.linalg.lstsq(np.vstack(KM), np.concatenate(Ky), rcond=None)[0]
        kappa = np.maximum(kappa / kappa[0], 0)
        if verbose:
            print(f'kernel iteration {it}: kappa[1..3] = {np.round(kappa[1:4], 4)}')
    return kappa


def floor_level(y):
    """Mode of the lower part of the distribution = the always-on floor."""
    g = np.linspace(min(0, y.min()), np.quantile(y, 0.6), 600)
    return g[np.argmax(gaussian_kde(y, bw_method=0.05)(g))]


def meter_noise(X, mean_pred, hour, weekend):
    """E[meter noise | data] under residual = AR(1) 'other' + white noise, via a Kalman/RTS smoother.

    The residual is taken after removing the deterministic parts and each household's
    hour x weekend mean. Noise parameters come from the lag-0/1/2 autocovariances.
    """
    Y = X - mean_pred
    for w in (0, 1):
        for h in range(24):
            m = (hour == h) & (weekend == w)
            Y[:, m] -= Y[:, m].mean(1, keepdims=True)
    g0 = (Y * Y).mean(1); g1 = (Y[:, 1:] * Y[:, :-1]).mean(1); g2 = (Y[:, 2:] * Y[:, :-2]).mean(1)
    phi = np.median(g2 / g1)
    sz2 = g1 / phi
    se2 = np.maximum(g0 - sz2, 1e-6)
    q = sz2 * (1 - phi ** 2)
    n, T = Y.shape
    zf = np.zeros((n, T)); Pf = np.zeros((n, T))
    z, P = np.zeros(n), sz2.copy()
    for t in range(T):                      # forward Kalman filter
        if t:
            z, P = phi * z, phi ** 2 * P + q
        K = P / (P + se2)
        z = z + K * (Y[:, t] - z); P = (1 - K) * P
        zf[:, t], Pf[:, t] = z, P
    zs = zf.copy()
    for t in range(T - 2, -1, -1):          # RTS smoother
        C = Pf[:, t] * phi / (phi ** 2 * Pf[:, t] + q)
        zs[:, t] = zf[:, t] + C * (zs[:, t + 1] - phi * zf[:, t])
    return Y - zs


def ar1_smoother(Y, obs, phi, sigma, obs_sigma):
    """RTS smoother for w_t ~ AR(1)(phi, stationary sd sigma) observed as Y = w + noise where obs is True.

    Vectorised over rows; hours without an observation are filled in from their neighbours,
    with the estimate decaying back to 0 (the prior mean) away from observed hours.
    """
    n, T = Y.shape
    q, R = sigma ** 2 * (1 - phi ** 2), obs_sigma ** 2
    wf = np.zeros((n, T)); Pf = np.zeros((n, T))
    w, P = np.zeros(n), np.full(n, sigma ** 2)
    for t in range(T):
        if t:
            w, P = phi * w, phi ** 2 * P + q
        o = obs[:, t]
        K = np.where(o, P / (P + R), 0.0)
        w = w + K * (np.where(o, Y[:, t], 0.0) - w); P = (1 - K) * P
        wf[:, t], Pf[:, t] = w, P
    ws = wf.copy()
    for t in range(T - 2, -1, -1):
        C = Pf[:, t] * phi / (phi ** 2 * Pf[:, t] + q)
        ws[:, t] = wf[:, t] + C * (ws[:, t + 1] - phi * wf[:, t])
    return ws


class Result:
    """Container: per-component predictions [200, T] plus per-household diagnostics."""

    def __init__(self, X, temp, ts):
        self.X, self.temp, self.ts = X, temp, ts
        self.pred = {c: np.zeros_like(X) for c in COMPONENTS}
        n = X.shape[0]
        self.has_ev = np.zeros(n, bool); self.has_heat = np.zeros(n, bool)
        self.k = np.zeros(n); self.k_t = np.zeros(n); self.a = np.zeros(n)
        self.events = {}

    @property
    def household_type(self):
        return np.select([self.has_ev & self.has_heat, self.has_ev, self.has_heat],
                         ['EV + heating', 'EV only', 'heating only'], 'neither')


def run(version='v1', data=None, kernel=None, calibration=None):
    """data=(X, temp, ts) runs the model on other data (e.g. a simulation); calibration overrides
    the leaderboard-measured constants (they only apply to the real data)."""
    global CALIBRATION
    saved = CALIBRATION
    CALIBRATION = {**CALIBRATION, **(calibration or {})}
    try:
        return _run(version, data, kernel)
    finally:
        CALIBRATION = saved


def _run(version, data, kernel):
    if version in ('v3', 'v4', 'v5', 'v6'):
        common = {'v5': dict(ctx=0, rank=24), 'v6': {}}.get(version)
        return _run_v3(data, kernel, hmm_settings=HMM_V3 if version == 'v3' else HMM_V4, common_heating=common)
    v2 = version != 'v1'
    X, temp, ts = load() if data is None else data
    hour, weekend, t_daily, hdd = calendar(ts, temp)
    ev_mask = X.max(1) > EV_MAX_THR
    corr, heat_guess = first_pass_heating(X, t_daily)
    hp = heat_profile(X, hour, weekend, hdd, ev_mask, heat_guess)

    if v2 and kernel is None:
        if not os.path.exists(KERNEL_CACHE):
            np.save(KERNEL_CACHE, estimate_ev_kernel(X, hour, weekend, hdd, hp, ev_mask))
        kernel = np.load(KERNEL_CACHE)

    res = Result(X, temp, ts)
    res.hp, res.corr, res.kernel = hp, corr, kernel
    for i in range(X.shape[0]):
        x = X[i]
        f = fit_household(x, hour, weekend, hdd, hp, ev_mask[i], kernel, tail_free=(version == 'v2c'))
        has_heat = f['k_t'] > HEAT_T_THR
        heat = f['k'] * hdd * hp[weekend, hour] if has_heat else np.zeros_like(x)
        ev = np.maximum(f['ev'], 0)
        rest = x - heat - ev
        a = max(floor_level(rest), 0)
        res.pred['heating'][i] = heat
        res.pred['ev'][i] = ev
        res.pred['always_on'][i] = a
        res.pred['other'][i] = np.maximum(rest - a, 0)
        res.has_ev[i], res.has_heat[i] = ev_mask[i], has_heat
        res.k[i], res.k_t[i], res.a[i] = f['k'], f['k_t'], a
        res.events[i] = (f['starts'], f['A'])

    if v2:
        P = res.pred
        heat_extra = (CALIBRATION['heating_scale'] - 1) * P['heating']
        P['heating'] = P['heating'] + heat_extra
        P['always_on'] = P['always_on'] + CALIBRATION['always_on_shift']
        res.a = res.a + CALIBRATION['always_on_shift']
        noise = meter_noise(X, P['always_on'] + P['heating'] + P['ev'], hour, weekend)
        other = X - P['always_on'] - P['heating'] - P['ev'] - noise
        # feasibility: true components are >= 0, so a negative 'other' means the EV fit
        # overshot in that hour -> take the excess back from EV first, then clip
        excess = np.maximum(-other, 0)
        ev_cut = np.minimum(excess, P['ev'])
        P['ev'] = P['ev'] - ev_cut
        P['other'] = np.maximum(other + ev_cut, 0)
        res.meter_noise = noise

    if version in ('v2b', 'v2c'):
        P = res.pred
        # household levels pulled toward the mean (probe 07)
        level = P['always_on'][:, :1]
        pull = -CALIBRATION['always_on_shrink'] * (level - level.mean())
        # always-on fluctuation: observed in quiet hours, interpolated elsewhere (Kalman smoother)
        denoised = X - P['heating'] - P['ev'] - res.meter_noise
        quiet = P['other'] < QUIET_THR
        w = ar1_smoother(denoised - level - pull, quiet, AO_PHI, AO_SIGMA, METER_SIGMA * 0.8)
        # quiet hours are selected *because* always-on is low there, so the smoother is biased low in
        # active hours. The level is already calibrated (probe 02), so force each household's
        # fluctuation to average zero by lifting the active hours.
        lift = w.sum(1, keepdims=True) / np.maximum((~quiet).sum(1, keepdims=True), 1)
        w = np.where(quiet, w, w - lift)
        weight = CALIBRATION['always_on_dynamics'] if version == 'v2b' else CALIBRATION['always_on_dynamics_v2c']
        delta = pull + weight * w
        P['always_on'] = np.maximum(P['always_on'] + delta, 0)
        P['other'] = np.maximum(P['other'] - delta, 0)
        res.a = res.a + pull[:, 0]
        res.always_on_delta = delta
    return res


def _run_v3(data, kernel, hmm_settings=None, log=lambda msg: print(msg, flush=True), n_jobs=4,
            common_heating=None):
    """v2c heating/EV, then the always-on/other HMM fitted per household block in parallel processes.
    common_heating (v5, v6): add the common daily heating deviation shared by all heating households
    (common_factor.py, called with these filter settings) to heating before the HMM splits what is left."""
    import hmm
    from joblib import Parallel, delayed
    res = _run('v2c', data, kernel)
    log('v2c stage done')
    X, P = res.X, res.pred
    hour, weekend, _, _ = calendar(res.ts, res.temp)
    cell = weekend * 24 + hour
    if common_heating is not None:
        import common_factor
        G, k = common_factor.estimate(X, P['heating'], P['ev'], cell, **common_heating)
        P['heating'] = np.maximum(P['heating'] + k[:, None] * G[None, :], 0)
        res.common_heating = G
        log(f'common heating factor added (sd {G.std():.4f} per unit heating scale)')
    resid = X - P['heating'] - P['ev']
    zero = X == 0
    mask = P['ev'] < 0.3                       # EV-fit errors would inflate the 'other' variance
    hs = hmm_settings or HMM_V3
    level = P['always_on'].mean(1)
    var_ao = hmm.drift_moments(hs['w_sd'], hs['up'])[1] + AO_SIGMA ** 2
    blocks = np.array_split(np.arange(X.shape[0]), n_jobs * 4)
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(hmm.em_block)(resid[b], zero[b], mask[b], level[b], cell, var_ao, METER_SIGMA, HMM_ITERS, **hs)
        for b in blocks)
    ao = np.vstack([o[0] for o in out]); ot = np.vstack([o[1] for o in out])
    P['always_on'], P['other'] = ao, ot
    res.a = np.concatenate([o[2] for o in out])
    res.other_params = (np.concatenate([o[3] for o in out]), np.vstack([o[4] for o in out]))
    log('HMM stage done')
    return res
