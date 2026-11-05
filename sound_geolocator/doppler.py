"""Doppler physics.

A stationary listener at O hears, at reception time t, sound the aircraft
emitted at time tau, where  t - tau = |P(tau) - O| / c.  The observed
frequency is

    f_obs = f_src * c / (c - v(tau) . n),   n = (O - P(tau)) / |O - P(tau)|

so we model the *ratio* r(t) = f_obs / f_src, which is independent of the
(unknown) source tone frequency.
"""

from __future__ import annotations

import numpy as np

from . import DEFAULT_SPEED_OF_SOUND


def emission_times(track, observer_ecef, t_recv, c=DEFAULT_SPEED_OF_SOUND, iters=6):
    """Solve the retarded-time equation by fixed-point iteration.

    observer_ecef: (..., 3) array (broadcasts against t_recv).
    Returns tau with the same broadcast shape as (observer, t_recv).
    """
    t_recv = np.asarray(t_recv, float)
    O = np.asarray(observer_ecef, float)
    tau = np.broadcast_to(t_recv, np.broadcast_shapes(O.shape[:-1], t_recv.shape)).copy()
    for _ in range(iters):
        P = track.position(tau.ravel()).reshape(tau.shape + (3,))
        tau = t_recv - np.linalg.norm(P - O, axis=-1) / c
    return tau


def doppler_ratio(track, observer_ecef, t_recv, c=DEFAULT_SPEED_OF_SOUND, return_extra=False):
    """Observed/emitted frequency ratio heard at ``observer_ecef`` at times ``t_recv`` (unix s).

    Broadcasts: observer (G, 3) with t_recv (T,) -> pass observer[:, None, :] to get (G, T).
    """
    O = np.asarray(observer_ecef, float)
    tau = emission_times(track, O, t_recv, c)
    flat = tau.ravel()
    P = track.position(flat).reshape(tau.shape + (3,))
    V = track.velocity(flat).reshape(tau.shape + (3,))
    d = O - P
    rng = np.linalg.norm(d, axis=-1)
    vr = np.sum(V * d, axis=-1) / rng  # speed toward observer
    ratio = c / (c - vr)
    if return_extra:
        return ratio, tau, rng, vr
    return ratio


def straight_flyover_ratio(t, t_cpa, v, d_cpa, c=DEFAULT_SPEED_OF_SOUND):
    """Closed-form ratio for a constant-velocity straight pass (for curve diagnostics).

    t is reception time; t_cpa is the *emission* time of closest approach.
    """
    t = np.asarray(t, float)
    tau = t.copy()
    for _ in range(8):
        x = v * (tau - t_cpa)
        tau = t - np.hypot(x, d_cpa) / c
    x = v * (tau - t_cpa)
    vr = -v * x / np.hypot(x, d_cpa)
    return c / (c - vr)


def fit_flyover(t, f, c=DEFAULT_SPEED_OF_SOUND):
    """Fit f0, t_cpa, speed and closest-approach distance to one extracted curve.

    Useful as a sanity check that a picked line really is a Doppler pass, and
    for a quick range estimate without any flight data.
    """
    from scipy.optimize import least_squares

    t = np.asarray(t, float)
    f = np.asarray(f, float)
    ok = np.isfinite(f)
    t, f = t[ok], f[ok]
    # initial guess: steepest descent point
    df = np.gradient(f, t)
    i0 = int(np.argmin(df))
    hi, lo = np.percentile(f, 95), np.percentile(f, 5)
    f0 = np.sqrt(hi * lo)
    v0 = np.clip(c * (hi - lo) / (hi + lo), 20, 300)

    def res(p):
        f0_, tc, v, d = p
        return f0_ * straight_flyover_ratio(t, tc, v, d, c) - f

    best = None
    for d0 in (300.0, 1000.0, 3000.0, 8000.0):
        r = least_squares(res, [f0, t[i0], v0, d0], bounds=([0, t[0] - 300, 5, 30], [np.inf, t[-1] + 300, 340, 50000]),
                          loss="soft_l1", f_scale=1.0)  # robust: a few stray points barely matter
        if best is None or r.cost < best.cost:
            best = r
    f0_, tc, v, d = best.x
    r = res(best.x)
    rms = np.sqrt(np.mean(r ** 2))
    out = {"f0": f0_, "t_cpa": tc, "speed": v, "d_cpa": d, "rms_hz": rms}
    out.update(_fit_errors(res, best.x, r, t))
    return out


def _fit_errors(res, x, r, t):
    """1-sigma errors of (f0, t_cpa, speed, d_cpa) from the curvature of the misfit.

    Covariance = s^2 (J^T J)^-1 with s^2 the residual variance, using only the
    points the robust fit kept (|residual| < 3 rms-of-the-core). Neighbouring
    spectrogram frames overlap and the aircraft wobbles, so residuals are
    correlated; the errors are inflated by sqrt(n / n_eff), with n_eff from the
    lag-1 autocorrelation of the residuals.
    """
    names = ("f0", "t_cpa", "speed", "d_cpa")
    core = np.abs(r) < 3 * max(np.median(np.abs(r)) * 1.4826, 1e-6)
    n = int(core.sum())
    if n <= len(x) + 2:
        return {k + "_err": float("nan") for k in names}
    J = np.empty((len(r), len(x)))
    for k in range(len(x)):
        h = max(abs(x[k]) * 1e-6, 1e-4)
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        J[:, k] = (res(xp) - res(xm)) / (2 * h)
    J, rc = J[core], r[core]
    s2 = float(np.sum(rc ** 2) / (n - len(x)))
    o = np.argsort(t[core])
    rs = rc[o]
    rho = float(np.clip(np.corrcoef(rs[:-1], rs[1:])[0, 1], 0, 0.98)) if n > 5 else 0.0
    inflate = (1 + rho) / (1 - rho)
    try:
        cov = np.linalg.inv(J.T @ J) * s2 * inflate
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(x), np.nan)
    return {k + "_err": float(e) for k, e in zip(names, err)}
