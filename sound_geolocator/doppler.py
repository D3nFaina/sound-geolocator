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
        r = least_squares(res, [f0, t[i0], v0, d0], bounds=([0, t[0] - 300, 5, 30], [np.inf, t[-1] + 300, 340, 50000]))
        if best is None or r.cost < best.cost:
            best = r
    f0_, tc, v, d = best.x
    rms = np.sqrt(np.mean(best.fun ** 2))
    return {"f0": f0_, "t_cpa": tc, "speed": v, "d_cpa": d, "rms_hz": rms}
