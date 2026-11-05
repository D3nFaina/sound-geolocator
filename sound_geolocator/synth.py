"""Synthetic scenarios: fake flights and the recording a listener would have made.

Used for tests and for checking the whole pipeline end to end without real data.
"""

from __future__ import annotations

import numpy as np

from . import DEFAULT_SPEED_OF_SOUND
from .doppler import emission_times
from .flights import Track
from .geo import enu_offsets_to_lla, lla_to_ecef


def straight_track(ident, lat0, lon0, t0, east0, north0, heading_deg, speed, alt, duration=900.0,
                   report_dt=1.0, noise_m=0.0, climb=0.0, rng=None, callsign="", typecode=""):
    """A constant-heading flight starting at (east0, north0) metres from (lat0, lon0)."""
    rng = rng or np.random.default_rng(0)
    t = t0 + np.arange(0, duration, report_dt)
    h = np.radians(heading_deg)
    e = east0 + speed * np.sin(h) * (t - t0) + rng.normal(0, noise_m, t.size)
    n = north0 + speed * np.cos(h) * (t - t0) + rng.normal(0, noise_m, t.size)
    a = alt + climb * (t - t0)
    lat, lon = enu_offsets_to_lla(lat0, lon0, e, n)
    return Track(ident, t, lat, lon, a, callsign or ident.upper(), typecode)


def render_recording(tracks, observer, t_start, duration, sr=4000, tones=None, noise=0.02,
                     c=DEFAULT_SPEED_OF_SOUND, ref_range=1000.0, seed=1, hum_hz=50.0):
    """Render what a microphone at ``observer`` (lat, lon, alt) hears.

    tones: per-track list of (freq_hz, amplitude) harmonics; default a blade
    tone at 90-130 Hz with three harmonics. The phase is integrated in
    emission time, so the Doppler shift is exact.
    """
    rng = np.random.default_rng(seed)
    O = lla_to_ecef(*observer)
    t = t_start + np.arange(int(duration * sr)) / sr
    x = np.zeros(t.size)
    for k, tr in enumerate(tracks):
        tt = tones[k] if tones else [(90 + 13 * k, 1.0), (180 + 26 * k, 0.6), (270 + 39 * k, 0.4)]
        tau = emission_times(tr, O, t, c)
        m = tr.covered(tau)  # only where the sound was emitted inside the track
        if not m.any():
            continue
        tau = tau[m]
        P = tr.position(tau)
        rng_m = np.linalg.norm(P - O, axis=-1)
        amp = ref_range / np.maximum(rng_m, 50.0)
        sig = sum(a * np.sin(2 * np.pi * f * (tau - tr.t[0]) + rng.uniform(0, 6.3)) for f, a in tt)
        x[m] += amp * sig
    x /= max(1e-9, np.max(np.abs(x)))
    x += noise * rng.standard_normal(t.size)
    if hum_hz:
        x += 0.05 * np.sin(2 * np.pi * hum_hz * (t - t_start))
    return x.astype(np.float32), sr


def demo_scenario():
    """The scenario used by the demo and tests.

    Returns (flights, truth (lat, lon, alt), t_start). Two aircraft pass a
    listener in Milan ~100 s and ~200 s into a 6-minute recording; a third
    flies far away and is never heard.
    """
    lat0, lon0, alt0 = 45.4642, 9.1900, 120.0
    t0 = 1_760_000_000.0
    flights = [
        # northbound at 120 m/s, 1500 m up, closest ~1.5 km east of the listener at t0+100
        straight_track("a1b2c3", lat0, lon0, t0 - 400, 1500, -60000, 0, 120, 1500, duration=1400, noise_m=8,
                       typecode="AT76"),
        # eastbound at 140 m/s, 2500 m up, closest ~2 km north at t0+200
        straight_track("d4e5f6", lat0, lon0, t0 - 400, -84000, 2000, 90, 140, 2500, duration=1400, noise_m=8,
                       typecode="DH8D"),
        # distractor
        straight_track("0f0f0f", lat0, lon0, t0 - 400, 15000, -40000, 20, 200, 9000, duration=1400, noise_m=8,
                       typecode="A320"),
    ]
    return flights, (lat0, lon0, alt0), t0
