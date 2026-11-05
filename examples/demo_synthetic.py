"""End-to-end demo on a synthetic recording with a known answer.

    python examples/demo_synthetic.py [outdir]
"""

import sys
import time
from pathlib import Path

import numpy as np

from sound_geolocator.extract import clean_curve, trace_from_seed, track_ridge
from sound_geolocator.geo import haversine_m
from sound_geolocator.locate import locate
from sound_geolocator.spectrogram import compute_spectrogram, enhance
from sound_geolocator.synth import demo_scenario, render_recording

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_out")
out.mkdir(exist_ok=True)

flights, TRUE, T0 = demo_scenario()
lat0, lon0 = TRUE[0], TRUE[1]

x, sr = render_recording(flights[:2], TRUE, T0, 360.0, sr=4000, noise=0.02)
spec = enhance(compute_spectrogram(x, sr, window_s=1.0, hop_s=0.5, fmax=600))

# "Select" the two Doppler curves on the fundamental tones: one with a box,
# one by clicking a single point on the line and letting the tracer follow it.
c1 = clean_curve(track_ridge(spec, 60, 145, 40, 170, name="pass A"))
c2 = clean_curve(trace_from_seed(spec, 200, 125, name="pass B"))

try:
    from sound_geolocator.plot import plot_curves, plot_cost_map
    plot_curves(spec, [c1, c2], out / "spectrogram.png", fmax=320)
except ImportError:
    plot_curves = None

# We only "know" the area to ~15 km and the start time to +-2 minutes.
guess = (lat0 + 0.08, lon0 - 0.1)
tic = time.time()
res = locate([c1, c2], flights, guess, t_start_guess=T0 + 75, t_uncertainty=120,
             radius_km=20, grid_step_m=1000, refine_levels=2, observer_alt_m=TRUE[2])
err = haversine_m(res.lat, res.lon, TRUE[0], TRUE[1])
print(f"estimate {res.lat:.5f}, {res.lon:.5f}  true {TRUE[0]:.5f}, {TRUE[1]:.5f}  error {err:.0f} m")
print(f"start time error {res.t_start - T0:+.1f} s, runtime {time.time() - tic:.1f} s")
for a in res.assignments:
    print(f"  {a.curve}: {a.flight}  f_src={a.f_src:.2f} Hz  rms={a.rms_hz:.3f} Hz")
if plot_curves:
    plot_cost_map(res, out / "cost_map.png", truth=TRUE[:2], tracks=flights)
