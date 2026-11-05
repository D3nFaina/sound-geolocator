"""Command-line interface: ``sgeo <command> ...``

Typical workflow::

    sgeo spectrogram clip.mp4 --fmax 1500 --out spec.png         # look for Doppler lines
    sgeo extract clip.mp4 --seed 42,310 --seed 118,265 --out curves.json --plot picked.png
    sgeo fit curves.json                                          # sanity-check each pass
    sgeo locate curves.json tracks.csv --center 45.46,9.19 \
         --start 2025-06-01T14:03:00Z --uncertainty 300 --radius 25 --plot map.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import DEFAULT_SPEED_OF_SOUND


def _time(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def _pair(s):
    a, b = s.split(",")
    return float(a), float(b)


def _spec(args):
    from .audio import load_audio
    from .spectrogram import compute_spectrogram, enhance

    x, sr = load_audio(args.input, sr=args.sr)
    spec = compute_spectrogram(x, sr, window_s=args.window, hop_s=args.hop, fmin=args.fmin, fmax=args.fmax)
    return enhance(spec) if not args.raw else spec


def _add_spec_args(p):
    p.add_argument("input", help="audio or video file")
    p.add_argument("--sr", type=int, default=8000, help="resample rate (Hz); fmax <= sr/2")
    p.add_argument("--window", type=float, default=1.0, help="STFT window seconds (1 s -> 1 Hz bins)")
    p.add_argument("--hop", type=float, default=0.25, help="STFT hop seconds")
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=2000.0)
    p.add_argument("--raw", action="store_true", help="skip enhancement")


def cmd_spectrogram(args):
    from .plot import plot_spectrogram

    spec = _spec(args)
    plot_spectrogram(spec, args.out, fmax=args.fmax)
    print(f"wrote {args.out}  ({spec.times[-1]:.0f} s, {spec.freqs[0]:.0f}-{spec.freqs[-1]:.0f} Hz)")


def cmd_extract(args):
    from .extract import clean_curve, save_curves, trace_from_seed, track_ridge

    spec = _spec(args)
    curves = []
    for k, seed in enumerate(args.seed or []):
        v = [float(x) for x in seed.split(",")]
        t, f = v[0], v[1]
        tmin, tmax = (v[2], v[3]) if len(v) == 4 else (None, None)
        cv = trace_from_seed(spec, t, f, tmin, tmax, max_slope_hz_s=args.max_slope,
                             min_strength_db=args.min_db, name=f"seed{k}")
        curves.append(clean_curve(cv))
    for k, box in enumerate(args.box or []):
        t0, t1, f0, f1 = (float(v) for v in box.split(","))
        cv = track_ridge(spec, f0, f1, t0, t1, max_slope_hz_s=args.max_slope, min_strength_db=args.min_db,
                         name=f"box{k}")
        curves.append(clean_curve(cv))
    if not curves:
        raise SystemExit("give at least one --seed T,F or --box T0,T1,F0,F1")
    save_curves(curves, args.out)
    for cv in curves:
        ok = np.isfinite(cv.f)
        print(f"{cv.name}: {ok.sum()} points, t {cv.t[ok][0]:.1f}-{cv.t[ok][-1]:.1f} s, "
              f"f {np.nanmin(cv.f):.1f}-{np.nanmax(cv.f):.1f} Hz")
    if args.plot:
        from .plot import plot_curves

        plot_curves(spec, curves, args.plot, fmax=args.fmax)
    print(f"wrote {args.out}")


def cmd_fit(args):
    from .doppler import fit_flyover
    from .extract import load_curves

    for cv in load_curves(args.curves):
        r = fit_flyover(cv.t, cv.f, c=args.c)
        print(f"{cv.name}: f_src={r['f0']:.2f} Hz  closest approach at t={r['t_cpa']:.1f} s  "
              f"speed={r['speed']:.0f} m/s ({r['speed'] * 1.944:.0f} kt)  "
              f"slant range={r['d_cpa']:.0f} m  rms={r['rms_hz']:.2f} Hz")


def cmd_fetch(args):
    from .flights import fetch_opensky_states, save_tracks_csv

    lat, lon = _pair(args.center)
    t0 = _time(args.start)
    tracks = fetch_opensky_states(lat, lon, args.radius, t0 - args.margin, t0 + args.duration + args.margin, args.step)
    save_tracks_csv(tracks, args.out)
    print(f"wrote {len(tracks)} tracks to {args.out}")


def cmd_locate(args):
    from .extract import load_curves
    from .flights import load_tracks
    from .locate import locate

    curves = load_curves(args.curves)
    if args.only:
        keep = set(args.only.split(","))
        curves = [c for c in curves if c.name in keep]
    tracks = load_tracks(args.tracks)
    res = locate(curves, tracks, _pair(args.center), _time(args.start), args.uncertainty, args.radius,
                 args.step, args.refine, args.alt, args.dt, args.c)
    start = datetime.fromtimestamp(res.t_start, timezone.utc).isoformat()
    print(f"\nestimated location: {res.lat:.5f}, {res.lon:.5f}")
    print(f"recording started: {start}")
    print(f"https://www.openstreetmap.org/?mlat={res.lat:.5f}&mlon={res.lon:.5f}#map=14/{res.lat:.5f}/{res.lon:.5f}")
    for a in res.assignments:
        print(f"  {a.curve} -> {a.flight}: source tone {a.f_src:.2f} Hz, misfit {a.rms_hz:.2f} Hz")
    out = {"lat": res.lat, "lon": res.lon, "t_start": res.t_start, "t_start_iso": start, "cost": res.cost,
           "assignments": [a.__dict__ for a in res.assignments]}
    Path(args.out).write_text(json.dumps(out, indent=2))
    if args.plot:
        from .plot import plot_cost_map

        plot_cost_map(res, args.plot, tracks=tracks)


def cmd_demo(args):
    import runpy

    import sys

    demo = Path(__file__).resolve().parent.parent / "examples" / "demo_synthetic.py"
    sys.argv = [str(demo), args.out]
    runpy.run_path(str(demo), run_name="__main__")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sgeo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("spectrogram", help="render the enhanced spectrogram to pick curves from")
    _add_spec_args(p)
    p.add_argument("--out", default="spectrogram.png")
    p.set_defaults(func=cmd_spectrogram)

    p = sub.add_parser("extract", help="extract Doppler curves from seeds or boxes")
    _add_spec_args(p)
    p.add_argument("--seed", action="append",
                   help="T,F[,TMIN,TMAX]: a point on a line (seconds, Hz), optional time limits; repeatable")
    p.add_argument("--box", action="append", help="T0,T1,F0,F1: box containing one line; repeatable")
    p.add_argument("--max-slope", type=float, default=8.0, help="steepest line to follow, Hz per second")
    p.add_argument("--min-db", type=float, default=3.0, help="min line strength above background")
    p.add_argument("--out", default="curves.json")
    p.add_argument("--plot", help="also save a spectrogram PNG with the picked curves")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("fit", help="fit a straight-flyover model to each curve (speed, range, CPA time)")
    p.add_argument("curves")
    p.add_argument("--c", type=float, default=DEFAULT_SPEED_OF_SOUND)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("fetch-opensky", help="download recent ADS-B tracks from the OpenSky REST API")
    p.add_argument("--center", required=True, help="LAT,LON")
    p.add_argument("--radius", type=float, default=30.0, help="km")
    p.add_argument("--start", required=True, help="recording start, unix s or ISO 8601")
    p.add_argument("--duration", type=float, default=600.0, help="recording length, s")
    p.add_argument("--margin", type=float, default=600.0, help="extra time either side, s")
    p.add_argument("--step", type=float, default=10.0, help="polling step, s")
    p.add_argument("--out", default="tracks.csv")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("locate", help="search for the recording position")
    p.add_argument("curves")
    p.add_argument("tracks", help="CSV or JSON flight tracks")
    p.add_argument("--center", required=True, help="LAT,LON to search around")
    p.add_argument("--radius", type=float, default=20.0, help="search half-width, km")
    p.add_argument("--start", required=True, help="approx recording start, unix s or ISO 8601")
    p.add_argument("--uncertainty", type=float, default=300.0, help="+- seconds on --start")
    p.add_argument("--step", type=float, default=1000.0, help="coarse grid step, m")
    p.add_argument("--refine", type=int, default=2)
    p.add_argument("--alt", type=float, default=0.0, help="observer altitude, m (same datum as tracks)")
    p.add_argument("--dt", type=float, default=0.5, help="time resolution, s")
    p.add_argument("--c", type=float, default=DEFAULT_SPEED_OF_SOUND, help="speed of sound, m/s")
    p.add_argument("--only", help="comma-separated curve names to use")
    p.add_argument("--out", default="location.json")
    p.add_argument("--plot", help="save the cost map PNG")
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser("demo", help="run the synthetic end-to-end demo")
    p.add_argument("--out", default="demo_out")
    p.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
