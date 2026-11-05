"""Local web GUI for the whole pipeline.

    sgeo gui            # then open http://127.0.0.1:8765

Single-user and local: state lives in memory in this process.
"""

from __future__ import annotations

import io
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from ..audio import load_audio
from ..doppler import fit_flyover
from ..extract import DopplerCurve, clean_curve, load_curves, save_curves, trace_from_seed, track_ridge
from ..flights import Track, fetch_opensky_states, load_tracks
from ..spectrogram import compute_spectrogram, enhance

STATIC = Path(__file__).parent / "static"


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(v):
    v = str(v).strip()
    try:
        return float(v)
    except ValueError:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.audio = None  # (x, sr)
        self.audio_name = ""
        self.raw = None
        self.spec = None
        self.params = dict(window=1.0, hop=0.25, fmin=20.0, fmax=1000.0, sr=8000)
        self.curves: list[DopplerCurve] = []
        self.tracks: list[Track] = []
        self.result = None
        self.truth = None  # demo only

    def compute_spec(self):
        p = self.params
        x, sr = self.audio
        fmax = min(p["fmax"], sr / 2)
        self.raw = compute_spectrogram(x, sr, window_s=p["window"], hop_s=p["hop"], fmin=p["fmin"], fmax=fmax)
        self.spec = enhance(self.raw)


S = State()


def _curve_json(cv: DopplerCurve, i: int):
    ok = np.isfinite(cv.f)
    info = {"name": cv.name or f"curve{i}", "t": [round(float(v), 3) for v in cv.t[ok]],
            "f": [round(float(v), 3) for v in cv.f[ok]], "n": int(ok.sum())}
    try:
        r = fit_flyover(cv.t, cv.f)
        info["fit"] = {k: round(float(v), 2) for k, v in r.items()}
    except Exception:
        info["fit"] = None
    return info


def _spec_json():
    sp = S.spec
    return {"loaded": True, "name": S.audio_name, "t0": float(sp.times[0]), "t1": float(sp.times[-1]),
            "f0": float(sp.freqs[0]), "f1": float(sp.freqs[-1]), "params": S.params,
            "duration": float(len(S.audio[0]) / S.audio[1])}


def create_app():
    app = Flask(__name__, static_folder=None)

    @app.errorhandler(Exception)
    def err(e):
        traceback.print_exc()
        return jsonify(error=str(e)), 400

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:p>")
    def static_files(p):
        return send_from_directory(STATIC, p)

    @app.get("/api/state")
    def state():
        return jsonify(spec=_spec_json() if S.spec is not None else {"loaded": False},
                       curves=[_curve_json(c, i) for i, c in enumerate(S.curves)],
                       tracks=_tracks_json(), truth=S.truth)

    @app.post("/api/audio")
    def upload_audio():
        f = request.files["file"]
        suffix = Path(f.filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            path = tmp.name
        sr = int(request.form.get("sr", S.params["sr"]))
        x, sr = load_audio(path, sr=sr)
        Path(path).unlink(missing_ok=True)
        with S.lock:
            S.audio, S.audio_name = (x, sr), f.filename
            S.params["sr"] = sr
            S.curves, S.result, S.truth = [], None, None
            S.compute_spec()
        return jsonify(spec=_spec_json())

    @app.post("/api/params")
    def params():
        if S.audio is None:
            raise ValueError("load a recording first")
        d = request.get_json()
        with S.lock:
            for k in ("window", "hop", "fmin", "fmax"):
                if k in d:
                    S.params[k] = float(d[k])
            S.compute_spec()
        return jsonify(spec=_spec_json())

    @app.get("/api/spectrogram.png")
    def spectrogram_png():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        which = request.args.get("view", "enhanced")
        sp = S.spec if which == "enhanced" else S.raw
        p = sp.power_db
        lo, hi = np.percentile(p, 5), np.percentile(p, 99.7)
        img = np.clip((p - lo) / max(hi - lo, 1e-9), 0, 1)[::-1]
        buf = io.BytesIO()
        plt.imsave(buf, img, cmap="magma", format="png")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", max_age=0)

    @app.post("/api/curves")
    def add_curve():
        if S.spec is None:
            raise ValueError("load a recording first")
        d = request.get_json()
        name = d.get("name") or f"pass {len(S.curves) + 1}"
        rise = None if d.get("allow_rise") else 0.5
        if d["mode"] == "seed":
            cv = trace_from_seed(S.spec, float(d["t"]), float(d["f"]), d.get("tmin"), d.get("tmax"),
                                 max_rise_hz_s=rise, name=name)
        else:
            cv = track_ridge(S.spec, float(d["f0"]), float(d["f1"]), float(d["t0"]), float(d["t1"]),
                             max_rise_hz_s=rise, name=name)
        cv = clean_curve(cv)
        if np.isfinite(cv.f).sum() < 10:
            raise ValueError("no clear line found there; try clicking right on a bright line")
        with S.lock:
            S.curves.append(cv)
        return jsonify(curves=[_curve_json(c, i) for i, c in enumerate(S.curves)])

    @app.delete("/api/curves/<int:i>")
    def delete_curve(i):
        with S.lock:
            S.curves.pop(i)
        return jsonify(curves=[_curve_json(c, i) for i, c in enumerate(S.curves)])

    @app.get("/api/curves.json")
    def export_curves():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            save_curves(S.curves, tmp.name)
        return send_file(tmp.name, as_attachment=True, download_name="curves.json")

    @app.post("/api/curves/import")
    def import_curves():
        f = request.files["file"]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            f.save(tmp.name)
            with S.lock:
                S.curves = load_curves(tmp.name)
        return jsonify(curves=[_curve_json(c, i) for i, c in enumerate(S.curves)])

    def _tracks_json():
        out = []
        for tr in S.tracks:
            step = max(1, len(tr.t) // 400)
            out.append({"id": tr.ident, "label": tr.label, "type": tr.typecode, "category": tr.category,
                        "t0": float(tr.t[0]), "t1": float(tr.t[-1]),
                        "path": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(tr.lat[::step], tr.lon[::step])]})
        return out

    @app.post("/api/tracks")
    def upload_tracks():
        f = request.files["file"]
        suffix = Path(f.filename).suffix or ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            tracks = load_tracks(tmp.name)
        with S.lock:
            S.tracks = tracks
        return jsonify(tracks=_tracks_json())

    @app.post("/api/tracks/opensky")
    def opensky():
        d = request.get_json()
        t0 = _parse_time(d["start"])
        dur = S.audio and len(S.audio[0]) / S.audio[1] or 600
        tracks = fetch_opensky_states(float(d["lat"]), float(d["lon"]), float(d.get("radius", 30)),
                                      t0 - 600, t0 + dur + 600, float(d.get("step", 10)))
        with S.lock:
            S.tracks = tracks
        return jsonify(tracks=_tracks_json())

    @app.post("/api/dem")
    def upload_dem():
        from ..elevation import DEM

        f = request.files["file"]
        suffix = Path(f.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            dem = DEM.load(tmp.name) if suffix == ".npz" else DEM.from_geotiff(tmp.name)
        S.dem_file = dem
        return jsonify(ok=True, name=f.filename, lat=[float(dem.lats.min()), float(dem.lats.max())],
                       lon=[float(dem.lons.min()), float(dem.lons.max())])

    @app.post("/api/locate")
    def run_locate():
        from ..aircraft_types import filter_tracks
        from ..elevation import DEM
        from ..locate import locate

        d = request.get_json()
        if len(S.curves) < 1:
            raise ValueError("pick at least one curve (two or more to fix a point)")
        if not S.tracks:
            raise ValueError("load flight tracks first")
        tracks = filter_tracks(S.tracks, d.get("types") or ("prop", "jet", "heli", "unknown"))
        if not tracks:
            raise ValueError("no flights left after the aircraft-type filter")
        use = d.get("use")
        curves = [c for i, c in enumerate(S.curves) if use is None or i in use]
        lat, lon, radius = float(d["lat"]), float(d["lon"]), float(d.get("radius", 20))
        geoid = float(d.get("geoid", 0) or 0)
        alt = float(d.get("alt", 0))
        dem_choice = d.get("dem") or "none"
        if dem_choice == "file":
            if getattr(S, "dem_file", None) is None:
                raise ValueError("upload a DEM file first")
            alt = S.dem_file
            alt.geoid_offset_m = geoid
        elif dem_choice != "none":
            cache = Path(tempfile.gettempdir()) / f"sgeo_dem_{dem_choice}_{lat:.3f}_{lon:.3f}_{radius:.0f}.npz"
            alt = DEM.from_opentopodata(lat, lon, radius, dataset=dem_choice, cache=cache, geoid_offset_m=geoid)
        rec_s = len(S.audio[0]) / S.audio[1] if S.audio else None
        silence = float(d.get("silence_km") or 0) or None
        res = locate(curves, tracks, (lat, lon), _parse_time(d["start"]),
                     float(d.get("uncertainty", 300)), radius, float(d.get("step", 1000)),
                     int(d.get("refine", 2)), alt, float(d.get("dt", 0.5)), float(d.get("c", 340)),
                     cpa_slack_s=float(d.get("cpa_slack", 2)), fit_gate=bool(d.get("fit_gate", True)),
                     silence_km=silence, recording_s=rec_s)
        S.result = res
        glat, glon = res.grid_lat, res.grid_lon
        return jsonify(lat=res.lat, lon=res.lon, t_start=res.t_start, t_start_iso=_iso(res.t_start),
                       cost=res.cost, observer_alt=res.observer_alt,
                       assignments=[a.__dict__ for a in res.assignments], candidates=res.candidates,
                       n_tracks=len(tracks),
                       bounds=[[float(glat.min()), float(glon.min())], [float(glat.max()), float(glon.max())]],
                       truth=S.truth)

    @app.get("/api/costmap.png")
    def costmap_png():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.log10(S.result.cost_map + 1e-12)
        lo, hi = np.percentile(cm, 0), np.percentile(cm, 60)
        z = np.clip((hi - cm) / max(hi - lo, 1e-9), 0, 1)
        rgba = plt.get_cmap("inferno")(z)
        rgba[..., 3] = z ** 1.5 * 0.85
        buf = io.BytesIO()
        plt.imsave(buf, rgba[::-1], format="png")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", max_age=0)

    @app.post("/api/demo")
    def demo():
        from ..synth import demo_scenario, render_recording

        flights, truth, t0 = demo_scenario()
        x, sr = render_recording(flights[:2], truth, t0, 360.0, sr=4000)
        with S.lock:
            S.audio, S.audio_name = (x, sr), "synthetic demo"
            S.params.update(sr=4000, fmax=400.0)
            S.curves, S.result = [], None
            S.tracks = flights
            S.truth = {"lat": truth[0], "lon": truth[1], "t_start": t0}
            S.compute_spec()
        guess = {"lat": round(truth[0] + 0.08, 4), "lon": round(truth[1] - 0.1, 4), "start": _iso(t0 + 75),
                 "uncertainty": 120, "radius": 20, "alt": truth[2]}
        return jsonify(spec=_spec_json(), tracks=_tracks_json(), guess=guess)

    return app


def main(host="127.0.0.1", port=8765, open_browser=True):
    app = create_app()
    url = f"http://{host}:{port}"
    print(f"Sound Geolocator GUI on {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)
