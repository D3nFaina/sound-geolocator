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
from ..extract import (DopplerCurve, clean_curve, digitise_along, isolate_curve, load_curves, save_curves,
                       trace_from_seed, track_ridge)
from ..flights import Track, fetch_opensky_states, load_tracks
from ..spectrogram import compute_spectrogram, enhance

STATIC = Path(__file__).parent / "static"


# continental United States: lat_min, lat_max, lon_min, lon_max
CONUS_BBOX = (24.5, 49.5, -125.0, -66.5)

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
        self.clim = {}
        self.clean = None
        self.progress = {"stage": "idle", "fraction": 0.0}
        self.dem_file = None
        self.version = 0  # bumped on every change of the curves, so other windows can follow
        self.day = None  # DayArchive
        self.day_build = None  # (path, subprocess) while converting
        self.overlays = {}  # key -> predicted flight curves for one place and start time
        self.area = None  # (candidates, curves) of the last 'places that hear the passes' search
        self.flow = {}  # guided search: region, passes, candidates, ranking, fine grid
        self.terrain = None  # TerrainTiles, created on first use

    def compute_spec(self):
        p = self.params
        x, sr = self.audio
        fmax = min(p["fmax"], sr / 2)
        self.progress = {"stage": "spectrogram", "fraction": 0.0}

        def prog(fr):
            self.progress = {"stage": "spectrogram", "fraction": round(fr * 0.6, 3)}

        self.raw = compute_spectrogram(x, sr, window_s=p["window"], hop_s=p["hop"], fmin=p["fmin"], fmax=fmax,
                                       progress=prog)
        self.progress = {"stage": "enhancing", "fraction": 0.65}
        self.spec = enhance(self.raw)
        self.clean = None  # 'lines removed' view, made on first use
        # colour limits from the whole recording, so the view doesn't flicker when panning
        self.clim = {}
        for k, sp in (("enhanced", self.spec), ("raw", self.raw)):
            sub = sp.power_db[:, ::max(1, sp.power_db.shape[1] // 2000)]
            self.clim[k] = (float(np.percentile(sub, 5)), float(np.percentile(sub, 99.7)))
        self.progress = {"stage": "done", "fraction": 1.0}


S = State()


def _clean_spec():
    """The spectrogram with lasting lines removed (30 s running median per frequency), made once."""
    if S.clean is None:
        S.progress = {"stage": "removing steady lines", "fraction": 0.3}
        sp = enhance(S.raw, hum_window_s=30.0)
        sub = sp.power_db[:, ::max(1, sp.power_db.shape[1] // 2000)]
        S.clim["clean"] = (float(np.percentile(sub, 5)), float(np.percentile(sub, 99.7)))
        S.clean = sp
        S.progress = {"stage": "done", "fraction": 1.0}
    return S.clean


def _curve_json(cv: DopplerCurve, i: int):
    from ..doppler import straight_flyover_ratio

    ok = np.isfinite(cv.f)
    info = {"name": cv.name or f"curve{i}", "t": [round(float(v), 3) for v in cv.t[ok]],
            "f": [round(float(v), 3) for v in cv.f[ok]], "n": int(ok.sum()),
            "show": cv.meta.get("show", True), "use": cv.meta.get("use", True),
            "guide": [cv.meta["guide_t"], cv.meta["guide_f"]] if "guide_t" in cv.meta else None}
    fit = cv.meta.get("fit")
    if fit is None and "fitted" not in cv.meta:  # curves saved by older versions: fit on the fly
        try:
            fit = {k: float(v) for k, v in fit_flyover(cv.t, cv.f).items()}
        except Exception:
            fit = None
        cv.meta["fit"] = fit
    info["fit"] = None if fit is None else {k: (round(v, 3) if np.isfinite(v) else None) for k, v in fit.items()}
    if fit:
        tm = np.linspace(cv.t[ok][0] - 20, cv.t[ok][-1] + 20, 300)
        fm = fit["f0"] * straight_flyover_ratio(tm, fit["t_cpa"], fit["speed"], fit["d_cpa"])
        info["model_t"] = [round(float(v), 2) for v in tm]
        info["model_f"] = [round(float(v), 2) for v in fm]
    return info


def _fits_path():
    """Where the picked curves of the current recording are kept between sessions."""
    d = Path.home() / ".sound_geolocator" / "fits"
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_. " else "_" for c in (S.audio_name or "unnamed"))
    return d / f"{safe}.json"


def _save_fits():
    S.version += 1
    if S.audio_name:
        save_curves(S.curves, _fits_path())


def _load_fits():
    p = _fits_path()
    S.curves = load_curves(p) if p.exists() else []


def _curves_payload():
    return [_curve_json(c, i) for i, c in enumerate(S.curves)]


def _spec_json():
    sp = S.spec
    return {"loaded": True, "name": S.audio_name, "t0": float(sp.times[0]), "t1": float(sp.times[-1]),
            "f0": float(sp.freqs[0]), "f1": float(sp.freqs[-1]), "params": S.params,
            "duration": float(len(S.audio[0]) / S.audio[1])}



def _model_curve(cv):
    """The fitted straight-flyover model of a curve, on its own time span (1 s steps)."""
    from ..doppler import straight_flyover_ratio

    f = cv.meta["fit"]
    t = np.arange(np.nanmin(cv.t), np.nanmax(cv.t) + 0.5, 1.0)
    return DopplerCurve(t=t, f=f["f0"] * straight_flyover_ratio(t, f["t_cpa"], f["speed"], f["d_cpa"]), name=cv.name,
                        meta=cv.meta)


def _terrain(zoom):
    from ..elevation import TerrainTiles

    if S.terrain is None:
        S.terrain = {}
    if zoom not in S.terrain:
        S.terrain[zoom] = TerrainTiles(zoom=zoom)
    return S.terrain[zoom]


def _overlay_png(r, vmin=0.2, vmax=5.0):
    """A misfit grid as a jet-coloured PNG (data URL), north up, for a map overlay."""
    import base64

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rgba = plt.get_cmap("jet")(LogNorm(vmin, vmax)(np.clip(np.nan_to_num(r, nan=vmax), vmin, vmax)))[::-1]
    b = io.BytesIO()
    plt.imsave(b, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

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
                       curves=_curves_payload(), version=S.version, overlays=S.overlays,
                       tracks=_tracks_json(), truth=S.truth, day=S.day.summary() if S.day else None)

    @app.post("/api/audio")
    def upload_audio():
        f = request.files["file"]
        suffix = Path(f.filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            path = tmp.name
        sr = int(request.form.get("sr", S.params["sr"]))
        S.progress = {"stage": "decoding audio", "fraction": 0.0}
        x, sr = load_audio(path, sr=sr)
        Path(path).unlink(missing_ok=True)
        with S.lock:
            S.audio, S.audio_name = (x, sr), f.filename
            S.params["sr"] = sr
            S.result, S.truth = None, None
            S.overlays = {}
            _load_fits()
            S.compute_spec()
        return jsonify(spec=_spec_json(), curves=_curves_payload(), fits_file=str(_fits_path()))

    @app.post("/api/audio/path")
    def load_path():
        """Load a file that is already on this computer (avoids uploading a big video)."""
        d = request.get_json()
        path = Path(d["path"]).expanduser()
        if not path.exists():
            raise ValueError(f"file not found: {path}")
        sr = int(d.get("sr", S.params["sr"]))
        with S.lock:
            S.progress = {"stage": "decoding audio", "fraction": 0.0}
            x, sr = load_audio(path, sr=sr)
            S.audio, S.audio_name = (x, sr), path.name
            S.params["sr"] = sr
            S.result, S.truth = None, None
            S.overlays = {}
            _load_fits()
            S.compute_spec()
        return jsonify(spec=_spec_json(), curves=_curves_payload(), fits_file=str(_fits_path()))

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
        """Render the visible region (t0..t1, f0..f1) at about w x h pixels.

        Time is max-pooled when there are more frames than pixels, so thin
        lines stay visible in the hour-long overview.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        which = request.args.get("view", "enhanced")
        sp = {"enhanced": S.spec, "raw": S.raw}.get(which) or _clean_spec()
        a = request.args
        t0, t1 = float(a.get("t0", sp.times[0])), float(a.get("t1", sp.times[-1]))
        f0, f1 = float(a.get("f0", sp.freqs[0])), float(a.get("f1", sp.freqs[-1]))
        w, h = int(a.get("w", 2000)), int(a.get("h", 800))
        ti = np.searchsorted(sp.times, [t0, t1])
        fi = np.searchsorted(sp.freqs, [f0, f1])
        p = sp.power_db[max(0, fi[0] - 1):fi[1] + 1, max(0, ti[0] - 1):ti[1] + 1]
        if p.size == 0:
            p = np.zeros((2, 2), np.float32)
        kt = max(1, p.shape[1] // max(w, 1))
        kf = max(1, p.shape[0] // max(h, 1))
        if kt > 1 or kf > 1:
            T, F = p.shape[1] // kt * kt, p.shape[0] // kf * kf
            p = p[:F, :T].reshape(F // kf, kf, T // kt, kt).max(axis=(1, 3))
        lo, hi = S.clim.get(which, (np.percentile(p, 5), np.percentile(p, 99.7)))
        if kt * kf > 2:
            # max-pooling lifts the noise floor; rescale so lines still stand out
            lo, hi = float(np.percentile(p, 60)), max(float(np.percentile(p, 99.9)), hi)
        img = np.clip((p - lo) / max(hi - lo, 1e-9), 0, 1)[::-1]
        buf = io.BytesIO()
        plt.imsave(buf, img, cmap="magma", format="png")
        buf.seek(0)
        # exact extent of the returned pixels (edges of the first/last bins)
        ts, fs = sp.times, sp.freqs
        dt = float(ts[1] - ts[0]) if len(ts) > 1 else 1.0
        df = float(fs[1] - fs[0]) if len(fs) > 1 else 1.0
        j0, i0 = max(0, ti[0] - 1), max(0, fi[0] - 1)
        T, F = p.shape[1] * kt, p.shape[0] * kf
        ext = [ts[j0] - dt / 2, ts[j0] + (T - 0.5) * dt, fs[i0] - df / 2, fs[i0] + (F - 0.5) * df]
        resp = send_file(buf, mimetype="image/png", max_age=0)
        resp.headers["X-Extent"] = ",".join(f"{v:.4f}" for v in ext)
        resp.headers["Access-Control-Expose-Headers"] = "X-Extent"
        return resp

    @app.get("/api/progress")
    def progress():
        return jsonify(S.progress)

    @app.post("/api/curves")
    def add_curve():
        if S.spec is None:
            raise ValueError("load a recording first")
        d = request.get_json()
        name = d.get("name") or f"pass {len(S.curves) + 1}"
        rise = None if d.get("allow_rise") else 0.5
        if d["mode"] == "guide" and d.get("isolate", True):
            def prog(fr, stage):
                S.progress = {"stage": stage, "fraction": round(fr, 3)}
            cv = isolate_curve(S.raw, d["t"], d["f"], line_window_s=float(d.get("line_window") or 30.0),
                               c=float(d.get("c") or 340.0), name=name, progress=prog)
            with S.lock:
                S.curves.append(cv)
                _save_fits()
            return jsonify(curves=_curves_payload())
        if d["mode"] == "guide":
            cv = digitise_along(S.spec, d["t"], d["f"], halfwidth_hz=float(d["width"]) if d.get("width") else None,
                                name=name)
        elif d["mode"] == "seed":
            t = float(d["t"])
            tmin = max(t - 180, d["tmin"]) if d.get("tmin") is not None else t - 180
            tmax = min(t + 180, d["tmax"]) if d.get("tmax") is not None else t + 180
            cv = clean_curve(trace_from_seed(S.spec, t, float(d["f"]), tmin, tmax, max_rise_hz_s=rise, name=name))
        else:
            cv = clean_curve(track_ridge(S.spec, float(d["f0"]), float(d["f1"]), float(d["t0"]), float(d["t1"]),
                                         max_rise_hz_s=rise, name=name))
        if np.isfinite(cv.f).sum() < 10:
            raise ValueError("no clear line found there: click a few points right on a bright line")
        cv.meta["fitted"] = False
        cv.meta["fit"] = None
        with S.lock:
            S.curves.append(cv)
            _save_fits()
        return jsonify(curves=_curves_payload())

    @app.delete("/api/curves/<int:i>")
    def delete_curve(i):
        with S.lock:
            S.curves.pop(i)
            _save_fits()
        return jsonify(curves=_curves_payload())

    @app.patch("/api/curves/<int:i>")
    def edit_curve(i):
        d = request.get_json()
        with S.lock:
            cv = S.curves[i]
            if "name" in d:
                cv.name = str(d["name"])[:40]
            for k in ("show", "use"):
                if k in d:
                    cv.meta[k] = bool(d[k])
            _save_fits()
        return jsonify(curves=_curves_payload())

    @app.post("/api/curves/<int:i>/fit")
    def fit_curve(i):
        """Fit the flyover model to the digitised points of one curve."""
        with S.lock:
            cv = S.curves[i]
            ok = np.isfinite(cv.f)
            if ok.sum() < 8:
                raise ValueError("too few points to fit")
            cv.meta["fit"] = {k: float(v) for k, v in fit_flyover(cv.t, cv.f).items()}
            cv.meta["fitted"] = True
            _save_fits()
        return jsonify(curves=_curves_payload())

    @app.post("/api/curves/<int:i>/erase")
    def erase_points(i):
        """Remove the digitised points of one curve inside a box (the fit has to be redone)."""
        d = request.get_json()
        with S.lock:
            cv = S.curves[i]
            m = (cv.t >= float(d["t0"])) & (cv.t <= float(d["t1"])) & (cv.f >= float(d["f0"])) & (cv.f <= float(d["f1"]))
            if not m.any():
                return jsonify(curves=_curves_payload(), removed=0)
            cv.f = cv.f.copy()
            cv.f[m] = np.nan
            cv.meta["fit"] = None
            cv.meta["fitted"] = False
            _save_fits()
        return jsonify(curves=_curves_payload(), removed=int(m.sum()))

    @app.get("/api/version")
    def version():
        return jsonify(version=S.version, name=S.audio_name)

    @app.get("/api/curves")
    def get_curves():
        return jsonify(curves=_curves_payload(), version=S.version, overlays=S.overlays)

    # ------------------------------------------------------------------ candidate flights on the spectrogram
    @app.post("/api/predict")
    def predict():
        """Predicted curves of the flights near one place, for one recording start time."""
        from ..predict import predict_flights

        if S.audio is None:
            raise ValueError("load a recording first")
        d = request.get_json()
        lat, lon = float(d["lat"]), float(d["lon"])
        t0 = _parse_time(d["start"])
        alt = float(d.get("alt") or 0)
        max_km = float(d.get("max_km") or 15)
        dur = len(S.audio[0]) / S.audio[1]
        if S.day is not None:
            tracks = S.day.tracks(t0 - 300, t0 + dur + 300, lat, lon, max_km + 10)
        else:
            tracks = S.tracks
        if not tracks:
            raise ValueError("no flights: open the day archive or load flight tracks first")
        S.progress = {"stage": "predicting curves", "fraction": 0.5}
        flights = predict_flights(tracks, [c for c in S.curves if np.isfinite(c.f).sum() >= 5], lat, lon, alt, t0, dur,
                                  max_km=max_km)
        S.progress = {"stage": "done", "fraction": 1.0}
        for fl in flights:  # compact for the browser
            step = max(1, len(fl["t"]) // 1500)
            fl["t"] = [round(float(v), 1) for v in fl["t"][::step]]
            fl["r"] = [round(float(v), 5) for v in fl["r"][::step]]
        key = str(d.get("key") or f"{lat:.4f},{lon:.4f},{int(t0)}")
        with S.lock:
            S.overlays[key] = {"key": key, "title": d.get("title") or f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon,
                               "start": _iso(t0), "show": True, "flights": flights}
            S.version += 1
        return jsonify(overlays=S.overlays, key=key)

    @app.patch("/api/overlays/<path:key>")
    def edit_overlay(key):
        d = request.get_json()
        with S.lock:
            ov = S.overlays[key]
            if "show" in d:
                ov["show"] = bool(d["show"])
            S.version += 1
        return jsonify(overlays=S.overlays)

    @app.delete("/api/overlays/<path:key>")
    def delete_overlay(key):
        with S.lock:
            S.overlays.pop(key, None)
            S.version += 1
        return jsonify(overlays=S.overlays)

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
                _save_fits()
        return jsonify(curves=_curves_payload())

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

    @app.post("/api/tracks/history")
    def history_tracks():
        """Load a readsb/tar1090 globe_history day (adsb.lol or ADS-B Exchange) from a local path."""
        from ..flights import load_readsb_history, save_tracks_csv

        d = request.get_json()
        src = Path(d["path"]).expanduser()
        if not src.exists():
            raise ValueError(f"not found: {src}")
        t0 = _parse_time(d["start"])
        unc = float(d.get("uncertainty", 600))
        dur = len(S.audio[0]) / S.audio[1] if S.audio else 3600
        S.progress = {"stage": "reading flight history", "fraction": 0.0}

        def prog(fr):
            S.progress = {"stage": "reading flight history", "fraction": fr or S.progress["fraction"]}

        tracks = load_readsb_history(src, t0 - unc - 300, t0 + unc + dur + 300, float(d["lat"]), float(d["lon"]),
                                     float(d.get("radius", 40)), progress=prog)
        S.progress = {"stage": "done", "fraction": 1.0}
        with S.lock:
            S.tracks = tracks
        cache = Path.home() / ".sound_geolocator" / "tracks_last.csv"
        save_tracks_csv(tracks, cache)
        return jsonify(tracks=_tracks_json(), saved=str(cache))

    # ------------------------------------------------------------------ whole-day worldwide archive
    def _day_json():
        if S.day is None:
            return None
        return S.day.summary()

    @app.post("/api/day/open")
    def day_open():
        """Open a converted day (.sgday), or start converting the tar parts (in a separate process)."""
        import subprocess
        import sys

        from ..adsb_day import DayArchive, build_progress, sgday_path

        src = Path(request.get_json()["path"].strip().replace("\\ ", " ")).expanduser()
        out = sgday_path(src)
        if (out / "aircraft.json").exists():
            S.day = DayArchive(out)
            return jsonify(day=_day_json())
        if not src.exists():
            raise ValueError(f"not found: {src}")
        if S.day_build is None or S.day_build[1].poll() is not None:
            root = Path(__file__).resolve().parents[2]
            proc = subprocess.Popen([sys.executable, "-m", "sound_geolocator.adsb_day", "build", str(src)], cwd=root)
            S.day_build = (out, proc)
        return jsonify(building=True, progress=build_progress(out))

    @app.get("/api/day/status")
    def day_status():
        from ..adsb_day import DayArchive, build_progress

        if S.day is not None and S.day_build is None:
            return jsonify(day=_day_json())
        if S.day_build is None:
            return jsonify(day=None)
        out, proc = S.day_build
        pr = build_progress(out)
        if pr["done"]:
            S.day, S.day_build = DayArchive(out), None
            return jsonify(day=_day_json())
        if proc.poll() is not None:
            S.day_build = None
            raise ValueError("the conversion stopped; open the day again to resume it")
        return jsonify(building=True, progress=pr)

    @app.post("/api/day/tracks")
    def day_tracks():
        """Flights from the day archive around the search centre and time."""
        if S.day is None:
            raise ValueError("open the day archive first")
        d = request.get_json()
        t0 = _parse_time(d["start"])
        unc = float(d.get("uncertainty", 300))
        dur = len(S.audio[0]) / S.audio[1] if S.audio else 3600
        tracks = S.day.tracks(t0 - unc - 300, t0 + unc + dur + 300, float(d["lat"]), float(d["lon"]),
                              float(d.get("radius", 40)))
        with S.lock:
            S.tracks = tracks
        return jsonify(tracks=_tracks_json())

    @app.post("/api/day/search")
    def day_search():
        """Where on Earth and when: vote with every fitted pass, then match the curves at the best places."""
        from ..adsb_day import refine_world

        if S.day is None:
            raise ValueError("open the day archive first")
        d = request.get_json()
        curves = [c for c in S.curves if c.meta.get("use", True) and c.meta.get("fit")]
        if len(curves) < 2:
            raise ValueError("fit at least two passes (and tick 'use') for a worldwide search")
        types = d.get("types")
        passes = [dict(c.meta["fit"], types=types) for c in curves]
        S.progress = {"stage": "worldwide vote", "fraction": 0.0}

        def prog_v(fr):
            S.progress = {"stage": "worldwide vote", "fraction": round(fr * 0.3, 3)}

        cands = S.day.search_world(passes, speed_tol=float(d.get("speed_tol", 0.4)), n_best=int(d.get("n_votes", 40)),
                                   c=float(d.get("c", 340)), progress=prog_v)
        if not cands:
            S.progress = {"stage": "done", "fraction": 1.0}
            return jsonify(candidates=[])

        def prog_r(fr):
            S.progress = {"stage": "matching curves at the best places", "fraction": round(0.3 + 0.7 * fr, 3)}

        rec_s = len(S.audio[0]) / S.audio[1] if S.audio else None
        res = refine_world(S.day, curves, cands, n=int(d.get("n_refine", 8)), c=float(d.get("c", 340)),
                           recording_s=rec_s, progress=prog_r)
        S.progress = {"stage": "done", "fraction": 1.0}
        out = []
        for cd, r in res:
            item = {"vote": cd.to_json(), "vote_iso": _iso(cd.t_start)}
            if r is not None:
                item.update(lat=r.lat, lon=r.lon, t_start=r.t_start, t_start_iso=_iso(r.t_start), cost=r.cost,
                            assignments=[a.__dict__ for a in r.assignments])
            out.append(item)
        return jsonify(candidates=out)

    @app.post("/api/day/area")
    def day_area():
        """Every place and start time that could hear all the ticked passes, then the silence filter.

        Speed and distance windows come from each fit: 3 sigma + 3 % + the wind
        allowance for the speed, max(3 sigma, 20 %) for the distance. The quiet
        gaps are the stretches between consecutive ticked passes; a place is
        dropped when some other aircraft came within the per-type slant
        distance there during a gap.
        """
        if S.day is None:
            raise ValueError("open the day archive first")
        d = request.get_json()
        curves = sorted([c for c in S.curves if c.meta.get("use", True) and c.meta.get("fit")], key=lambda c: np.nanmin(c.t))
        if len(curves) < 2:
            raise ValueError("fit at least two passes and tick 'use' on the ones to search with")
        wind = float(d.get("wind", 15))
        passes = []
        for c in curves:
            f = c.meta["fit"]
            se = f.get("speed_err") or 0.0
            de = f.get("d_cpa_err") or 0.0
            tv = 3 * se + 0.03 * f["speed"] + wind
            td = max(3 * de, 0.2 * f["d_cpa"])
            passes.append(dict(t_cpa=f["t_cpa"], speed=f["speed"], d_cpa=f["d_cpa"], speed_lo=f["speed"] - tv,
                               speed_hi=f["speed"] + tv, d_lo=max(f["d_cpa"] - td, 30.0), d_hi=f["d_cpa"] + td,
                               name=c.name))
        bbox = CONUS_BBOX if d.get("area", "conus") == "conus" else None
        S.progress = {"stage": "places that hear every pass", "fraction": 0.0}

        def prog_v(fr):
            S.progress = {"stage": "places that hear every pass", "fraction": round(fr * 0.5, 3)}

        rec_s = len(S.audio[0]) / S.audio[1] if S.audio else 3600.0
        cands = S.day.search_world(passes, n_best=None, bbox=bbox, cell_km=2.0, recording_s=rec_s,
                                   c=float(d.get("c", 340)), progress=prog_v)
        n_all = len(cands)
        gaps = []
        if d.get("silence", True) and cands:
            radius = {k: float(d.get("r_" + k, v)) * 1000 for k, v in
                      {"prop": 3.5, "heli": 3.5, "jet": 6.0, "unknown": 4.5}.items()}
            for a, b in zip(curves[:-1], curves[1:]):
                w0, w1 = float(np.nanmax(a.t)) + 10, float(np.nanmin(b.t)) - 20
                if w1 - w0 < 60:
                    continue
                gaps.append([w0, w1])
                S.progress = {"stage": f"silence filter {a.name}-{b.name}", "fraction": 0.5}

                def prog_s(fr):
                    S.progress = {"stage": "silence filter", "fraction": round(0.5 + 0.5 * fr, 3)}

                ok = S.day.silence_ok(cands, w0, w1, radius, cell_km=2.0, progress=prog_s)
                cands = [c for c, o in zip(cands, ok) if o]
        S.area = (cands, curves)
        # one point per 2 km cell, with the range of start times that work there
        cells = {}
        for c in cands:
            k = (round(c.lat, 3), round(c.lon, 3))
            e = cells.setdefault(k, [c.t_start, c.t_start, 0, set()])
            e[0], e[1], e[2] = min(e[0], c.t_start), max(e[1], c.t_start), e[2] + 1
            e[3].update(c.aircraft)
        pts = [{"lat": k[0], "lon": k[1], "t0": v[0], "t1": v[1], "t0_iso": _iso(v[0]), "t1_iso": _iso(v[1]), "n": v[2],
                "aircraft": sorted(v[3])[:6]} for k, v in cells.items()]
        pts.sort(key=lambda p: p["t0"])
        S.progress = {"stage": "done", "fraction": 1.0}
        return jsonify(n_all=n_all, n_quiet=len(cands), gaps=gaps, passes=passes, points=pts[:40000],
                       truncated=len(pts) > 40000)

    @app.post("/api/day/rank")
    def day_rank():
        """Simulate each candidate's aircraft and fit the passes: misfit per pass, for the B-vs-C plot."""
        from ..rank import rank_candidates

        if not S.area or not S.area[0]:
            raise ValueError("run 'Find places that hear the passes' first")
        cands, curves = S.area
        if request.get_json().get("against", "model") == "model":
            # compare with each pass's fitted flyover model (smooth, no stray points) over the digitised span
            from ..doppler import straight_flyover_ratio

            model = []
            for cv in curves:
                f = cv.meta["fit"]
                t = np.arange(np.nanmin(cv.t), np.nanmax(cv.t) + 0.5, 1.0)
                model.append(DopplerCurve(t=t, f=f["f0"] * straight_flyover_ratio(t, f["t_cpa"], f["speed"], f["d_cpa"]),
                                          name=cv.name))
            curves = model
        S.progress = {"stage": "simulating Doppler curves", "fraction": 0.0}

        def prog(fr):
            S.progress = {"stage": "ground heights" if fr < 0.1 else "simulating Doppler curves",
                          "fraction": round(fr, 3)}

        rk = rank_candidates(S.day, cands, curves, c=float(request.get_json().get("c", 340)), progress=prog)
        S.progress = {"stage": "done", "fraction": 1.0}
        rms = rk["rms"]
        ok = np.all(np.isfinite(rms), 1)
        order = np.argsort(np.where(ok, np.sqrt(np.mean(rms ** 2, 1)), np.inf))
        pts = []
        for i in order[:int(request.get_json().get("max_points", 60000))]:
            if not ok[i]:
                break
            cd = cands[i]
            pts.append({"lat": round(float(rk["lat"][i]), 5), "lon": round(float(rk["lon"][i]), 5),
                        "t_start": float(rk["t_start"][i]), "t_start_iso": _iso(float(rk["t_start"][i])),
                        "rms": [round(float(v), 3) for v in rms[i]], "f0": [round(float(v), 2) for v in rk["f0"][i]],
                        "h": round(float(rk["h"][i])), "ground": round(float(rk["ground"][i])),
                        "aircraft": cd.aircraft})
        return jsonify(passes=[c.name for c in curves], n=len(cands), points=pts)

    # ------------------------------------------------------------ guided search, one step at a time
    @app.get("/api/regions")
    def regions():
        from ..regions import REGIONS

        return jsonify(regions=[{"name": k, "bbox": v} for k, v in REGIONS.items()])

    def _flow_passes(d):
        fitted = [i for i, c in enumerate(S.curves) if c.meta.get("fit")]
        a = int(d["anchor"])
        others = [int(i) for i in d.get("others", []) if int(i) != a]
        if a not in fitted or not others or any(i not in fitted for i in others):
            raise ValueError("pick a fitted anchor pass and at least one other fitted pass")
        return a, others

    @app.post("/api/flow/search")
    def flow_search():
        """Step: anchor on one pass, lock its time, keep places where every other pass agrees."""
        from ..anchor import anchor_search
        from ..regions import region_bbox

        if S.day is None:
            raise ValueError("open the day archive first (step 3)")
        d = request.get_json()
        a, others = _flow_passes(d)
        bbox = region_bbox(d.get("region"))
        S.progress = {"stage": "terrain heights and anchored search", "fraction": 0.0}

        def prog(fr):
            S.progress = {"stage": "anchored search", "fraction": round(fr, 3)}

        cands, dt, dp, info = anchor_search(S.day, S.curves[a].meta["fit"], [S.curves[i].meta["fit"] for i in others],
                                            bbox=bbox, sys_frac=float(d.get("sys", 3)) / 100, wind=float(d.get("wind", 15)),
                                            ground_fn=_terrain(8), progress=prog)
        S.flow = {"region": d.get("region"), "anchor": a, "others": others, "cands": cands, "dt": dt, "dp": dp,
                  "silence": None, "rank": None}
        S.progress = {"stage": "done", "fraction": 1.0}
        names = [S.curves[a].name] + [S.curves[i].name for i in others]
        pts = [{"lat": round(c.lat, 5), "lon": round(c.lon, 5), "t0": c.t_start, "iso": _iso(c.t_start), "aircraft": c.aircraft}
               for c in cands[:60000]]
        return jsonify(n=len(cands), points=pts, passes=names,
                       tol={"anchor": info["anchor"], "others": info["others"], "pos_tol": info["pos_tol"],
                            "t_tol": info["t_tol"]})

    @app.post("/api/flow/silence")
    def flow_silence():
        """Step: drop places where some other aircraft was close enough to be heard between the passes."""
        if not S.flow.get("cands"):
            raise ValueError("run the anchored search first")
        d = request.get_json()
        radius = {k: float(d.get("r_" + k, v)) * 1000 for k, v in
                  {"prop": 2.5, "heli": 2.5, "jet": 4.5, "unknown": 3.5}.items()}
        idx = [S.flow["anchor"]] + S.flow["others"]
        order = sorted(idx, key=lambda i: np.nanmin(S.curves[i].t))
        cands = S.flow["cands"]
        ok = np.ones(len(cands), bool)
        gaps = []
        if d.get("on", True):
            for i, j in zip(order[:-1], order[1:]):
                w0, w1 = float(np.nanmax(S.curves[i].t)) + 10, float(np.nanmin(S.curves[j].t)) - 20
                if w1 - w0 < 60:
                    continue
                gaps.append([w0, w1])
                S.progress = {"stage": f"silence filter {S.curves[i].name}-{S.curves[j].name}", "fraction": 0.5}
                ok &= S.day.silence_ok(cands, w0, w1, radius, cell_km=2.0)
        S.flow["silence"] = ok
        S.flow["rank"] = None
        S.progress = {"stage": "done", "fraction": 1.0}
        keep = np.nonzero(ok)[0]
        pts = [{"lat": round(cands[k].lat, 5), "lon": round(cands[k].lon, 5), "iso": _iso(cands[k].t_start),
                "aircraft": cands[k].aircraft} for k in keep[:60000]]
        return jsonify(n=len(cands), n_quiet=int(ok.sum()), gaps=gaps, points=pts)

    @app.post("/api/flow/rank")
    def flow_rank():
        """Step: fit the anchor pass with the real aircraft and lock it; score the other passes with that start time."""
        from ..rank import rank_locked

        if not S.flow.get("cands"):
            raise ValueError("run the anchored search first")
        d = request.get_json()
        ok = S.flow["silence"] if S.flow.get("silence") is not None else np.ones(len(S.flow["cands"]), bool)
        keep = np.nonzero(ok)[0][:int(d.get("max", 20000))]
        cands = [S.flow["cands"][k] for k in keep]
        if not cands:
            raise ValueError("no place left to rank")
        model = d.get("against", "model") == "model"
        cv = lambda i: _model_curve(S.curves[i]) if model else S.curves[i]
        ground = _terrain(9)(np.array([c.lat for c in cands]), np.array([c.lon for c in cands]))

        def prog(fr):
            S.progress = {"stage": "fitting the anchor pass, scoring the others", "fraction": round(fr, 3)}

        rk = rank_locked(S.day, cands, cv(S.flow["anchor"]), [cv(i) for i in S.flow["others"]], ground,
                         c=float(d.get("c", 340)), joint=bool(d.get("joint", True)),
                         pos_box_m=1200.0 if d.get("joint", True) else 600.0, shift_box_s=5.0, progress=prog)
        S.progress = {"stage": "done", "fraction": 1.0}
        score = np.sqrt((rk["rms_anchor"] ** 2 + np.sum(rk["rms_locked"] ** 2, 1)) / (1 + rk["rms_locked"].shape[1]))
        order = np.argsort(np.where(np.isfinite(score), score, np.inf))
        pts = []
        for i in order:
            if not np.isfinite(score[i]):
                break
            pts.append({"k": int(i), "lat": round(float(rk["lat"][i]), 6), "lon": round(float(rk["lon"][i]), 6),
                        "t_start": float(rk["t_start"][i]), "iso": _iso(float(rk["t_start"][i])),
                        "rms_anchor": round(float(rk["rms_anchor"][i]), 3),
                        "rms_others": [round(float(v), 3) for v in rk["rms_locked"][i]],
                        "slide": [round(float(v), 2) for v in rk["dt_other"][i]], "score": round(float(score[i]), 3),
                        "h": round(float(rk["h"][i])), "aircraft": cands[i].aircraft})
        S.flow["rank"] = pts
        names = [S.curves[S.flow["anchor"]].name] + [S.curves[i].name for i in S.flow["others"]]
        return jsonify(n=len(cands), passes=names, points=pts)

    @app.post("/api/flow/grid")
    def flow_grid():
        """Step: fine misfit maps around one ranked place, per pass and combined."""
        from ..rank import misfit_grid

        if not S.flow.get("rank"):
            raise ValueError("rank the places first")
        d = request.get_json()
        p = S.flow["rank"][int(d.get("rank", 0))]
        idx = [S.flow["anchor"]] + S.flow["others"]
        use = [k for k, i in enumerate(idx) if i in set(int(x) for x in d.get("passes", idx))] or list(range(len(idx)))
        model = d.get("against", "model") == "model"
        curves = [_model_curve(S.curves[idx[k]]) if model else S.curves[idx[k]] for k in use]
        half, step = float(d.get("half", 500)), float(d.get("step", 5))
        if (2 * half / step + 1) ** 2 > 400_000:
            raise ValueError("too many grid points: make the box smaller or the step larger")

        def prog(fr):
            S.progress = {"stage": "fine grid", "fraction": round(fr, 3)}

        g = misfit_grid(S.day, [p["aircraft"][k] for k in use], curves, p["lat"], p["lon"], p["t_start"], half_m=half,
                        step_m=step, shift_s=float(d.get("shift", 4)), ground_fn=_terrain(12), c=float(d.get("c", 340)),
                        progress=prog)
        S.progress = {"stage": "done", "fraction": 1.0}
        la, lo = g["lat"], g["lon"]
        dla, dlo = (la[1, 0] - la[0, 0]) / 2, (lo[0, 1] - lo[0, 0]) / 2
        bounds = [[float(la.min() - dla), float(lo.min() - dlo)], [float(la.max() + dla), float(lo.max() + dlo)]]
        layers = []
        maps = [(curves[k].name + " alone", g["rms"][k]) for k in range(len(curves))]
        if len(curves) > 1:
            maps += [("combined (one start time)", g["combined"]), ("combined, each pass weighted by its fit", g["combined_w"])]
        for name, r in maps:
            j = np.unravel_index(np.nanargmin(r), r.shape)
            layers.append({"name": name, "img": _overlay_png(r), "best": [float(la[j]), float(lo[j])],
                           "rms": round(float(r[j]), 3)})
        jc = np.unravel_index(np.nanargmin(g["combined"]), g["combined"].shape)
        return jsonify(bounds=bounds, layers=layers, center=[p["lat"], p["lon"]],
                       best_start_iso=_iso(p["t_start"] + float(g["shift"][jc])))

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
        curves = [c for c in S.curves if c.meta.get("use", True)]
        if not curves:
            raise ValueError("tick 'use' on at least one curve")
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
            S.overlays = {}
            S.tracks = flights
            S.truth = {"lat": truth[0], "lon": truth[1], "t_start": t0}
            S.compute_spec()
        guess = {"lat": round(truth[0] + 0.08, 4), "lon": round(truth[1] - 0.1, 4), "start": _iso(t0 + 75),
                 "uncertainty": 120, "radius": 20, "alt": truth[2]}
        return jsonify(spec=_spec_json(), tracks=_tracks_json(), guess=guess, curves=_curves_payload())

    return app


def main(host="127.0.0.1", port=8765, open_browser=True, preload=None):
    app = create_app()
    if preload:
        print(f"Loading {preload} …")
        x, sr = load_audio(preload, sr=S.params["sr"])
        S.audio, S.audio_name = (x, sr), Path(preload).name
        _load_fits()
        S.compute_spec()
        print(f"Loaded {len(x) / sr / 60:.1f} min of audio")
    url = f"http://{host}:{port}"
    print(f"Sound Geolocator GUI on {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)
