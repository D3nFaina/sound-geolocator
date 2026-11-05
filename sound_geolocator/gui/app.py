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
from ..extract import (DopplerCurve, clean_curve, digitise_along, load_curves, save_curves, trace_from_seed,
                       track_ridge)
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
        self.clim = {}
        self.progress = {"stage": "idle", "fraction": 0.0}
        self.dem_file = None
        self.version = 0  # bumped on every change of the curves, so other windows can follow
        self.day = None  # DayArchive
        self.day_build = None  # (path, subprocess) while converting

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
        # colour limits from the whole recording, so the view doesn't flicker when panning
        self.clim = {}
        for k, sp in (("enhanced", self.spec), ("raw", self.raw)):
            sub = sp.power_db[:, ::max(1, sp.power_db.shape[1] // 2000)]
            self.clim[k] = (float(np.percentile(sub, 5)), float(np.percentile(sub, 99.7)))
        self.progress = {"stage": "done", "fraction": 1.0}


S = State()


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
    info["fit"] = None if fit is None else {k: round(v, 2) for k, v in fit.items()}
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
                       curves=_curves_payload(), version=S.version,
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
        sp = S.spec if which == "enhanced" else S.raw
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
        return jsonify(curves=_curves_payload(), version=S.version)

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
