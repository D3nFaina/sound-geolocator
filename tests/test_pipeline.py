import numpy as np
import pytest

from sound_geolocator.doppler import doppler_ratio, fit_flyover, straight_flyover_ratio
from sound_geolocator.extract import clean_curve, load_curves, save_curves, trace_from_seed, track_ridge
from sound_geolocator.flights import load_tracks_csv, save_tracks_csv
from sound_geolocator.geo import enu_offsets_to_lla, haversine_m, lla_to_ecef, lla_to_enu_offsets
from sound_geolocator.locate import locate
from sound_geolocator.spectrogram import compute_spectrogram, enhance
from sound_geolocator.synth import demo_scenario, render_recording, straight_track

LAT0, LON0, ALT0 = 45.4642, 9.19, 120.0
T0 = 1_760_000_000.0


@pytest.fixture(scope="module")
def scenario():
    fl, truth, t0 = demo_scenario()
    assert truth == (LAT0, LON0, ALT0) and t0 == T0
    x, sr = render_recording(fl[:2], truth, T0, 360.0, sr=4000)
    spec = enhance(compute_spectrogram(x, sr, window_s=1.0, hop_s=0.5, fmax=400))
    return fl, spec


def test_geo_roundtrip():
    lat, lon = enu_offsets_to_lla(LAT0, LON0, 3000.0, -2000.0)
    e, n = lla_to_enu_offsets(LAT0, LON0, lat, lon)
    assert abs(e - 3000) < 0.5 and abs(n + 2000) < 0.5
    assert abs(haversine_m(LAT0, LON0, lat, lon) - np.hypot(3000, 2000)) < 15


def test_ratio_matches_closed_form():
    tr = straight_track("x", LAT0, LON0, T0, 0, -20000, 0, 100, 1000, duration=400, report_dt=1)
    O = lla_to_ecef(LAT0, LON0 + 0.0, 0.0)
    t = T0 + np.arange(150, 250, 2.0)
    r = doppler_ratio(tr, O, t)
    r2 = straight_flyover_ratio(t, T0 + 200, 100, 1000)
    assert np.max(np.abs(r - r2)) < 2e-3
    assert r[0] > 1 > r[-1]


def test_fit_flyover_recovers_parameters():
    t = np.arange(0, 120, 0.5)
    f = 100 * straight_flyover_ratio(t, 60, 120, 1500) + np.random.default_rng(0).normal(0, 0.1, t.size)
    r = fit_flyover(t, f)
    assert abs(r["f0"] - 100) < 1 and abs(r["speed"] - 120) < 10 and abs(r["d_cpa"] - 1500) < 300


def test_extraction_follows_lines(scenario):
    fl, spec = scenario
    c = clean_curve(track_ridge(spec, 60, 145, 40, 170))
    O = lla_to_ecef(LAT0, LON0, ALT0)
    truth = 90 * doppler_ratio(fl[0], O, T0 + c.t)
    ok = np.isfinite(c.f)
    assert ok.mean() > 0.9
    assert np.median(np.abs(c.f[ok] - truth[ok])) < 0.6

    s = trace_from_seed(spec, 200, 125)
    truth = 103 * doppler_ratio(fl[1], O, T0 + s.t)
    ok = np.isfinite(s.f)
    assert np.median(np.abs(s.f[ok] - truth[ok])) < 0.6


def test_io_roundtrip(tmp_path, scenario):
    fl, spec = scenario
    save_tracks_csv(fl, tmp_path / "t.csv")
    back = load_tracks_csv(tmp_path / "t.csv")
    assert {t.ident for t in back} == {"a1b2c3", "d4e5f6", "0f0f0f"}
    c = track_ridge(spec, 60, 145, 40, 170, name="x")
    save_curves([c], tmp_path / "c.json")
    c2 = load_curves(tmp_path / "c.json")[0]
    assert c2.name == "x" and np.allclose(np.nan_to_num(c.f), np.nan_to_num(c2.f))


def test_locate_two_passes(scenario):
    fl, spec = scenario
    c1 = clean_curve(track_ridge(spec, 60, 145, 40, 170, name="A"))
    c2 = clean_curve(trace_from_seed(spec, 200, 125, name="B"))
    res = locate([c1, c2], fl, (LAT0 + 0.05, LON0 - 0.06), T0 + 40, t_uncertainty=60,
                 radius_km=12, grid_step_m=1000, refine_levels=2, observer_alt_m=ALT0, verbose=False)
    assert haversine_m(res.lat, res.lon, LAT0, LON0) < 400
    assert abs(res.t_start - T0) < 3
    assert [a.flight for a in res.assignments] == ["A1B2C3", "D4E5F6"]


def test_aircraft_type_filter(tmp_path):
    from sound_geolocator.aircraft_types import classify, filter_tracks

    assert [classify(x) for x in ("AT76", "A320", "EC35", "C172", "ZZZZ")] == ["prop", "jet", "heli", "prop", "unknown"]
    fl, _, _ = demo_scenario()
    save_tracks_csv(fl, tmp_path / "t.csv")
    back = load_tracks_csv(tmp_path / "t.csv")
    assert {t.typecode for t in back} == {"AT76", "DH8D", "A320"}
    assert {t.ident for t in filter_tracks(back, ("prop",))} == {"a1b2c3", "d4e5f6"}


def test_silence_and_dem(scenario):
    """An unheard flight passing close during the recording penalises the area under it,
    and a DEM callable works as the observer altitude."""
    from sound_geolocator.elevation import DEM

    fl, spec = scenario
    c1 = clean_curve(track_ridge(spec, 60, 145, 40, 170, name="A"))
    c2 = clean_curve(trace_from_seed(spec, 200, 125, name="B"))
    # a silent helicopter crossing 4 km south of the listener during the recording
    heli = straight_track("ccc111", LAT0, LON0, T0 - 300, -20000, -4000, 90, 60, 300, duration=1200,
                          typecode="EC35")
    dem = DEM([LAT0 - 1, LAT0 + 1], [LON0 - 1, LON0 + 1], np.full((2, 2), ALT0), height_above_ground_m=0)
    kw = dict(t_uncertainty=30, radius_km=8, grid_step_m=1000, refine_levels=1, observer_alt_m=dem,
              verbose=False, recording_s=360, n_candidates=1)
    quiet = locate([c1, c2], fl[:2] + [heli], (LAT0, LON0), T0 + 10, silence_km=2.0, **kw)
    loud = locate([c1, c2], fl[:2] + [heli], (LAT0, LON0), T0 + 10, **kw)
    diff = quiet.cost_map - loud.cost_map
    assert diff.max() == pytest.approx(0.01, rel=0.01)  # penalised somewhere (under the helicopter)
    e, n = lla_to_enu_offsets(LAT0, LON0, quiet.grid_lat, quiet.grid_lon)
    assert np.all(diff[np.abs(n + 4000) > 3500] == 0)  # ...and only near its path
    assert haversine_m(quiet.lat, quiet.lon, LAT0, LON0) < 400
    assert quiet.observer_alt == pytest.approx(ALT0)


def _write_day_tar(tmp_path, flights, n_split=2):
    """A fake adsb.lol day: one gzip readsb trace per aircraft, tar split into parts."""
    import gzip
    import io
    import json
    import tarfile

    from sound_geolocator.flights import FT
    from sound_geolocator.geo import lla_to_enu_offsets

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for tr in flights:
            t, la, lo, al = tr.t[::2], tr.lat[::2], tr.lon[::2], tr.alt[::2]
            e, n = lla_to_enu_offsets(la[0], lo[0], la, lo)
            # reported velocity comes from GNSS: smooth, unlike differences of noisy positions
            k9 = np.ones(15) / 15
            ve = np.gradient(np.convolve(e, k9, mode='same'), t)
            vn = np.gradient(np.convolve(n, k9, mode='same'), t)
            ve[:8], ve[-8:], vn[:8], vn[-8:] = ve[8], ve[-9], vn[8], vn[-9]
            gs, trk = np.hypot(ve, vn) / 0.514444, np.degrees(np.arctan2(ve, vn)) % 360
            rows = [[float(t[i] - t[0]), float(la[i]), float(lo[i]), int(al[i] / FT), float(gs[i]), float(trk[i]),
                     0, 0, None, "adsb_icao", int(al[i] / FT)] for i in range(len(t))]
            raw = gzip.compress(json.dumps({"icao": tr.ident, "timestamp": float(t[0]), "t": tr.typecode,
                                            "trace": rows}).encode())
            ti = tarfile.TarInfo(f"./traces/{tr.ident[-2:]}/trace_full_{tr.ident}.json")
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
    b = buf.getvalue()
    cut = np.linspace(0, len(b), n_split + 1).astype(int)
    for k in range(n_split):
        (tmp_path / f"day.tar.a{'abcdefgh'[k]}").write_bytes(b[cut[k]:cut[k + 1]])
    return tmp_path / "day.tar.aa"


def test_day_archive_and_world_search(tmp_path):
    from sound_geolocator.adsb_day import build_day, open_day
    from sound_geolocator.geo import haversine_m
    from sound_geolocator.synth import demo_scenario, straight_track

    flights, truth, t0 = demo_scenario()
    rng = np.random.default_rng(3)
    for k in range(150):  # distractors all over the world
        flights.append(straight_track(f"{k:06x}", rng.uniform(-50, 60), rng.uniform(-180, 180), t0 + rng.uniform(-30000, 30000),
                                      0, 0, rng.uniform(0, 360), rng.uniform(60, 250), rng.uniform(300, 6000),
                                      duration=2000, noise_m=5, typecode="C172"))
    src = _write_day_tar(tmp_path, flights)
    # resumable: stop after every chunk, then carry on
    out = tmp_path / "day.sgday"
    for _ in range(50):
        if build_day(src, out, time_budget_s=0.0, chunk=40, workers=1):
            break
    day = open_day(out)
    assert len(day) > 100  # low-enough aircraft of the fake day
    near = day.tracks(t0, t0 + 360, truth[0], truth[1], 30)
    assert {t.ident for t in near} == {"a1b2c3", "d4e5f6"}
    c = 340.0
    d1, d2 = np.hypot(1500, 1380), np.hypot(2000, 2380)
    passes = [dict(t_cpa=100.0, speed=120, d_cpa=d1), dict(t_cpa=200.0, speed=140, d_cpa=d2)]
    cands = day.search_world(passes, recording_s=360)
    assert cands, "the true place should be voted for"
    best = min(cands[:3], key=lambda w: haversine_m(w.lat, w.lon, truth[0], truth[1]))
    assert haversine_m(best.lat, best.lon, truth[0], truth[1]) < 6000
    assert abs(best.t_start - t0) < 60
    # silence filter: loud while pass 1 is overhead, quiet once both have gone
    R = {"prop": 5000, "jet": 7000, "heli": 5000, "unknown": 6000}
    assert not day.silence_ok([best], 90, 110, R)[0]
    assert day.silence_ok([best], 300, 360, {"prop": 1500, "jet": 1500, "heli": 1500, "unknown": 1500})[0]
    # simulate the voting aircraft's Doppler curves at each candidate and fit them
    from sound_geolocator.doppler import doppler_ratio
    from sound_geolocator.extract import DopplerCurve
    from sound_geolocator.geo import lla_to_ecef
    from sound_geolocator.rank import rank_candidates

    O = lla_to_ecef(*truth)
    curves = []
    for fl, f0, (t_a, t_b) in zip(flights[:2], (90.0, 120.0), ((60, 140), (160, 240))):
        tr = np.arange(t_a, t_b, 0.5)
        curves.append(DopplerCurve(t=tr, f=f0 * doppler_ratio(fl, O, t0 + tr, c), name=fl.ident))
    top = [w for w in day.search_world(passes, recording_s=360, n_best=None, cell_km=2.0)
           if w.aircraft == ["a1b2c3", "d4e5f6"]]
    assert top
    rk = rank_candidates(day, top, curves, ground_m=np.full(len(top), truth[2]), c=c)
    i = int(np.argmin(np.sum(rk["rms"] ** 2, 1)))
    assert rk["rms"][i].max() < 0.3
    assert haversine_m(rk["lat"][i], rk["lon"][i], truth[0], truth[1]) < 100
    assert abs(rk["t_start"][i] - t0) < 1
    assert abs(rk["f0"][i, 0] - 90) < 0.5
    # anchor on pass 1, lock its time, test pass 2: only the true pair at the true place survives
    from sound_geolocator.anchor import anchor_search
    from sound_geolocator.rank import rank_locked

    fa = dict(t_cpa=100.0, speed=120.0, d_cpa=d1, speed_err=0.3, d_cpa_err=20, t_cpa_err=0.1)
    fb = dict(t_cpa=200.0, speed=140.0, d_cpa=d2, speed_err=0.3, d_cpa_err=20, t_cpa_err=0.2)
    ac, _, _, _ = anchor_search(day, fa, fb, ground_fn=lambda la, lo: np.full(len(la), truth[2]))
    assert ac and all(w.aircraft == ["a1b2c3", "d4e5f6"] for w in ac)
    lk = rank_locked(day, ac, curves[0], curves[1], np.full(len(ac), truth[2]), c=c)
    k = int(np.argmin(lk["rms_locked"]))
    assert lk["rms_locked"][k] < 0.3 and abs(lk["dt_other"][k]) < 0.3
    assert haversine_m(lk["lat"][k], lk["lon"][k], truth[0], truth[1]) < 100


def test_predicted_flight_curves_match_passes():
    from sound_geolocator.predict import predict_flights
    from sound_geolocator.synth import demo_scenario, render_recording

    flights, truth, t0 = demo_scenario()
    x, sr = render_recording(flights[:2], truth, t0, 360.0, sr=4000)
    sp = enhance(compute_spectrogram(x, sr, fmax=400))
    cv = clean_curve(track_ridge(sp, 60, 160, 40, 170, name="A"))
    out = predict_flights(flights, [cv], truth[0], truth[1], truth[2], t0, 360.0, max_km=15)
    by = {o["ident"]: o for o in out}
    assert "0f0f0f" not in by  # the far-away distractor is not drawn
    assert by["a1b2c3"]["match"] == "A" and by["a1b2c3"]["rms_hz"] < 1.5


def test_isolate_curve_ignores_steady_lines():
    """A faint sweep crossed by strong steady tones: two clicks are enough to isolate and fit it."""
    from sound_geolocator.extract import isolate_curve

    sr, dur = 2000, 300.0
    rng = np.random.default_rng(3)
    t = np.arange(int(sr * dur)) / sr
    f_inst = 150 * straight_flyover_ratio(t, 150.0, 130.0, 2500.0)
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(f_inst) / sr)
    for f_hum in (120.0, 147.0, 172.0):  # steady lines the sweep crosses, stronger than it
        x += 1.0 * np.sin(2 * np.pi * f_hum * t)
    x += 0.04 * rng.normal(0, 1, t.size) * np.sqrt(sr)
    raw = compute_spectrogram(x, sr, window_s=1.0, hop_s=0.25, fmax=400)
    cv = isolate_curve(raw, [80, 225], [f_inst[80 * sr], f_inst[225 * sr]])
    p = cv.meta["fit"]
    assert cv.meta["points"] > 300
    assert abs(p["t_cpa"] - 150) < 1 and abs(p["speed"] - 130) < 5 and abs(p["d_cpa"] - 2500) / 2500 < 0.1
