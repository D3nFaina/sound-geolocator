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
