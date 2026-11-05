"""Matplotlib figures: spectrogram with picked curves, and the location cost map."""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_spectrogram(spec, path, curves=(), fmax=None, title="", vmax=None, grid=True):
    fig, ax = plt.subplots(figsize=(14, 6))
    p = spec.power_db
    vmax = vmax or np.percentile(p, 99.7)
    ax.pcolormesh(spec.times, spec.freqs, p, shading="auto", cmap="magma", vmin=np.percentile(p, 5), vmax=vmax)
    for cv in curves:
        ax.plot(cv.t, cv.f, "-", lw=1.4, color="cyan", alpha=0.85)
        ok = np.isfinite(cv.f)
        if ok.any():
            ax.annotate(cv.name, (cv.t[ok][0], cv.f[ok][0]), color="cyan", fontsize=9,
                        xytext=(0, 6), textcoords="offset points")
    if fmax:
        ax.set_ylim(spec.freqs[0], fmax)
    if grid:
        ax.minorticks_on()
        ax.grid(which="major", color="w", alpha=0.35, lw=0.6)
        ax.grid(which="minor", color="w", alpha=0.12, lw=0.4)
    ax.set_xlabel("time from recording start (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title or "enhanced spectrogram")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(spec, curves, path, fmax=None):
    plot_spectrogram(spec, path, curves, fmax=fmax, title="enhanced spectrogram with picked Doppler curves")


def plot_cost_map(res, path, truth=None, tracks=()):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    maps = [(res.grid_lat, res.grid_lon, res.cost_map)] + (res.refine_maps[:1] if res.refine_maps else [])
    for ax, (glat, glon, cm) in zip(axes, maps):
        im = ax.pcolormesh(glon, glat, np.log10(cm + 1e-12), shading="auto", cmap="viridis_r")
        fig.colorbar(im, ax=ax, label="log10 cost")
        lo, la = (glon.min(), glon.max()), (glat.min(), glat.max())
        for tr in tracks:
            ax.plot(tr.lon, tr.lat, "-", color="w", lw=0.8, alpha=0.6)
        ax.set_xlim(*lo); ax.set_ylim(*la)
        ax.plot(res.lon, res.lat, "r*", ms=14, label="estimate")
        if truth is not None:
            ax.plot(truth[1], truth[0], "wo", mfc="none", ms=12, mew=2, label="truth")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_aspect(1 / np.cos(np.radians(res.lat)))
        ax.legend(loc="lower right")
    axes[0].set_title("coarse search")
    if len(maps) > 1:
        axes[1].set_title("refined")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
