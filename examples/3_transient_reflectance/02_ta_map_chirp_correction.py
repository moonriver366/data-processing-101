"""
3.02 — TA maps: wavelength calibration and chirp correction
===========================================================

Raw format note: our camera software writes paired text files per scan
— a "para" file (first row = pump-probe delays in ps, plus crop/pixel
metadata) and a "sum" file (all camera frames concatenated row-wise,
one frame per delay). Once reshaped to (delay, y, wavelength) the two
generic preprocessing steps below apply to any TA/transient-
reflectance setup.

STEP 1 — WAVELENGTH CALIBRATION with edge filters. The probe is white
light; the camera axis is pixels. Drop a longpass/shortpass filter in
the probe path and the white light truncates exactly at the filter
edge — the 50% crossing pixel maps to the edge wavelength printed on
the filter. Two filters -> two (pixel, lambda) points -> the usual
two-point linear calibration. (More points if the dispersion is not
linear, but for our gratings linear has always been enough.)

STEP 2 — CHIRP CORRECTION. Group-velocity dispersion in the optics
makes time zero wavelength-dependent: t0(lambda), the "chirp", visible
as a bent onset in the raw map. Model t0(lambda) with a low-order
polynomial (fit it to the coherent artifact / signal onset), then
re-interpolate every wavelength column onto a common time axis.

All data synthetic.

Run:  python 02_ta_map_chirp_correction.py
"""

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=42)


# ----------------------------------------------------------------------
# STEP 1: filter-edge wavelength calibration
# ----------------------------------------------------------------------
def edge_crossing_pixel(profile):
    """Pixel where the truncated white light crosses 50% of its plateau."""
    plateau = np.percentile(profile, 90)
    half = 0.5 * plateau
    above = profile > half
    i = int(np.argmax(above)) if above[0] == False else int(np.argmax(~above))
    # linear interpolation between the bracketing pixels
    y0, y1 = profile[i - 1], profile[i]
    return (i - 1) + (half - y0) / (y1 - y0)


def calibrate_two_point(px1, wl1, px2, wl2, n_px):
    m = (wl2 - wl1) / (px2 - px1)
    b = wl1 - m * px1
    return m * np.arange(n_px) + b


def synthetic_filter_edges(n_px=240):
    """White-light spectra with a 700 nm longpass and 780 nm shortpass in."""
    px = np.arange(n_px)
    true_wl = 660.0 + 0.625 * px                # hidden truth: 0.625 nm/px
    lamp = 1000.0 * np.exp(-0.5 * ((true_wl - 740) / 70) ** 2)
    lp700 = lamp / (1 + np.exp(-(true_wl - 700) / 1.5))
    sp780 = lamp / (1 + np.exp((true_wl - 780) / 1.5))
    noise = lambda: RNG.normal(0, 6, n_px)
    return px, lp700 + noise(), sp780 + noise(), true_wl


# ----------------------------------------------------------------------
# STEP 2: chirp correction (polynomial t0, column re-interpolation)
# ----------------------------------------------------------------------
def t0_of_wavelength(wl, coeffs=(-3.8e-5, 0.072, -31.6)):
    return np.polyval(coeffs, wl)


def make_synthetic_ta_map(wl):
    t = np.concatenate([np.linspace(-2, 2, 41), np.linspace(2.2, 50, 60)])
    t0 = t0_of_wavelength(wl)
    t0 = t0 - t0.mean()
    T, W = np.meshgrid(t, wl, indexing="ij")
    tt = T - t0[None, :]
    rise = 0.5 * (1 + np.tanh(tt / 0.25))
    signal = (-1.0e-3 * np.exp(-0.5 * ((W - 750) / 8) ** 2) * rise
              * np.exp(-np.clip(tt, 0, None) / 6.0)
              + 4.0e-4 * np.exp(-0.5 * ((W - 715) / 12) ** 2) * rise
              * np.exp(-np.clip(tt, 0, None) / 15.0))
    return t, signal + RNG.normal(0, 4e-5, size=signal.shape), t0


def chirp_correct(t, wl, data, coeffs, dt_uniform=0.1):
    t_uni = np.arange(t[0], t[-1], dt_uniform)
    data_uni = np.empty((t_uni.size, wl.size))
    for j in range(wl.size):
        data_uni[:, j] = np.interp(t_uni, t, data[:, j])
    t0 = t0_of_wavelength(wl, coeffs)
    t0 = t0 - t0.mean()
    idx = (t_uni[:, None] + t0[None, :] - t_uni[0]) / dt_uniform
    coords = np.stack([idx, np.broadcast_to(np.arange(wl.size), idx.shape)])
    corrected = map_coordinates(data_uni, coords, order=1, mode="nearest")
    lo = max(0, int(np.ceil(-t0.min() / dt_uniform)))
    hi = corrected.shape[0] - max(0, int(np.ceil(t0.max() / dt_uniform)))
    return t_uni[lo:hi], corrected[lo:hi]


def main():
    # ---- calibration ------------------------------------------------------
    px, prof_lp, prof_sp, true_wl = synthetic_filter_edges()
    p1 = edge_crossing_pixel(prof_lp)
    p2 = edge_crossing_pixel(prof_sp)
    wl = calibrate_two_point(p1, 700.0, p2, 780.0, px.size)
    err = np.max(np.abs(wl - true_wl))
    print(f"edge pixels: {p1:.1f} (700 nm LP), {p2:.1f} (780 nm SP); "
          f"max calibration error = {err:.2f} nm")

    # ---- chirp ------------------------------------------------------------
    t, data, t0_true = make_synthetic_ta_map(wl)
    t_corr, data_corr = chirp_correct(t, wl, data, (-3.8e-5, 0.072, -31.6))

    vmax = np.nanmax(np.abs(data))
    kw = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(px, prof_lp, lw=1, label="700 nm longpass in")
    ax0.plot(px, prof_sp, lw=1, label="780 nm shortpass in")
    ax0.axvline(p1, color="C0", ls=":")
    ax0.axvline(p2, color="C1", ls=":")
    ax0.set(xlabel="camera pixel", ylabel="counts",
            title="filter edges truncate the white light")
    ax0.legend(frameon=False, fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(px, true_wl, "0.7", lw=3, label="hidden truth")
    ax1.plot(px, wl, "r--", lw=1.2, label="two-point calibration")
    ax1.plot([p1, p2], [700, 780], "ko", ms=6)
    ax1.set(xlabel="camera pixel", ylabel="wavelength (nm)",
            title="two (pixel, wavelength) anchors fix the axis")
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(data, extent=[wl[0], wl[-1], t[0], t[-1]], **kw)
    ax2.plot(wl, t0_true, "k--", lw=1)
    ax2.set(ylim=(-2, 8), xlabel="wavelength (nm)", ylabel="delay (ps)",
            title="raw: onset bent by chirp")

    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(data_corr, extent=[wl[0], wl[-1], t_corr[0], t_corr[-1]], **kw)
    ax3.set(ylim=(-2, 8), xlabel="wavelength (nm)", title="chirp-corrected")
    fig.colorbar(im, ax=ax3, label="dR/R")

    j750 = int(np.argmin(np.abs(wl - 750)))
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t, data[:, j750] * 1e3, ".-", ms=3, lw=0.7, color="0.6", label="raw")
    ax4.plot(t_corr, data_corr[:, j750] * 1e3, "-", lw=1.4, color="C3",
             label="corrected")
    ax4.axvline(0, color="k", lw=0.6, ls=":")
    ax4.set(xlabel="delay (ps)", ylabel="dR/R (x10⁻³)", xlim=(-2, 20),
            title="transient at 750 nm")
    ax4.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s3_02_ta_chirp_calibration.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
