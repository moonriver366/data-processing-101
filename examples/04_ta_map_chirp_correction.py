"""
Example 04 — Transient absorption map: chirp correction + linecuts
==================================================================

Workflow demonstrated (as used for TA / TAM pump-probe maps):

1. Simulate a dR/R map on a (delay, wavelength) grid: a ground-state
   bleach and a photoinduced absorption, each decaying exponentially,
   whose time zero t0 depends on wavelength (probe "chirp" from group
   velocity dispersion in the optics).
2. Model the chirp as a polynomial t0(lambda) — in practice the
   coefficients come from fitting the coherent artifact — and undo it
   by re-interpolating every wavelength column onto a common time axis
   (scipy.ndimage.map_coordinates, same as the lab pipeline).
3. Extract the standard linecuts: a transient (dR/R vs delay at fixed
   wavelength) and a spectrum (dR/R vs wavelength at fixed delay),
   before and after correction.

No real data is used anywhere — everything is synthesized in-script.

Run:  python 04_ta_map_chirp_correction.py
"""

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=42)


# ----------------------------------------------------------------------
# Synthetic chirped TA map
# ----------------------------------------------------------------------
def t0_of_wavelength(wl, coeffs=(-3.8e-5, 0.072, -31.6)):
    """Chirp model: polynomial arrival time t0(lambda) in ps.

    The quadratic form (and even these magnitudes) mirror what a fit to
    the coherent artifact typically returns for a visible probe.
    """
    return np.polyval(coeffs, wl)


def make_synthetic_ta_map():
    wl = np.linspace(680.0, 800.0, 240)              # probe wavelength (nm)
    t = np.concatenate([np.linspace(-2, 2, 41),      # dense around t0
                        np.linspace(2.2, 50, 60)])    # sparse at long delay
    t0 = t0_of_wavelength(wl)
    t0 = t0 - t0.mean()                               # center chirp around 0

    def bleach(wl_, c, g):                            # Gaussian spectral shapes
        return np.exp(-0.5 * ((wl_ - c) / g) ** 2)

    T, W = np.meshgrid(t, wl, indexing="ij")          # (n_t, n_wl)
    tt = T - t0[None, :]                              # local (chirped) delay
    rise = 0.5 * (1 + np.tanh(tt / 0.25))             # ~pump-probe cross-correlation
    signal = (-1.0e-3 * bleach(W, 750, 8) * rise * np.exp(-np.clip(tt, 0, None) / 6.0)
              + 4.0e-4 * bleach(W, 715, 12) * rise * np.exp(-np.clip(tt, 0, None) / 15.0))
    noisy = signal + RNG.normal(0, 4e-5, size=signal.shape)
    return t, wl, noisy, t0


# ----------------------------------------------------------------------
# Chirp correction
# ----------------------------------------------------------------------
def chirp_correct(t, wl, data, coeffs, dt_uniform=0.1):
    """Undo wavelength-dependent time zero.

    Steps (mirroring the lab implementation):
      1. interpolate the map onto a uniform time grid,
      2. shift each wavelength column by its own t0(lambda) using
         map_coordinates (spline interpolation along the time axis),
      3. return the corrected map on the uniform grid, cropped to the
         time window where every wavelength has valid data.
    """
    t_uni = np.arange(t[0], t[-1], dt_uniform)
    # uniform-time resampling, column by column (linear)
    data_uni = np.empty((t_uni.size, wl.size))
    for j in range(wl.size):
        data_uni[:, j] = np.interp(t_uni, t, data[:, j])

    t0 = t0_of_wavelength(wl, coeffs)
    t0 = t0 - t0.mean()
    # fractional index of (t_uni + t0_j) in the uniform grid for column j
    idx = (t_uni[:, None] + t0[None, :] - t_uni[0]) / dt_uniform
    coords = np.stack([idx, np.broadcast_to(np.arange(wl.size), idx.shape)])
    corrected = map_coordinates(data_uni, coords, order=1, mode="nearest")

    # crop to the window valid for all columns
    lo = max(0, int(np.ceil(-t0.min() / dt_uniform)))
    hi = corrected.shape[0] - max(0, int(np.ceil(t0.max() / dt_uniform)))
    return t_uni[lo:hi], corrected[lo:hi]


def main():
    t, wl, data, t0_true = make_synthetic_ta_map()
    t_corr, data_corr = chirp_correct(t, wl, data, (-3.8e-5, 0.072, -31.6))

    vmax = np.nanmax(np.abs(data))
    kw = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(data, extent=[wl[0], wl[-1], t[0], t[-1]], **kw)
    ax0.plot(wl, t0_true, "k--", lw=1, label="true $t_0(\\lambda)$")
    ax0.set(ylim=(-2, 8), xlabel="wavelength (nm)", ylabel="delay (ps)",
            title="raw map (chirped)")
    ax0.legend(frameon=False, fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(data_corr, extent=[wl[0], wl[-1], t_corr[0], t_corr[-1]], **kw)
    ax1.set(ylim=(-2, 8), xlabel="wavelength (nm)", title="chirp-corrected")

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(data_corr, extent=[wl[0], wl[-1], t_corr[0], t_corr[-1]], **kw)
    ax2.set(xlabel="wavelength (nm)", title="corrected, full range")
    fig.colorbar(im2, ax=ax2, label="$\\Delta R/R$")

    # -- linecuts ---------------------------------------------------------
    j750 = int(np.argmin(np.abs(wl - 750)))
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(t, data[:, j750] * 1e3, ".-", ms=3, lw=0.7, label="raw", color="0.6")
    ax3.plot(t_corr, data_corr[:, j750] * 1e3, "-", lw=1.4, label="corrected", color="C3")
    ax3.axvline(0, color="k", lw=0.6, ls=":")
    ax3.set(xlabel="delay (ps)", ylabel="$\\Delta R/R$ ($\\times 10^{-3}$)",
            title="transient at 750 nm (bleach)", xlim=(-2, 30))
    ax3.legend(frameon=False, fontsize=8)

    i1ps = int(np.argmin(np.abs(t_corr - 1.0)))
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(wl, data_corr[i1ps] * 1e3, lw=1.2)
    ax4.axhline(0, color="k", lw=0.6)
    ax4.set(xlabel="wavelength (nm)", ylabel="$\\Delta R/R$ ($\\times 10^{-3}$)",
            title="spectrum at +1 ps")

    fig.tight_layout()
    fig.savefig("04_ta_map_chirp_correction.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
