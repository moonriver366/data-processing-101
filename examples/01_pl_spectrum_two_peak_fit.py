"""
Example 01 — PL spectrum: baseline correction + windowed two-peak fit
=====================================================================

Workflow demonstrated (as used for gated-TMDC PL spectra):

1. Generate a synthetic PL spectrum with two species (neutral exciton X0
   and trion X-) on a sloped background with Poisson-like noise.
2. Baseline-correct using the mean of a signal-free wavelength window
   (same idea as subtracting the tail region of an Andor .asc spectrum).
3. Fit each peak inside a +/- window around a center guess, with
   physically bounded width (gamma) and a quality gate (SNR, R^2).
4. Fit both peaks globally and plot the decomposition.

No real data is used anywhere — everything is synthesized in-script.

Run:  python 01_pl_spectrum_two_peak_fit.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=7)


# ----------------------------------------------------------------------
# Model functions
# ----------------------------------------------------------------------
def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak: A * g^2 / ((x - c)^2 + g^2). FWHM = 2*gamma."""
    dx = x - center
    return amplitude * gamma * gamma / (dx * dx + gamma * gamma)


def gaussian(x, amplitude, center, sigma):
    """Gaussian peak: A * exp(-0.5 * (x - c)^2 / s^2)."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def two_lorentzians(x, a1, c1, g1, a2, c2, g2):
    return lorentzian(x, a1, c1, g1) + lorentzian(x, a2, c2, g2)


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------
def make_synthetic_spectrum():
    """Synthetic WSe2-like PL spectrum: X0 ~ 750 nm, trion ~ 764 nm."""
    wl = np.linspace(700.0, 820.0, 1024)                  # wavelength axis (nm)
    truth = dict(x0=dict(A=1800.0, c=750.0, g=4.0),        # neutral exciton
                 xm=dict(A=900.0, c=764.0, g=6.5))         # trion
    clean = (lorentzian(wl, **{k: v for k, v in zip("amplitude center gamma".split(),
                                                    truth["x0"].values())})
             + lorentzian(wl, **{k: v for k, v in zip("amplitude center gamma".split(),
                                                      truth["xm"].values())}))
    background = 120.0 + 0.35 * (wl - 700.0)               # sloped CCD background
    noisy = clean + background
    noisy = RNG.poisson(np.clip(noisy, 0, None)).astype(float)  # shot noise
    return wl, noisy, truth


# ----------------------------------------------------------------------
# Processing steps
# ----------------------------------------------------------------------
def baseline_correct(wl, spectrum, tail_lo=800.0, tail_hi=820.0):
    """Subtract a linear baseline estimated from a signal-free window.

    In the lab code this is done per camera row; a robust variant uses the
    10th percentile of the fit window instead of the mean.
    """
    tail = (wl >= tail_lo) & (wl <= tail_hi)
    head = (wl >= wl[0]) & (wl <= wl[0] + 20.0)
    # two anchor points -> linear baseline across the full axis
    x_anchor = np.array([wl[head].mean(), wl[tail].mean()])
    y_anchor = np.array([np.median(spectrum[head]), np.median(spectrum[tail])])
    slope = (y_anchor[1] - y_anchor[0]) / (x_anchor[1] - x_anchor[0])
    baseline = y_anchor[0] + slope * (wl - x_anchor[0])
    return spectrum - baseline


def noise_level_mad(y):
    """Noise floor via median absolute deviation of first differences.

    MAD is robust to the peaks themselves contaminating the estimate.
    """
    d = np.diff(y[np.isfinite(y)])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def fit_peak_windowed(wl, y, center_guess, window_nm=15.0,
                      gamma_bounds=(0.4, 15.0), min_snr=2.0):
    """Fit one Lorentzian inside +/- window_nm around center_guess.

    Returns None when the fit fails the quality gate — downstream maps
    then show a gap instead of a garbage point.
    """
    m = np.abs(wl - center_guess) <= window_nm
    if m.sum() < 7:
        return None
    x, yy = wl[m], y[m]
    yy = yy - np.percentile(yy, 10)          # local baseline inside the window
    peak = float(np.max(yy))
    # width guess from half-maximum crossing count
    above = x[yy > 0.5 * peak]
    g0 = max(0.5 * (above.max() - above.min()), gamma_bounds[0] * 1.5) if above.size > 1 else 2.0
    p0 = [peak, x[np.argmax(yy)], g0]
    lo = [0.0, center_guess - window_nm, gamma_bounds[0]]
    hi = [4.0 * peak, center_guess + window_nm, gamma_bounds[1]]
    try:
        popt, pcov = curve_fit(lorentzian, x, yy, p0=p0, bounds=(lo, hi), maxfev=10000)
    except RuntimeError:
        return None
    resid = yy - lorentzian(x, *popt)
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
    snr = popt[0] / max(noise_level_mad(yy), 1e-12)
    if snr < min_snr or r2 < 0.5:
        return None
    return dict(amplitude=popt[0], center_nm=popt[1], gamma_nm=popt[2],
                fwhm_nm=2.0 * popt[2], r2=r2, snr=snr)


def fit_two_peaks_global(wl, y, guess1, guess2, window_nm=15.0):
    """Sequential-then-global strategy: fit each peak alone for a good
    starting point, then refine both together so overlapping tails are
    shared correctly."""
    f1 = fit_peak_windowed(wl, y, guess1, window_nm)
    f2 = fit_peak_windowed(wl, y, guess2, window_nm)
    if f1 is None or f2 is None:
        return None
    p0 = [f1["amplitude"], f1["center_nm"], f1["gamma_nm"],
          f2["amplitude"], f2["center_nm"], f2["gamma_nm"]]
    m = (wl > min(guess1, guess2) - 3 * window_nm) & (wl < max(guess1, guess2) + 3 * window_nm)
    lo = [0, guess1 - window_nm, 0.4, 0, guess2 - window_nm, 0.4]
    hi = [np.inf, guess1 + window_nm, 15.0, np.inf, guess2 + window_nm, 15.0]
    popt, _ = curve_fit(two_lorentzians, wl[m], y[m], p0=p0, bounds=(lo, hi), maxfev=20000)
    return popt


# ----------------------------------------------------------------------
def main():
    wl, raw, truth = make_synthetic_spectrum()
    corrected = baseline_correct(wl, raw)
    popt = fit_two_peaks_global(wl, corrected, guess1=750.0, guess2=764.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(wl, raw, lw=0.8, color="0.4", label="raw (with background)")
    axes[0].plot(wl, corrected, lw=0.9, color="C0", label="baseline-corrected")
    axes[0].set(xlabel="wavelength (nm)", ylabel="counts", title="Baseline correction")
    axes[0].legend(frameon=False)

    axes[1].plot(wl, corrected, ".", ms=2, color="0.6", label="data")
    if popt is not None:
        axes[1].plot(wl, two_lorentzians(wl, *popt), "k-", lw=1.4, label="global fit")
        axes[1].plot(wl, lorentzian(wl, *popt[:3]), "--", color="C3",
                     label=f"X$^0$: {popt[1]:.1f} nm, FWHM {2*popt[2]:.1f} nm")
        axes[1].plot(wl, lorentzian(wl, *popt[3:]), "--", color="C2",
                     label=f"X$^-$: {popt[4]:.1f} nm, FWHM {2*popt[5]:.1f} nm")
    axes[1].set(xlabel="wavelength (nm)", ylabel="counts",
                title="Two-peak decomposition", xlim=(720, 800))
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("01_pl_spectrum_two_peak_fit.png", dpi=150)
    plt.show()

    if popt is not None:
        print("truth : X0 c=750.0 g=4.0 | X- c=764.0 g=6.5")
        print(f"fitted: X0 c={popt[1]:.2f} g={popt[2]:.2f} | "
              f"X- c={popt[4]:.2f} g={popt[5]:.2f}")


if __name__ == "__main__":
    main()
