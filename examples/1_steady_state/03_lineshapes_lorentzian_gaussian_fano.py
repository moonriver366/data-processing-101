"""
1.03 — Lineshapes: Lorentzian, Gaussian, Fano — and how to start a fit
======================================================================

Once RC (or PL) is on a clean axis, the next question is what function
to fit. My mental model:

  * LORENTZIAN — a single damped resonance. Heavy tails (~1/dE^2).
        L(E) = A * (G/2)^2 / ((E-E0)^2 + (G/2)^2)
  * GAUSSIAN — inhomogeneous broadening / instrument resolution.
    Tails die fast; if your data has visible wings, Gaussian alone
    will underfit them.
  * FANO — a discrete resonance interfering with a broad continuum
    (very common in reflectance of a thin resonant layer on a stack,
    e.g. an hBN-encapsulated monolayer — see the TMM script 1.04):
        F(E) = A * (q + eps)^2 / (1 + eps^2),  eps = 2 (E - E0) / G
    q -> inf recovers a Lorentzian peak; q = 0 is a pure dip
    (anti-resonance); q ~ 1 gives the classic asymmetric shape.
    CAUTION: for finite q the *apparent* extremum is displaced from E0
    by ~G/(2q), so "read the peak position off the plot" fails exactly
    when the shape is asymmetric.

Initial guesses and bounds — the part that actually decides whether
curve_fit converges:
  * center p0: argmax/argmin of the (smoothed) data; bounds = the fit
    window, nothing wider.
  * width p0: count samples above half max; bounds = [instrument
    resolution, window width]. A fit that pegs either end is telling
    you the model or window is wrong.
  * amplitude p0: extremum minus baseline; bounds = [0, ~4x that].
  * baseline p0: median of the window edges.

All data synthetic.

Run:  python 03_lineshapes_lorentzian_gaussian_fano.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=3)


# ----------------------------------------------------------------------
# The three lineshapes (with a constant baseline each)
# ----------------------------------------------------------------------
def lorentzian(E, A, E0, G, c):
    return A * (G / 2) ** 2 / ((E - E0) ** 2 + (G / 2) ** 2) + c


def gaussian(E, A, E0, G, c):
    sigma = G / 2.355                      # parametrized by FWHM like the others
    return A * np.exp(-0.5 * ((E - E0) / sigma) ** 2) + c


def fano(E, A, E0, G, q, c):
    eps = 2.0 * (E - E0) / G
    return A * (q + eps) ** 2 / (1.0 + eps ** 2) + c


# ----------------------------------------------------------------------
def initial_guess(E, y):
    """The p0/bounds recipe from the docstring, written once."""
    base = np.median(np.concatenate([y[:20], y[-20:]]))
    i_ext = int(np.argmax(np.abs(y - base)))
    A0 = y[i_ext] - base
    above = E[np.abs(y - base) > 0.5 * abs(A0)]
    G0 = max(above.max() - above.min(), (E[1] - E[0]) * 3)
    return base, E[i_ext], A0, G0


def fit_and_score(model, E, y, p0, bounds):
    popt, _ = curve_fit(model, E, y, p0=p0, bounds=bounds, maxfev=20000)
    resid = y - model(E, *popt)
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return popt, resid, r2


def main():
    E = np.linspace(1.60, 1.80, 800)

    # ------- panel 1: the q family — one formula, many shapes -------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for q in (0.0, 0.7, 1.5, 3.0, 8.0):
        f = fano(E, 0.01, 1.70, 0.012, q, 0.0)
        axes[0].plot(E, f / np.abs(f).max(), lw=1.3, label=f"q = {q:g}")
    axes[0].set(xlabel="energy (eV)", ylabel="normalized signal",
                title="Fano family: q=0 dip -> q>>1 Lorentzian", xlim=(1.66, 1.74))
    axes[0].legend(frameon=False, fontsize=8)

    # ------- panel 2: tails decide Gaussian vs Lorentzian ------------------
    y_lor = lorentzian(E, 1.0, 1.70, 0.012, 0.0)
    y_gau = gaussian(E, 1.0, 1.70, 0.012, 0.0)
    axes[1].semilogy(E, y_lor + 1e-6, lw=1.4, label="Lorentzian")
    axes[1].semilogy(E, y_gau + 1e-6, lw=1.4, label="Gaussian (same FWHM)")
    axes[1].set(xlabel="energy (eV)", ylabel="signal (log)", ylim=(1e-5, 2),
                title="same FWHM, very different wings —\nlook at your tails on a log axis")
    axes[1].legend(frameon=False, fontsize=8)

    # ------- panel 3: fit an asymmetric resonance with all three ----------
    truth = dict(A=0.012, E0=1.700, G=0.014, q=1.3, c=0.002)
    y = fano(E, **truth) + RNG.normal(0, 4e-4, size=E.size)

    base, E0g, A0, G0 = initial_guess(E, y)
    p0_LG = [A0, E0g, G0, base]
    b_LG = ([0, E.min(), 0.002, -np.inf], [4 * abs(A0) + 1e-9, E.max(), 0.1, np.inf])
    popt_L, res_L, r2_L = fit_and_score(lorentzian, E, y, p0_LG, b_LG)
    popt_G, res_G, r2_G = fit_and_score(gaussian, E, y, p0_LG, b_LG)
    p0_F = [abs(A0) / 2, E0g, G0, 1.0, base]
    b_F = ([0, E.min(), 0.002, -30, -np.inf], [np.inf, E.max(), 0.1, 30, np.inf])
    popt_F, res_F, r2_F = fit_and_score(fano, E, y, p0_F, b_F)

    axes[2].plot(E, y, ".", ms=2, color="0.7", label="data")
    axes[2].plot(E, lorentzian(E, *popt_L), lw=1.2, label=f"Lorentzian R²={r2_L:.3f}")
    axes[2].plot(E, gaussian(E, *popt_G), lw=1.2, label=f"Gaussian R²={r2_G:.3f}")
    axes[2].plot(E, fano(E, *popt_F), "k-", lw=1.4, label=f"Fano R²={r2_F:.3f}")
    axes[2].axvline(truth["E0"], color="0.5", ls=":", lw=1)
    axes[2].set(xlabel="energy (eV)", ylabel="RC",
                title="asymmetric data: symmetric models\nmiss it AND bias the center",
                xlim=(1.66, 1.75))
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s1_03_lineshapes.png", dpi=150)
    plt.show()

    print(f"true E0 = {truth['E0']:.4f} eV, true q = {truth['q']}")
    print(f"Lorentzian center: {popt_L[1]:.4f} eV   <- pulled by asymmetry")
    print(f"Gaussian   center: {popt_G[1]:.4f} eV   <- pulled by asymmetry")
    print(f"Fano       center: {popt_F[1]:.4f} eV, q = {popt_F[3]:.2f}")


if __name__ == "__main__":
    main()
