"""
Example 03 — TRPL: IRF-convolved multi-exponential fit + 1/e lifetime
=====================================================================

Workflow demonstrated (as used for time-resolved PL from a PicoHarp /
SPAD histogram):

1. Simulate a TRPL histogram: a bi-exponential decay convolved with a
   Gaussian instrument response function (IRF), plus a constant dark
   background, with Poisson counting noise.
2. Fit the analytic IRF-convolved model (the exponentially modified
   Gaussian used by Origin's ConvolutedDecay):

       y(t) = sum_i  A_i/2 * exp( (w^2 - 2*t_i*(t - t0)) / (2*t_i^2) )
                     * erfc( (w^2 - t_i*(t - t0)) / (sqrt(2)*w*t_i) )
              + y0

   where w is the IRF sigma, t_i the decay constants, t0 the pulse
   arrival time. Convolving analytically avoids deconvolution noise.
3. Extract a model-free 1/e lifetime from the smoothed curve as a
   sanity check on the fitted time constants.

No real data is used anywhere — everything is synthesized in-script.

Run:  python 03_trpl_irf_convolved_decay.py
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=3)


# ----------------------------------------------------------------------
# Model: exponential decay convolved with a Gaussian IRF (analytic form)
# ----------------------------------------------------------------------
def _conv_term(t, A, tau, t0, w):
    """One exponentially-modified-Gaussian component.

    Equals (exp(-t/tau) * H(t)) convolved with a normalized Gaussian of
    sigma w, scaled by A. Written to stay finite for large exponents.
    """
    arg_exp = (w * w - 2.0 * tau * (t - t0)) / (2.0 * tau * tau)
    arg_erfc = (w * w - tau * (t - t0)) / (np.sqrt(2.0) * w * tau)
    # exp() can overflow before erfc() kills it; clip in log-space
    out = np.zeros_like(t, dtype=float)
    safe = arg_exp < 700.0
    out[safe] = 0.5 * A * np.exp(arg_exp[safe]) * erfc(arg_erfc[safe])
    return out


def biexp_irf(t, y0, t0, A1, tau1, A2, tau2, w):
    """Bi-exponential decay convolved with Gaussian IRF + flat background."""
    return y0 + _conv_term(t, A1, tau1, t0, w) + _conv_term(t, A2, tau2, t0, w)


# ----------------------------------------------------------------------
# Model-free 1/e lifetime
# ----------------------------------------------------------------------
def smooth_boxcar(y, window=9):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def tau_1e_from_peak(t, y):
    """Time from the curve maximum to the first crossing of peak/e.

    Uses linear interpolation between the bracketing samples, exactly as
    done for quick lifetime maps where a full fit is too slow.
    """
    y = np.nan_to_num(y, nan=0.0).astype(float)
    imax = int(np.argmax(y))
    tail = y[imax:] / y[imax]
    below = np.where(tail <= 1.0 / np.e)[0]
    if below.size == 0:
        return np.nan
    j = int(below[0])
    if j == 0:
        return 0.0
    y0_, y1_ = tail[j - 1], tail[j]
    frac = (1.0 / np.e - y0_) / (y1_ - y0_)
    t_cross = t[imax + j - 1] + frac * (t[imax + j] - t[imax + j - 1])
    return t_cross - t[imax]


# ----------------------------------------------------------------------
# Synthetic TRPL histogram
# ----------------------------------------------------------------------
def make_synthetic_trpl():
    dt_ns = 0.128                                     # 128 ps time bin
    t = np.arange(0.0, 100.0, dt_ns)                  # 100 ns window
    truth = dict(y0=2.0, t0=8.0, A1=4000.0, tau1=1.2, A2=600.0, tau2=12.0, w=0.35)
    expected = biexp_irf(t, **truth)
    counts = RNG.poisson(np.clip(expected, 0, None)).astype(float)
    return t, counts, truth


def main():
    t, counts, truth = make_synthetic_trpl()

    # -- fit with Poisson-ish weighting (sigma = sqrt(max(y,1))) ---------
    p0 = [counts[:20].mean(),                    # y0 from pre-pulse region
          t[np.argmax(counts)],                  # t0 from histogram peak
          counts.max(), 1.0, counts.max() / 10, 10.0, 0.3]
    lo = [0, 0, 0, 0.01, 0, 0.5, 0.05]
    hi = [np.inf, t[-1], np.inf, 50.0, np.inf, 80.0, 2.0]
    sigma = np.sqrt(np.clip(counts, 1, None))
    popt, pcov = curve_fit(biexp_irf, t, counts, p0=p0, bounds=(lo, hi),
                           sigma=sigma, maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    names = ["y0", "t0", "A1", "tau1", "A2", "tau2", "w"]

    # -- model-free 1/e lifetime -----------------------------------------
    tau_1e = tau_1e_from_peak(t, smooth_boxcar(counts, 9))

    # -- amplitude-weighted average lifetime ------------------------------
    A1, tau1, A2, tau2 = popt[2], popt[3], popt[4], popt[5]
    tau_avg = (A1 * tau1 + A2 * tau2) / (A1 + A2)

    # -- plots -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(t, np.clip(counts, 0.5, None), ".", ms=2, color="0.6",
                     label="synthetic counts")
    axes[0].semilogy(t, biexp_irf(t, *popt), "r-", lw=1.3,
                     label=(f"IRF-convolved fit\n"
                            f"tau1={popt[3]:.2f} ns, tau2={popt[5]:.1f} ns"))
    axes[0].axvline(popt[1], color="0.5", ls=":", lw=0.8, label=f"t0={popt[1]:.1f} ns")
    axes[0].set(xlabel="time (ns)", ylabel="counts", title="TRPL fit (log scale)",
                xlim=(0, 80))
    axes[0].legend(frameon=False, fontsize=8)

    resid = (counts - biexp_irf(t, *popt)) / sigma
    axes[1].plot(t, resid, lw=0.6, color="C0")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="time (ns)", ylabel="weighted residual",
                title="residuals (should be structureless)", xlim=(0, 80),
                ylim=(-5, 5))
    fig.tight_layout()
    fig.savefig("03_trpl_irf_convolved_decay.png", dpi=150)
    plt.show()

    print("param   truth    fit")
    for n_, tr, po, pe in zip(names, truth.values(), popt, perr):
        print(f"{n_:5s} {tr:8.3g} {po:8.3g} +/- {pe:.2g}")
    print(f"model-free tau_1e = {tau_1e:.2f} ns | amplitude-weighted tau = {tau_avg:.2f} ns")


if __name__ == "__main__":
    main()
