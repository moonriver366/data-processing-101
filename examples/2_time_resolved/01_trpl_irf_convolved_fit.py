"""
2.01 — TRPL: fitting a time trace, and what the IRF is
======================================================

The raw product of TCSPC (PicoHarp-style counter or a SPAD) is a
histogram: photon counts vs arrival time. Two numbers define the file:
the time bin width (ns/channel) and the number of channels; everything
else is counts. That histogram is NOT the decay — it is the decay
convolved with the instrument response function (IRF): laser pulse
width + detector jitter + electronics, typically tens to hundreds of
ps. When lifetimes approach the IRF width, ignoring it inflates them.

My approach: never deconvolve (it amplifies noise). Fit the analytic
convolution of exponentials with a Gaussian IRF instead — the
exponentially-modified Gaussian:

    y(t) = sum_i  A_i/2 * exp( (w^2 - 2 tau_i (t - t0)) / (2 tau_i^2) )
                  * erfc( (w^2 - tau_i (t - t0)) / (sqrt(2) w tau_i) )
           + y0

with w the IRF sigma, t0 the pulse arrival, y0 the dark floor
(estimated from pre-pulse bins). Weight the fit by sqrt(counts)
(Poisson). Cross-check with the model-free 1/e time — if the fit and
the 1/e time tell different stories, the model is wrong, not the data.

All data synthetic (Poisson noise on a known bi-exponential).

Run:  python 01_trpl_irf_convolved_fit.py
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
    arg_exp = (w * w - 2.0 * tau * (t - t0)) / (2.0 * tau * tau)
    arg_erfc = (w * w - tau * (t - t0)) / (np.sqrt(2.0) * w * tau)
    out = np.zeros_like(t, dtype=float)
    safe = arg_exp < 700.0                      # exp() overflow guard
    out[safe] = 0.5 * A * np.exp(arg_exp[safe]) * erfc(arg_erfc[safe])
    return out


def biexp_irf(t, y0, t0, A1, tau1, A2, tau2, w):
    return y0 + _conv_term(t, A1, tau1, t0, w) + _conv_term(t, A2, tau2, t0, w)


# ----------------------------------------------------------------------
# Model-free 1/e lifetime — the honest cross-check
# ----------------------------------------------------------------------
def tau_1e_from_peak(t, y, smooth_window=9):
    kernel = np.ones(smooth_window) / smooth_window
    ys = np.convolve(np.nan_to_num(y, nan=0.0), kernel, mode="same")
    imax = int(np.argmax(ys))
    tail = ys[imax:] / ys[imax]
    below = np.where(tail <= 1.0 / np.e)[0]
    if below.size == 0:
        return np.nan
    j = int(below[0])
    frac = (1.0 / np.e - tail[j - 1]) / (tail[j] - tail[j - 1])
    return t[imax + j - 1] + frac * (t[imax + j] - t[imax + j - 1]) - t[imax]


def make_synthetic_histogram():
    dt_ns = 0.128                                # 128 ps/channel
    t = np.arange(0.0, 100.0, dt_ns)
    truth = dict(y0=2.0, t0=8.0, A1=4000.0, tau1=1.2, A2=600.0, tau2=12.0, w=0.35)
    counts = RNG.poisson(np.clip(biexp_irf(t, **truth), 0, None)).astype(float)
    return t, counts, truth


def main():
    t, counts, truth = make_synthetic_histogram()

    # p0 from the data itself: dark floor from pre-pulse bins, t0 from peak
    p0 = [counts[:20].mean(), t[np.argmax(counts)],
          counts.max(), 1.0, counts.max() / 10, 10.0, 0.3]
    lo = [0, 0, 0, 0.01, 0, 0.5, 0.05]
    hi = [np.inf, t[-1], np.inf, 50.0, np.inf, 80.0, 2.0]
    sigma = np.sqrt(np.clip(counts, 1, None))
    popt, pcov = curve_fit(biexp_irf, t, counts, p0=p0, bounds=(lo, hi),
                           sigma=sigma, maxfev=40000)
    perr = np.sqrt(np.diag(pcov))

    tau_1e = tau_1e_from_peak(t, counts)
    A1, tau1, A2, tau2 = popt[2], popt[3], popt[4], popt[5]
    tau_avg = (A1 * tau1 + A2 * tau2) / (A1 + A2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(t, np.clip(counts, 0.5, None), ".", ms=2, color="0.6",
                     label="counts (synthetic)")
    axes[0].semilogy(t, biexp_irf(t, *popt), "r-", lw=1.3,
                     label=f"fit: tau1={popt[3]:.2f} ns, tau2={popt[5]:.1f} ns, "
                           f"IRF w={popt[6]*1e3:.0f} ps")
    axes[0].set(xlabel="time (ns)", ylabel="counts", xlim=(0, 80),
                title="two slopes = two lifetimes; the rounded\nrising edge is the IRF")
    axes[0].legend(frameon=False, fontsize=8)

    resid = (counts - biexp_irf(t, *popt)) / sigma
    axes[1].plot(t, resid, lw=0.6, color="C0")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="time (ns)", ylabel="weighted residual", xlim=(0, 80),
                ylim=(-5, 5), title="residuals: structure here means the\nmodel (n of exponentials) is wrong")
    fig.tight_layout()
    fig.savefig("s2_01_trpl_irf_fit.png", dpi=150)
    plt.show()

    print("param   truth    fit")
    for name, tr, po, pe in zip(["y0", "t0", "A1", "tau1", "A2", "tau2", "w"],
                                truth.values(), popt, perr):
        print(f"{name:5s} {tr:8.3g} {po:8.3g} +/- {pe:.2g}")
    print(f"model-free tau_1e = {tau_1e:.2f} ns | "
          f"amplitude-weighted tau = {tau_avg:.2f} ns")


if __name__ == "__main__":
    main()
