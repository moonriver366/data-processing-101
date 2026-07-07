"""
1.06 — Sweeps: tracking peaks across power / temperature / gate
===============================================================

Almost every sample gets swept against SOMETHING — power, temperature,
gate voltage. The data becomes a stack of spectra vs a control knob,
and the question becomes "where did peak A go, and what grew in its
place". Two fitting strategies:

  * INDEPENDENT fits: every spectrum fitted from scratch with the same
    generic p0. Robust when peaks are well separated; falls apart when
    two peaks overlap or one gets weak — the optimizer happily swaps
    peak identities between neighboring steps.

  * SEQUENTIAL (constrained) fits: sweep in order, use step k's result
    as p0 for step k+1 and tighten the bounds to a window around it
    (e.g. center within +/- a few linewidths, amplitude within a
    factor). Identity is preserved by CONTINUITY, which is a physical
    assumption — the knob changes things smoothly. When that
    assumption is wrong (abrupt transition), sequential fitting will
    smooth over it, so always look at the residual map too.

Quality gate as always: a step whose fit fails (bad R^2, pegged
bounds) records NaN — a visible gap beats a silently wrong point.

All data synthetic: two generic peaks A and B; the knob transfers
weight from A to B and shifts B, with an overlap region in the middle.

Run:  python 06_parameter_sweep_peak_tracking.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=6)


def gaussian(E, A, mu, s):
    return A * np.exp(-0.5 * ((E - mu) / s) ** 2)


def two_peaks(E, A1, mu1, s1, A2, mu2, s2):
    return gaussian(E, A1, mu1, s1) + gaussian(E, A2, mu2, s2)


# ----------------------------------------------------------------------
def make_sweep(n_steps=50):
    """Knob k in [0,1]: peak A fades, peak B grows, shifts and crosses A."""
    E = np.linspace(1.60, 1.80, 500)
    knob = np.linspace(0, 1, n_steps)
    truth, spectra = [], []
    for k in knob:
        A1, mu1, s1 = 900 * (1 - 0.8 * k), 1.712, 0.006
        A2, mu2, s2 = 850 * k, 1.690 + 0.030 * k, 0.009   # B sweeps THROUGH A
        truth.append((A1, mu1, A2, mu2))
        clean = two_peaks(E, A1, mu1, s1, A2, mu2, s2) + 20.0
        spectra.append(RNG.poisson(np.clip(clean, 0, None)).astype(float) - 20.0)
    return E, knob, np.array(spectra), np.array(truth)


# ----------------------------------------------------------------------
def fit_independent(E, spectra):
    """Same generic p0 for every step."""
    out = []
    for spec in spectra:
        p0 = [spec.max(), 1.712, 0.006, spec.max() / 2, 1.700, 0.009]
        lo = [0, 1.60, 0.003, 0, 1.60, 0.003]
        hi = [np.inf, 1.80, 0.03, np.inf, 1.80, 0.03]
        try:
            popt, _ = curve_fit(two_peaks, E, spec, p0=p0, bounds=(lo, hi),
                                maxfev=20000)
            out.append(popt)
        except RuntimeError:
            out.append([np.nan] * 6)
    return np.array(out)


def fit_sequential(E, spectra, mu_window=0.006, amp_factor=2.0):
    """Previous popt becomes p0; bounds tighten around it."""
    out, prev = [], None
    for spec in spectra:
        if prev is None:
            p0 = [spec.max(), 1.712, 0.006, 1.0, 1.690, 0.009]
            lo = [0, 1.70, 0.003, 0, 1.65, 0.003]
            hi = [np.inf, 1.73, 0.03, np.inf, 1.72, 0.03]
        else:
            p0 = list(prev)
            lo = [max(prev[0] / amp_factor, 0), prev[1] - mu_window, 0.003,
                  max(prev[3] / amp_factor, 0), prev[4] - mu_window, 0.003]
            hi = [max(prev[0] * amp_factor, 50), prev[1] + mu_window, 0.03,
                  max(prev[3] * amp_factor, 50), prev[4] + mu_window, 0.03]
            p0 = [min(max(p, l), h) for p, l, h in zip(p0, lo, hi)]
        try:
            popt, _ = curve_fit(two_peaks, E, spec, p0=p0, bounds=(lo, hi),
                                maxfev=20000)
            out.append(popt)
            prev = popt
        except RuntimeError:
            out.append([np.nan] * 6)   # keep prev: bridge single bad steps
    return np.array(out)


def main():
    E, knob, spectra, truth = make_sweep()
    ind = fit_independent(E, spectra)
    seq = fit_sequential(E, spectra)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    im = axes[0].imshow(spectra, extent=[E[0], E[-1], knob[0], knob[-1]],
                        aspect="auto", origin="lower", cmap="magma")
    axes[0].plot(truth[:, 1], knob, "w:", lw=1)
    axes[0].plot(truth[:, 3], knob, "c:", lw=1)
    axes[0].set(xlabel="energy (eV)", ylabel="control knob (a.u.)",
                title="the sweep: B crosses A around knob ~ 0.7")
    fig.colorbar(im, ax=axes[0], label="counts")

    for ax, fit, name in ((axes[1], ind, "independent fits"),
                          (axes[2], seq, "sequential constrained fits")):
        ax.plot(knob, truth[:, 1], "-", color="0.75", lw=3, label="truth A")
        ax.plot(knob, truth[:, 3], "-", color="#b7e0f2", lw=3, label="truth B")
        ax.plot(knob, fit[:, 1], "o", ms=3, color="C3", label="fit A center")
        ax.plot(knob, fit[:, 4], "s", ms=3, color="C0", label="fit B center")
        ax.set(xlabel="control knob (a.u.)", ylabel="peak center (eV)",
               title=name, ylim=(1.675, 1.735))
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s1_06_sweep_tracking.png", dpi=150)
    plt.show()

    # count identity swaps (fitted A center jumping to B's track)
    swaps_ind = int(np.nansum(np.abs(ind[:, 1] - truth[:, 1]) >
                              np.abs(ind[:, 1] - truth[:, 3])))
    swaps_seq = int(np.nansum(np.abs(seq[:, 1] - truth[:, 1]) >
                              np.abs(seq[:, 1] - truth[:, 3])))
    print(f"steps where 'A' actually locked onto B: independent = {swaps_ind}, "
          f"sequential = {swaps_seq}")


if __name__ == "__main__":
    main()
