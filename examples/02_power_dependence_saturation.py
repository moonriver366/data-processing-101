"""
Example 02 — Power dependence: filename metadata, saturation fit, dI/dlogP
==========================================================================

Workflow demonstrated (as used for power-dependent PL of gated TMDCs):

1. Simulate a power sweep the way it arrives from the lab: one spectrum
   per file, with the excitation power encoded in the filename
   (e.g. ``PL_P1p5uW_Vf0.asc`` -> 1.5 uW). Parse it back with regex.
2. Integrate each spectrum to get I(P) and plot on log-log axes,
   where power laws I ~ P^k appear as straight lines of slope k.
3. Fit a Hill-type saturation model
       I(P) = Imax * P^n / (K^n + P^n) + c
4. Compute the NaN-aware derivative dI/d(log10 P) — a sensitive way to
   locate where the emission starts to saturate.

No real data is used anywhere — everything is synthesized in-script.

Run:  python 02_power_dependence_saturation.py
"""

import re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=11)

# ----------------------------------------------------------------------
# Filename metadata convention: p = decimal point, m = minus sign
#   P1p5uW  -> power 1.5 uW        Vfm2p5 -> front gate -2.5 V
# ----------------------------------------------------------------------
POWER_RE = re.compile(r"P(?P<val>m?\d+p?\d*)(?P<unit>nW|uW|mW|W)")
UNIT_SCALE = {"nW": 1e-9, "uW": 1e-6, "mW": 1e-3, "W": 1.0}


def parse_power_from_name(name):
    """'PL_P1p5uW_Vf0.asc' -> 1.5e-6 (watts)."""
    m = POWER_RE.search(name)
    if m is None:
        return None
    val = float(m.group("val").replace("m", "-").replace("p", "."))
    return val * UNIT_SCALE[m.group("unit")]


def format_power_token(power_w):
    """Inverse mapping used when *writing* measurement files."""
    for unit in ("W", "mW", "uW", "nW"):
        v = power_w / UNIT_SCALE[unit]
        if 1.0 <= v < 1000.0:
            token = f"{v:.4g}".replace("-", "m").replace(".", "p")
            return f"P{token}{unit}"
    return f"P{power_w:.3g}W"


# ----------------------------------------------------------------------
# Physics model
# ----------------------------------------------------------------------
def hill_saturation(P, Imax, n, K, c):
    """Hill model: linear (slope n on log-log) at low P, saturates at Imax."""
    return Imax * P ** n / (K ** n + P ** n) + c


def derivative_dlogp_nan(intensity, powers_w):
    """Centered finite difference of I against log10(P), tolerant of NaNs.

    Edge points fall back to one-sided differences.
    """
    logp = np.where(powers_w > 0, np.log10(powers_w), np.nan)
    out = np.full_like(intensity, np.nan, dtype=float)
    n = len(intensity)
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        if hi == lo:
            continue
        dy, dx = intensity[hi] - intensity[lo], logp[hi] - logp[lo]
        if np.isfinite(dy) and np.isfinite(dx) and dx != 0:
            out[i] = dy / dx
    return out


# ----------------------------------------------------------------------
# Synthetic power sweep
# ----------------------------------------------------------------------
def make_synthetic_sweep(n_points=24):
    """Log-spaced sweep 100 nW .. 300 uW with Hill-saturating emission."""
    powers_w = np.logspace(np.log10(100e-9), np.log10(300e-6), n_points)
    truth = dict(Imax=5.0e4, n=1.05, K=40e-6, c=0.0)
    intensity = hill_saturation(powers_w, **truth)
    intensity *= RNG.normal(1.0, 0.04, size=n_points)      # 4 % multiplicative noise
    filenames = [f"PL_{format_power_token(p)}_Vf0.asc" for p in powers_w]
    order = RNG.permutation(n_points)                       # files arrive unsorted
    return [filenames[i] for i in order], intensity[order], truth


def main():
    filenames, intensity, truth = make_synthetic_sweep()

    # -- parse powers back out of the filenames, then sort by power -----
    powers_w = np.array([parse_power_from_name(f) for f in filenames])
    order = np.argsort(powers_w)
    powers_w, intensity = powers_w[order], intensity[order]
    files_sorted = [filenames[i] for i in order]
    print("first three files:", files_sorted[:3])

    # -- saturation fit --------------------------------------------------
    p0 = [intensity.max(), 1.0, np.median(powers_w), 0.0]
    lo = [0, 0.3, powers_w.min() / 10, -np.inf]
    hi = [np.inf, 3.0, powers_w.max() * 10, np.inf]
    popt, pcov = curve_fit(hill_saturation, powers_w, intensity,
                           p0=p0, bounds=(lo, hi), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    dIdlogP = derivative_dlogp_nan(intensity, powers_w)

    # -- plots ------------------------------------------------------------
    P_fine = np.logspace(np.log10(powers_w.min()), np.log10(powers_w.max()), 400)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].loglog(powers_w * 1e6, intensity, "o", ms=5, label="data")
    axes[0].loglog(P_fine * 1e6, hill_saturation(P_fine, *popt), "-", lw=1.5,
                   label=f"Hill fit: n={popt[1]:.2f}, K={popt[2]*1e6:.1f} uW")
    guide = intensity[2] * (P_fine / powers_w[2]) ** 1.0
    axes[0].loglog(P_fine * 1e6, guide, ":", color="0.5", label="slope 1 guide")
    axes[0].set(xlabel="power (uW)", ylabel="integrated PL (counts)",
                title="log-log power dependence")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogx(powers_w * 1e6, intensity / hill_saturation(powers_w, *popt),
                     "o-", ms=4)
    axes[1].axhline(1.0, color="0.6", ls="--", lw=0.8)
    axes[1].set(xlabel="power (uW)", ylabel="data / fit", title="fit residual ratio")

    axes[2].semilogx(powers_w * 1e6, dIdlogP, "s-", ms=4, color="C3")
    axes[2].axvline(popt[2] * 1e6, color="0.5", ls=":",
                    label=f"K = {popt[2]*1e6:.1f} uW")
    axes[2].set(xlabel="power (uW)", ylabel="dI/d(log10 P)",
                title="saturation onset via dI/dlogP")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("02_power_dependence_saturation.png", dpi=150)
    plt.show()

    print(f"truth : Imax={truth['Imax']:.3g}  n={truth['n']:.2f}  K={truth['K']*1e6:.1f} uW")
    print(f"fitted: Imax={popt[0]:.3g}+/-{perr[0]:.2g}  n={popt[1]:.2f}+/-{perr[1]:.2g}  "
          f"K={popt[2]*1e6:.1f}+/-{perr[2]*1e6:.2g} uW")


if __name__ == "__main__":
    main()
