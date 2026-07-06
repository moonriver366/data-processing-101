"""
Example 05 — Gate-dependent PL map: species tracking + NaN-aware smoothing
==========================================================================

Workflow demonstrated (as used for gated TMDC devices with Keithley
voltage sweeps + Andor spectra):

1. Simulate a gate sweep: one PL spectrum per gate voltage. Around the
   charge-neutrality point the neutral exciton X0 dominates; with
   electron/hole doping, oscillator strength transfers to the charged
   trion X+/X-, which also red-shifts with |Vg|.
2. Assemble the (gate, energy) map — the standard energy-vs-gate
   colormap. A few spectra are "dropped" (NaN rows) to mimic skipped
   measurement points, then repaired with NaN-aware Gaussian smoothing
   (normalized convolution).
3. Track both species vs gate with a windowed double-Gaussian fit and
   plot peak position and amplitude trajectories.

No real data is used anywhere — everything is synthesized in-script.

Run:  python 05_gate_dependent_pl_map.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=5)

EV_NM = 1239.841984  # nm <-> eV conversion constant


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def gaussian(E, A, mu, sigma):
    return A * np.exp(-0.5 * ((E - mu) / sigma) ** 2)


def double_gaussian(E, A1, mu1, s1, A2, mu2, s2):
    """Exciton (peak 1) + trion (peak 2)."""
    return gaussian(E, A1, mu1, s1) + gaussian(E, A2, mu2, s2)


# ----------------------------------------------------------------------
# Synthetic gate sweep
# ----------------------------------------------------------------------
def make_synthetic_gate_map():
    gates = np.linspace(-6.0, 6.0, 61)                 # back-gate voltage (V)
    energy = np.linspace(1.55, 1.80, 400)              # photon energy (eV)

    E_X0, E_T0 = 1.735, 1.705                          # exciton / trion at Vg=0
    maps = np.empty((gates.size, energy.size))
    for i, vg in enumerate(gates):
        doping = abs(vg) / 6.0                          # 0 at CNP, 1 at max gate
        A_x = 1500.0 * (1.0 - 0.85 * doping)            # X0 quenches with doping
        A_t = 1400.0 * doping                           # trion grows
        mu_t = E_T0 - 0.010 * doping                    # trion red-shifts
        spec = double_gaussian(energy, A_x, E_X0, 0.008, A_t, mu_t, 0.011)
        maps[i] = RNG.poisson(np.clip(spec + 30.0, 0, None))  # +dark level
    maps -= 30.0                                        # dark subtraction

    # drop a few gate points (sweep aborted / skipped) -> NaN rows
    for i in RNG.choice(gates.size, size=4, replace=False):
        maps[i] = np.nan
    return gates, energy, maps


# ----------------------------------------------------------------------
# NaN-aware Gaussian smoothing (normalized convolution)
# ----------------------------------------------------------------------
def nan_gaussian_smooth_2d(z, sigma_gate=1.0, sigma_energy=1.5):
    """Gaussian-smooth a map containing NaNs without letting them spread.

    Smooth the zero-filled data and the finite-mask separately, then
    divide: each output pixel is a weighted mean over its *valid*
    neighbors only. This both denoises and fills small NaN gaps.
    """
    valid = np.isfinite(z).astype(float)
    filled = np.where(np.isfinite(z), z, 0.0)
    num = gaussian_filter(filled, sigma=(sigma_gate, sigma_energy), mode="nearest")
    den = gaussian_filter(valid, sigma=(sigma_gate, sigma_energy), mode="nearest")
    out = np.full_like(z, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 1e-12)
    return out


# ----------------------------------------------------------------------
# Species tracking
# ----------------------------------------------------------------------
def track_species(gates, energy, maps):
    """Windowed double-Gaussian fit per gate voltage.

    Fixed energy windows keep each Gaussian assigned to its species;
    rows that fail the fit return NaN and simply leave a gap.
    """
    rows = []
    for i in range(gates.size):
        spec = maps[i]
        if not np.all(np.isfinite(spec)):
            rows.append((np.nan,) * 4)
            continue
        p0 = [spec.max(), 1.735, 0.008, spec.max() / 2, 1.700, 0.011]
        lo = [0, 1.720, 0.003, 0, 1.670, 0.003]
        hi = [np.inf, 1.755, 0.020, np.inf, 1.715, 0.025]
        try:
            popt, _ = curve_fit(double_gaussian, energy, spec, p0=p0,
                                bounds=(lo, hi), maxfev=20000)
            rows.append((popt[1], popt[0], popt[4], popt[3]))  # muX, AX, muT, AT
        except RuntimeError:
            rows.append((np.nan,) * 4)
    return np.array(rows)  # columns: mu_X0, A_X0, mu_T, A_T


def main():
    gates, energy, maps = make_synthetic_gate_map()
    smoothed = nan_gaussian_smooth_2d(maps)
    tracks = track_species(gates, energy, smoothed)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    extent = [energy[0], energy[-1], gates[0], gates[-1]]

    im0 = axes[0, 0].imshow(maps, extent=extent, aspect="auto", origin="lower",
                            cmap="magma")
    axes[0, 0].set(xlabel="energy (eV)", ylabel="gate voltage (V)",
                   title="raw map (NaN rows = skipped points)")
    fig.colorbar(im0, ax=axes[0, 0], label="PL (counts)")

    im1 = axes[0, 1].imshow(smoothed, extent=extent, aspect="auto", origin="lower",
                            cmap="magma")
    axes[0, 1].plot(tracks[:, 0], gates, "w--", lw=1, label="X$^0$ center")
    axes[0, 1].plot(tracks[:, 2], gates, "c--", lw=1, label="trion center")
    axes[0, 1].set(xlabel="energy (eV)", title="NaN-aware smoothed + tracked peaks")
    axes[0, 1].legend(frameon=False, fontsize=8, labelcolor="w")
    fig.colorbar(im1, ax=axes[0, 1], label="PL (counts)")

    ok = np.isfinite(tracks[:, 0])
    axes[1, 0].plot(gates[ok], tracks[ok, 0], "o-", ms=3, label="X$^0$")
    axes[1, 0].plot(gates[ok], tracks[ok, 2], "s-", ms=3, label="trion")
    axes[1, 0].set(xlabel="gate voltage (V)", ylabel="peak energy (eV)",
                   title="peak position vs gate")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(gates[ok], tracks[ok, 1], "o-", ms=3, label="X$^0$")
    axes[1, 1].plot(gates[ok], tracks[ok, 3], "s-", ms=3, label="trion")
    axes[1, 1].axvline(0, color="0.6", ls=":", label="charge neutrality")
    axes[1, 1].set(xlabel="gate voltage (V)", ylabel="fit amplitude (counts)",
                   title="oscillator-strength transfer X$^0 \\to$ trion")
    axes[1, 1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig("05_gate_dependent_pl_map.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
