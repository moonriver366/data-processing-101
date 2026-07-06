"""
Example 06 — Exciton-polariton dispersion: coupled-oscillator fit
=================================================================

Workflow demonstrated (as used for angle-resolved reflectance of a
TMDC microcavity):

1. Simulate an angle-resolved white-light reflectance map R(theta, E):
   two polariton branches (Lorentzian dips) whose energies follow the
   2x2 coupled-oscillator model,

       E_LP/UP(theta) = (Ec(theta) + EX)/2 -/+ sqrt(d^2 + (2g)^2)/2,
       d = Ec(theta) - EX,   Ec(theta) = Ec0 / sqrt(1 - sin^2(theta)/n_eff^2)

2. Extract the branch minima per angle (ridge finding), as done on
   the real E(k) camera images.
3. Fit the extracted branch positions with the coupled-oscillator
   model to recover the Rabi splitting 2g, exciton energy EX, and
   cavity detuning — and compute the Hopfield coefficients that give
   the photon/exciton content of each branch.

No real data is used anywhere — everything is synthesized in-script.

Run:  python 06_polariton_dispersion_fit.py
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=12)


# ----------------------------------------------------------------------
# Coupled-oscillator model
# ----------------------------------------------------------------------
def cavity_dispersion(theta_deg, Ec0, n_eff):
    """Planar-cavity photon dispersion vs external angle."""
    s = np.sin(np.deg2rad(theta_deg)) / n_eff
    return Ec0 / np.sqrt(1.0 - s * s)


def polariton_bands(Ec, EX, twog):
    """LP and UP energies from diagonalizing [[Ec, g], [g, EX]]."""
    d = Ec - EX
    Om = np.sqrt(d * d + twog * twog)
    Emid = 0.5 * (Ec + EX)
    return Emid - 0.5 * Om, Emid + 0.5 * Om


def hopfield_fractions(Ec, EX, twog):
    """|C|^2 (photon) and |X|^2 (exciton) content of the LP branch."""
    d = Ec - EX
    Om = np.maximum(np.sqrt(d * d + twog * twog), 1e-30)
    C_LP2 = 0.5 * (1.0 - d / Om)
    X_LP2 = 0.5 * (1.0 + d / Om)
    return C_LP2, X_LP2


# ----------------------------------------------------------------------
# Synthetic angle-resolved reflectance map
# ----------------------------------------------------------------------
def make_synthetic_dispersion_map():
    theta = np.linspace(-35.0, 35.0, 141)             # external angle (deg)
    energy = np.linspace(1.60, 1.85, 500)             # photon energy (eV)
    truth = dict(EX=1.735, twog=0.028, Ec0=1.722, n_eff=1.8)

    Ec = cavity_dispersion(theta, truth["Ec0"], truth["n_eff"])
    Elp, Eup = polariton_bands(Ec, truth["EX"], truth["twog"])
    C_LP2, X_LP2 = hopfield_fractions(Ec, truth["EX"], truth["twog"])

    # Lorentzian dips; branch visibility follows its photon fraction
    R = np.ones((theta.size, energy.size))
    g_lp = 0.004 + 0.004 * X_LP2                       # exciton content broadens LP
    g_up = 0.004 + 0.004 * C_LP2
    for i in range(theta.size):
        R[i] -= 0.55 * C_LP2[i] * g_lp[i] ** 2 / ((energy - Elp[i]) ** 2 + g_lp[i] ** 2)
        R[i] -= 0.55 * (1 - C_LP2[i]) * g_up[i] ** 2 / ((energy - Eup[i]) ** 2 + g_up[i] ** 2)
    R += RNG.normal(0, 0.01, size=R.shape)
    return theta, energy, R, truth


# ----------------------------------------------------------------------
# Ridge extraction + dispersion fit
# ----------------------------------------------------------------------
def extract_branch_minima(theta, energy, R, split_eV=1.735):
    """Per angle, find the reflectance minimum below and above split_eV.

    Mirrors the per-row ridge finding used on E(k) camera images. A dip
    shallower than the noise floor is rejected (NaN).
    """
    lp, up = np.full(theta.size, np.nan), np.full(theta.size, np.nan)
    lo_m, hi_m = energy < split_eV, energy >= split_eV
    for i in range(theta.size):
        row = R[i]
        for mask, out in ((lo_m, lp), (hi_m, up)):
            j = np.argmin(row[mask])
            depth = np.median(row[mask]) - row[mask][j]
            if depth > 0.05:                           # visibility gate
                out[i] = energy[mask][j]
    return lp, up


def fit_coupled_oscillator(theta, lp, up):
    """Simultaneous least-squares fit of both branches."""
    def residuals(p):
        EX, twog, Ec0, n_eff = p
        Ec = cavity_dispersion(theta, Ec0, n_eff)
        Elp, Eup = polariton_bands(Ec, EX, twog)
        r = np.concatenate([(Elp - lp)[np.isfinite(lp)],
                            (Eup - up)[np.isfinite(up)]])
        return r

    res = least_squares(residuals, x0=[1.73, 0.02, 1.72, 2.0],
                        bounds=([1.6, 0.005, 1.6, 1.2], [1.85, 0.08, 1.85, 4.0]))
    return res.x


def main():
    theta, energy, R, truth = make_synthetic_dispersion_map()
    lp, up = extract_branch_minima(theta, energy, R)
    EX, twog, Ec0, n_eff = fit_coupled_oscillator(theta, lp, up)

    th_f = np.linspace(theta[0], theta[-1], 400)
    Ec_f = cavity_dispersion(th_f, Ec0, n_eff)
    Elp_f, Eup_f = polariton_bands(Ec_f, EX, twog)
    C_LP2, X_LP2 = hopfield_fractions(Ec_f, EX, twog)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    im = axes[0].imshow(R.T, extent=[theta[0], theta[-1], energy[0], energy[-1]],
                        origin="lower", aspect="auto", cmap="gray")
    axes[0].plot(theta, lp, ".", ms=3, color="C3", label="LP minima")
    axes[0].plot(theta, up, ".", ms=3, color="C0", label="UP minima")
    axes[0].set(xlabel="angle (deg)", ylabel="energy (eV)",
                title="synthetic R($\\theta$, E) + extracted ridges")
    axes[0].legend(frameon=False, fontsize=8)
    fig.colorbar(im, ax=axes[0], label="reflectance")

    axes[1].plot(theta, lp, ".", ms=4, color="C3")
    axes[1].plot(theta, up, ".", ms=4, color="C0")
    axes[1].plot(th_f, Elp_f, "-", color="C3", lw=1.4, label="LP fit")
    axes[1].plot(th_f, Eup_f, "-", color="C0", lw=1.4, label="UP fit")
    axes[1].plot(th_f, Ec_f, "--", color="0.5", lw=1, label="cavity $E_c(\\theta)$")
    axes[1].axhline(EX, color="0.5", ls=":", lw=1, label=f"$E_X$ = {EX:.3f} eV")
    axes[1].set(xlabel="angle (deg)", ylabel="energy (eV)",
                title=f"coupled-oscillator fit: $2g$ = {twog*1000:.1f} meV")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(th_f, C_LP2, lw=1.5, color="C3", label="LP photon fraction $|C|^2$")
    axes[2].plot(th_f, X_LP2, lw=1.5, color="C0", label="LP exciton fraction $|X|^2$")
    axes[2].axhline(0.5, color="0.6", ls=":", lw=0.8)
    axes[2].set(xlabel="angle (deg)", ylabel="Hopfield fraction",
                title="LP branch composition", ylim=(0, 1))
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("06_polariton_dispersion_fit.png", dpi=150)
    plt.show()

    print(f"truth : EX={truth['EX']:.3f}  2g={truth['twog']*1e3:.1f} meV  "
          f"Ec0={truth['Ec0']:.3f}  n_eff={truth['n_eff']:.2f}")
    print(f"fitted: EX={EX:.3f}  2g={twog*1e3:.1f} meV  Ec0={Ec0:.3f}  n_eff={n_eff:.2f}")


if __name__ == "__main__":
    main()
