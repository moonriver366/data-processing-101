"""
1.05 — Steady-state PL: what the counts mean, what the profile means
====================================================================

Two things I remind myself of before interpreting any steady-state PL:

1) PL IS TIME-INTEGRATED. With a pulsed laser at repetition rate f_rep,
       counts ∝ f_rep x (photons emitted per pulse).
   So at the same *average* power, halving f_rep doubles the energy per
   pulse — and if anything nonlinear happens at high instantaneous
   density (saturation, annihilation), the "same power" spectrum
   changes. Also intensity and lifetime are entangled: for a single
   exponential, integrated PL ∝ n0 * tau. A sample can look "dimmer"
   purely because a new nonradiative channel shortened tau.

2) THE SPATIAL PROFILE IS NOT THE LASER. Whatever is excited moves
   before it emits. For a Gaussian excitation spot (variance s_L^2, per
   axis) and diffusive motion during an exponential lifetime, each
   emission moment t contributes a Gaussian of variance s_L^2 + 2Dt;
   time-integrating with weight e^{-t/tau} gives a profile that is
   slightly non-Gaussian but whose *variance* is exactly

       s_PL^2 = s_L^2 + 2 D tau        (per axis)

   so a Gaussian fit of laser and PL profiles gives the diffusion
   length directly:  L_D = sqrt(D tau) = sqrt((s_PL^2 - s_L^2)/2).
   This needs the laser profile measured through the SAME optics —
   which is the same measurement as the calibration spot in 1.01.

All data synthetic.

Run:  python 05_pl_and_spatial_profile.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=5)


def gaussian(x, A, x0, sigma, c):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


# ----------------------------------------------------------------------
# Time-integrated emission profile for diffusion + exponential decay
# ----------------------------------------------------------------------
def steady_state_profile(x, sigma_L, D, tau, n_t=400):
    """Numerical time integral of the spreading Gaussian.

    I(x) ∝ ∫ dt e^{-t/tau} / sqrt(s_L^2 + 2Dt) * exp(-x^2 / 2(s_L^2+2Dt))
    """
    t = np.linspace(0, 8 * tau, n_t)
    var = sigma_L ** 2 + 2 * D * t
    w = np.exp(-t / tau)
    prof = np.trapz(w[None, :] / np.sqrt(var[None, :])
                    * np.exp(-0.5 * x[:, None] ** 2 / var[None, :]), t, axis=1)
    return prof / prof.max()


def fit_sigma(x, y):
    p0 = [y.max(), 0.0, 0.5, 0.0]
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    return abs(popt[2])


def moment_sigma(x, y):
    """sqrt of the second central moment.

    The variance sum rule s_PL^2 = s_L^2 + 2 D tau holds for the true
    VARIANCE. A Gaussian FIT is core-weighted and underestimates sigma
    when the profile has heavy wings (which the time-integrated profile
    does), so for quantitative D the moment is the right estimator.
    Two traps, both x^2-amplified in the wings: (1) subtract the
    baseline, estimated from signal-free wings; (2) do NOT clip
    negative noise — clipping rectifies the wing noise into a fake
    positive pedestal that inflates the moment. Leave the noise
    zero-mean and it averages out.
    """
    yy = y - np.median(y[np.abs(x) > 0.8 * x.max()])
    mu = np.sum(x * yy) / np.sum(yy)
    return np.sqrt(np.sum((x - mu) ** 2 * yy) / np.sum(yy))


def main():
    # ------- part 1: intensity vs lifetime entanglement -------------------
    # same "brightness" question, three synthetic scenarios
    t = np.linspace(0, 20, 800)                        # ns
    scenarios = [("reference",            1.0, 4.0),
                 ("half the lifetime",    1.0, 2.0),
                 ("half the generation",  0.5, 4.0)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for name, n0, tau in scenarios:
        decay = n0 * np.exp(-t / tau)
        axes[0].plot(t, decay, lw=1.4,
                     label=f"{name}: integrated PL = {n0 * tau:.1f}")
    axes[0].set(xlabel="time (ns)", ylabel="population",
                title="steady-state PL only sees the AREA:\nn0*tau — dimmer ≠ fewer created")
    axes[0].legend(frameon=False, fontsize=8)

    # ------- part 2: spatial profile and diffusion length -----------------
    x = np.linspace(-4, 4, 401)                        # um
    sigma_L, D, tau = 0.35, 0.05, 4.0                  # um, um^2/ns, ns
    laser = np.exp(-0.5 * x ** 2 / sigma_L ** 2)
    pl = steady_state_profile(x, sigma_L, D, tau)
    laser_n = laser + RNG.normal(0, 0.01, x.size)
    pl_n = pl + RNG.normal(0, 0.01, x.size)

    s_L = fit_sigma(x, laser_n)
    s_PL = fit_sigma(x, pl_n)
    L_D = np.sqrt(max(s_PL ** 2 - s_L ** 2, 0) / 2)
    # moment-based estimate: obeys the variance sum rule exactly
    m_L = moment_sigma(x, laser_n)
    m_PL = moment_sigma(x, pl_n)
    L_D_mom = np.sqrt(max(m_PL ** 2 - m_L ** 2, 0) / 2)

    axes[1].plot(x, laser_n, ".", ms=2, color="0.7")
    axes[1].plot(x, gaussian(x, 1, 0, s_L, 0), "-", color="C0",
                 label=f"laser: sigma = {s_L:.3f} um")
    axes[1].plot(x, pl_n, ".", ms=2, color="#f2c0c0")
    axes[1].plot(x, gaussian(x, 1, 0, s_PL, 0), "-", color="C3",
                 label=f"PL: sigma = {s_PL:.3f} um")
    axes[1].set(xlabel="position (um)", ylabel="normalized intensity",
                title=f"PL is broader than the laser:\nL_D = sqrt((s_PL²-s_L²)/2) = {L_D:.3f} um")
    axes[1].legend(frameon=False, fontsize=8)
    print(f"true sqrt(D*tau)             = {np.sqrt(D * tau):.3f} um")
    print(f"L_D from Gaussian fits       = {L_D:.3f} um  <- core-weighted, biased low")
    print(f"L_D from second moments      = {L_D_mom:.3f} um  <- obeys the variance sum rule")

    # log scale: the time-integrated profile has slightly heavier wings
    axes[2].semilogy(x, np.clip(pl, 1e-4, None), color="C3", lw=1.4,
                     label="PL (time-integrated)")
    axes[2].semilogy(x, gaussian(x, 1, 0, s_PL, 0) + 1e-12, "k--", lw=1,
                     label="Gaussian with fitted sigma")
    axes[2].semilogy(x, laser, color="C0", lw=1.4, label="laser")
    axes[2].set(xlabel="position (um)", ylabel="intensity (log)", ylim=(1e-4, 2),
                title="log axis: long-lived tails make the\nwings slightly super-Gaussian")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s1_05_pl_spatial_profile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
