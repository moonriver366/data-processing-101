"""
2.02 — Getting a time-resolved SPATIAL profile: point scan vs array
===================================================================

Two ways to see how an excitation spot evolves in space and time:

  A) POINT SCAN (single-pixel detector, e.g. PicoHarp + APD): park the
     detection spot, record a full decay histogram, move x/y stages,
     repeat over a grid. Pros: full time resolution everywhere.
     Cons: slow — N^2 acquisitions — and the sample must not drift.

  B) DETECTOR ARRAY (SPAD array): every pixel records simultaneously.
     A sparse array (here 23 pixels, like the ones we use) doesn't
     image the spot — it SAMPLES it, and a 2D Gaussian fit turns those
     samples into sigma_x, sigma_y per time bin. The single number I
     carry around is
         sigma_eq = sqrt((sigma_x^2 + sigma_y^2) / 2)

Either way the deliverable is the same: sigma(t) — the input for the
MSD analysis in 2.03.

All data synthetic: a spot that decays and diffuses, sampled both ways.

Run:  python 02_spatial_profiles_scan_vs_array.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=22)

SIGMA0, D, TAU = 0.8, 0.06, 5.0        # um, um^2/ns, ns


def spot(x, y, t):
    """Ground truth: decaying, diffusing 2D Gaussian."""
    var = SIGMA0 ** 2 + 2 * D * t
    return np.exp(-t / TAU) * np.exp(-0.5 * (x ** 2 + y ** 2) / var) / var


def gauss2d(coords, A, x0, y0, sx, sy, offset):
    x, y = coords
    return (A * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2)
                         + ((y - y0) ** 2) / (2 * sy ** 2))) + offset)


def fit_gauss2d(x, y, z):
    """The lab-standard 2D Gaussian fit; returns sigma_eq."""
    p0 = [max(z.max() - z.min(), 1e-9), float(np.average(x, weights=z - z.min() + 1e-12)),
          float(np.average(y, weights=z - z.min() + 1e-12)), 1.0, 1.0, float(z.min())]
    lo = [0, x.min() - 1, y.min() - 1, 0.05, 0.05, -np.inf]
    hi = [np.inf, x.max() + 1, y.max() + 1, 20, 20, np.inf]
    popt, _ = curve_fit(gauss2d, (x, y), z, p0=p0, bounds=(lo, hi), maxfev=20000)
    sigma_eq = np.sqrt((popt[3] ** 2 + popt[4] ** 2) / 2)
    return popt, sigma_eq


def main():
    times = np.array([0.5, 2.0, 5.0, 10.0])

    # ---- A) point scan: 15x15 stage grid --------------------------------
    g = np.linspace(-3, 3, 15)
    XX, YY = np.meshgrid(g, g)
    xs, ys = XX.ravel(), YY.ravel()

    # ---- B) sparse array: 23 pixels on a hex-ish layout ------------------
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    xa = np.concatenate([[0], 1.4 * np.cos(ang), 2.8 * np.cos(ang + 0.26),
                         2.0 * np.cos(ang + 0.79)[:4]])
    ya = np.concatenate([[0], 1.4 * np.sin(ang), 2.8 * np.sin(ang + 0.26),
                         2.0 * np.sin(ang + 0.79)[:4]])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    sig_scan, sig_arr = [], []
    for t in times:
        z_scan = spot(xs, ys, t) * 5000
        z_scan = RNG.poisson(np.clip(z_scan, 0, None)).astype(float)
        _, s1 = fit_gauss2d(xs, ys, z_scan)
        sig_scan.append(s1)

        z_arr = spot(xa, ya, t) * 5000
        z_arr = RNG.poisson(np.clip(z_arr, 0, None)).astype(float)
        _, s2 = fit_gauss2d(xa, ya, z_arr)
        sig_arr.append(s2)

    # show the two sampling geometries at one time
    t_show = 2.0
    z = spot(XX, YY, t_show)
    axes[0].pcolormesh(XX, YY, z, cmap="magma", shading="auto")
    axes[0].plot(xs, ys, "+", ms=4, color="w", alpha=0.6)
    axes[0].set(title="A) point scan: stage grid\n(one decay histogram per +)",
                xlabel="x (um)", ylabel="y (um)", aspect="equal")

    axes[1].pcolormesh(XX, YY, z, cmap="magma", shading="auto")
    axes[1].plot(xa, ya, "o", ms=7, mfc="none", mec="w")
    axes[1].set(title="B) sparse array: 23 pixels sample\nthe spot simultaneously",
                xlabel="x (um)", aspect="equal")

    t_fine = np.linspace(0.3, 11, 100)
    axes[2].plot(t_fine, np.sqrt(SIGMA0 ** 2 + 2 * D * t_fine), "-", color="0.7",
                 lw=3, label="truth sigma(t)")
    axes[2].plot(times, sig_scan, "s", ms=7, color="C0", label="point scan fit")
    axes[2].plot(times, sig_arr, "o", ms=7, color="C3", label="array fit (23 px)")
    axes[2].set(xlabel="time (ns)", ylabel="sigma (um)",
                title="both routes give the same sigma(t)\n-> input for MSD analysis (2.03)")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s2_02_scan_vs_array.png", dpi=150)
    plt.show()

    print("time(ns)  truth   scan    array")
    for t, s1, s2 in zip(times, sig_scan, sig_arr):
        print(f"{t:7.1f} {np.sqrt(SIGMA0**2 + 2*D*t):7.3f} {s1:7.3f} {s2:7.3f}")


if __name__ == "__main__":
    main()
