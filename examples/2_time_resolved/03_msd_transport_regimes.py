"""
2.03 — MSD analysis: broadening is not always transport
=======================================================

The standard move: fit a Gaussian to the profile at each time, plot

    MSD(t) = sigma(t)^2 - sigma(0)^2

and read the physics off the shape. But several very different
mechanisms broaden a profile, and only some of them are motion:

  * DIFFUSION        MSD = 2 D t          (linear, slope = 2D per axis)
  * BALLISTIC        MSD = v^2 t^2        (superlinear at early t)
  * ANNIHILATION     bimolecular decay ~ gamma n^2 eats the CENTER of
    the profile first (density is highest there), so the profile
    flattens and the fitted sigma GROWS — with zero actual transport.
  * EMISSION SATURATION  detected signal ~ n/(1 + n/n_sat) clips the
    center at early times; apparent sigma starts large and SHRINKS as
    the density drops — "negative diffusion" that isn't.

This script simulates all four on the same 1D grid (explicit Euler)
and compares the apparent MSD curves. My rule of thumb: before calling
it diffusion, check that (a) MSD is linear over a decade of time, and
(b) the extracted D does NOT change with pump power. Power-dependent
"D" is the classic annihilation fingerprint.

All data synthetic.

Run:  python 03_msd_transport_regimes.py
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1D kinetic simulation:  dn/dt = D n'' - n/tau - gamma n^2
# ----------------------------------------------------------------------
def simulate(D=0.0, tau=5.0, gamma=0.0, v=0.0, n0=1.0,
             sigma0=0.8, L=12.0, nx=241, t_end=10.0, nt=2001):
    x = np.linspace(-L / 2, L / 2, nx)
    dx = x[1] - x[0]
    dt = t_end / (nt - 1)
    if D > 0 and dt > 0.4 * dx * dx / D:
        raise ValueError("explicit Euler unstable: reduce dt or D")
    times = np.linspace(0, t_end, nt)
    n = n0 * np.exp(-0.5 * x ** 2 / sigma0 ** 2)
    if v > 0:                      # ballistic: split into +v and -v movers
        nR, nL = 0.5 * n.copy(), 0.5 * n.copy()
    snaps, snap_t = [], np.linspace(0, t_end, 41)
    k = 0
    for i, t in enumerate(times):
        if v > 0:
            n = nR + nL
        if k < snap_t.size and t >= snap_t[k] - 1e-9:
            snaps.append(n.copy())
            k += 1
        if v > 0:                  # upwind advection for the two streams
            nR = nR - v * dt / dx * (nR - np.roll(nR, 1)) - dt * nR / tau
            nL = nL + v * dt / dx * (np.roll(nL, -1) - nL) - dt * nL / tau
            nR[0] = nL[-1] = 0.0
        else:
            lap = (np.roll(n, 1) - 2 * n + np.roll(n, -1)) / dx ** 2
            lap[0] = lap[-1] = 0.0
            n = n + dt * (D * lap - n / tau - gamma * n * n)
            n = np.clip(n, 0, None)
    return x, snap_t, np.array(snaps)


def gaussian(x, A, x0, s, c):
    return A * np.exp(-0.5 * ((x - x0) / s) ** 2) + c


def sigma_vs_time(x, snaps, saturate=None):
    out = []
    for prof in snaps:
        sig = prof / (1.0 + prof / saturate) if saturate else prof
        if sig.max() <= 0:
            out.append(np.nan)
            continue
        try:
            popt, _ = curve_fit(gaussian, x, sig,
                                p0=[sig.max(), 0, 1.0, 0], maxfev=10000)
            out.append(abs(popt[2]))
        except RuntimeError:
            out.append(np.nan)
    return np.array(out)


def main():
    n0_high = 4.0
    cases = {
        "diffusion (D=0.10)":        dict(D=0.10),
        "ballistic (v=0.25)":        dict(v=0.25),
        "annihilation only (D=0)":   dict(gamma=0.8, n0=n0_high),
        "saturation only (D=0)":     dict(),          # applied at detection
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    results = {}
    for name, kw in cases.items():
        x, ts, snaps = simulate(**kw)
        sat = 0.5 if "saturation" in name else None
        results[name] = (ts, sigma_vs_time(x, snaps, saturate=sat))

    # profiles for the annihilation case: center eaten first
    x, ts, snaps = simulate(gamma=0.8, n0=n0_high)
    for i in (0, 8, 20, 40):
        p = snaps[i]
        axes[0].plot(x, p / p.max(), lw=1.3, label=f"t = {ts[i]:.1f} ns")
    axes[0].set(xlabel="x (um)", ylabel="normalized profile",
                title="annihilation flattens the center:\nlooks like spreading, isn't",
                xlim=(-4, 4))
    axes[0].legend(frameon=False, fontsize=8)

    for name, (ts, sig) in results.items():
        axes[1].plot(ts, sig ** 2 - sig[0] ** 2, lw=1.6, label=name)
    axes[1].set(xlabel="time (ns)", ylabel="MSD = sigma² - sigma0² (um²)",
                title="apparent MSD: same observable,\nfour different mechanisms")
    axes[1].legend(frameon=False, fontsize=8)

    # log-log: slopes separate diffusive (1) from ballistic (2)
    for name, (ts, sig) in results.items():
        msd = sig ** 2 - sig[0] ** 2
        m = (ts > 0.2) & (msd > 1e-4)
        axes[2].loglog(ts[m], msd[m], lw=1.6, label=name)
    axes[2].loglog([0.3, 3], [0.06 * (t / 0.3) for t in (0.3, 3)], "k:", lw=1)
    axes[2].text(1.1, 0.13, "slope 1", fontsize=8)
    axes[2].loglog([0.3, 3], [0.006 * (t / 0.3) ** 2 for t in (0.3, 3)], "k--", lw=1)
    axes[2].text(1.1, 0.011, "slope 2", fontsize=8)
    axes[2].set(xlabel="time (ns)", ylabel="MSD (um²)",
                title="log-log: check the exponent AND\nthe power dependence before claiming D")
    axes[2].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig("s2_03_msd_regimes.png", dpi=150)
    plt.show()

    ts, sig = results["diffusion (D=0.10)"]
    slope = np.polyfit(ts[5:], sig[5:] ** 2, 1)[0]
    print(f"diffusion case: fitted MSD slope = {slope:.3f} um^2/ns -> D = "
          f"{slope/2:.3f} (truth 0.100)")


if __name__ == "__main__":
    main()
