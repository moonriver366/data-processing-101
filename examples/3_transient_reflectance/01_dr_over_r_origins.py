"""
3.01 — What makes a dR/R signal (it is not always population)
=============================================================

Transient reflectance measures

    dR/R0 = (R_pump_on - R_pump_off) / R_pump_off

and the temptation is to read every trace as "excited population vs
time". But the pump can modify the pump-off resonance in three
independent ways, each with its own dR/R fingerprint:

  * AMPLITUDE change (bleach / oscillator-strength reduction, or the
    opposite sign for an induced absorption): dR/R has ONE lobe, same
    shape as the resonance itself. This is the population-like term.
  * SPECTRAL SHIFT: dR/R looks like the FIRST derivative of R0 —
    a dispersive two-lobe shape with zero net area near resonance.
  * BROADENING: dR/R looks like the SECOND derivative — three lobes.

A shift or broadening produces a large dR/R with essentially ZERO
population change (e.g. heating, screening, local fields). Before
fitting "the dynamics", look at the transient SPECTRUM at a few delays
and decide which mechanism(s) you actually have: a single-wavelength
trace on a shifting resonance can even change sign with time while the
population decays monotonically. That is the whole point of measuring
maps instead of single-wavelength kinetics.

All data synthetic.

Run:  python 01_dr_over_r_origins.py
"""

import numpy as np
import matplotlib.pyplot as plt


def reflectance_dip(E, A=0.15, E0=1.70, G=0.012, base=0.35, slope=-0.15):
    """Pump-off reflectance: broadband background with a resonance dip."""
    return base + slope * (E - 1.70) - A * (G / 2) ** 2 / ((E - E0) ** 2 + (G / 2) ** 2)


def pumped(E, bleach=0.0, shift=0.0, broaden=0.0):
    """Pump-on spectrum: modify amplitude, center, width of the dip."""
    return reflectance_dip(E, A=0.15 * (1 - bleach), E0=1.70 + shift,
                           G=0.012 * (1 + broaden))


def main():
    E = np.linspace(1.65, 1.75, 900)
    R0 = reflectance_dip(E)

    cases = [
        ("bleach 20%",             dict(bleach=0.20), "C3"),
        ("shift +2 meV",           dict(shift=0.002), "C0"),
        ("broaden 25%",            dict(broaden=0.25), "C2"),
        ("all three together",     dict(bleach=0.20, shift=0.002, broaden=0.25), "k"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].plot(E, R0, "k-", lw=1.8, label="pump off R0")
    axes[0].plot(E, pumped(E, **cases[3][1]), "r--", lw=1.4, label="pump on")
    axes[0].set(xlabel="energy (eV)", ylabel="reflectance",
                title="the raw ingredients: two nearly\nidentical spectra")
    axes[0].legend(frameon=False, fontsize=8)

    for name, kw, color in cases:
        dr = (pumped(E, **kw) - R0) / R0
        axes[1].plot(E, dr * 100, color=color, lw=1.5, label=name)
    axes[1].axhline(0, color="0.6", lw=0.7)
    axes[1].set(xlabel="energy (eV)", ylabel="dR/R0 (%)",
                title="one lobe = amplitude; two lobes = shift;\nthree lobes = broadening")
    axes[1].legend(frameon=False, fontsize=8)

    # single-wavelength trap: probe fixed on the zero crossing
    delays = np.linspace(0, 20, 200)
    pop = np.exp(-delays / 5.0)                 # true population decay
    shift_t = 0.004 * np.exp(-delays / 2.0)     # shift relaxes faster
    E_probe = 1.6965                            # sits on a dR/R zero crossing
    iE = np.argmin(np.abs(E - E_probe))
    trace = [((pumped(E, bleach=0.2 * p, shift=s) - R0) / R0)[iE]
             for p, s in zip(pop, shift_t)]
    axes[2].plot(delays, np.array(trace) * 100, "C4-", lw=1.6,
                 label=f"dR/R at {E_probe:.4f} eV")
    axes[2].plot(delays, pop * np.max(np.abs(trace)) * 100, "0.6", ls="--",
                 lw=1.2, label="true population (scaled)")
    axes[2].axhline(0, color="0.6", lw=0.7)
    axes[2].set(xlabel="delay (ps)", ylabel="dR/R0 (%)",
                title="single-wavelength trap: sign flip\nwhile population decays monotonically")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s3_01_dr_over_r_origins.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
