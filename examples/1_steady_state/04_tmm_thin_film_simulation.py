"""
1.04 — Why RC lineshapes look Fano: a minimal transfer-matrix simulation
========================================================================

The resonance in the layer is a plain Lorentz oscillator — perfectly
symmetric. Yet the measured RC of a thin resonant layer buried in a
dielectric stack (the classic case: an hBN-encapsulated monolayer on
SiO2/Si) is usually asymmetric, and its sign and shape change with the
oxide thickness. Nothing exotic is happening: the resonant reflection
interferes with the broadband reflection from the rest of the stack —
a Fano situation by construction. The transfer-matrix method (TMM)
makes this quantitative in ~30 lines.

Model stack (normal incidence):
    air / top dielectric 10 nm / resonant layer 0.6 nm / bottom
    dielectric 10 nm / SiO2 (variable) / Si substrate

The resonant layer gets a Lorentz permittivity
    eps(E) = eps_b + f / (E0^2 - E^2 - i*gamma*E)

RC is computed against the identical stack WITHOUT the resonant layer,
which is exactly what an off-sample reference measures.

All parameters generic/synthetic.

Run:  python 04_tmm_thin_film_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt

HC_EV_NM = 1239.841984


# ----------------------------------------------------------------------
# TMM core: characteristic matrix per layer, normal incidence
# ----------------------------------------------------------------------
def reflectance_stack(E_eV, layers, n_in=1.0, n_sub=3.88 + 0.02j):
    """R(E) for a stack of (refractive_index, thickness_nm) layers.

    `refractive_index` may be a scalar or an array over E (complex).
    Standard optics-textbook formulation:
        M_layer = [[cos d, i sin d / n], [i n sin d, cos d]],
        d = 2 pi n t / lambda
        r = (n_in*(m11 + m12 n_sub) - (m21 + m22 n_sub))
          / (n_in*(m11 + m12 n_sub) + (m21 + m22 n_sub))
    """
    lam_nm = HC_EV_NM / np.asarray(E_eV)
    m11 = np.ones_like(lam_nm, dtype=complex)
    m12 = np.zeros_like(m11)
    m21 = np.zeros_like(m11)
    m22 = np.ones_like(m11)
    for n, t_nm in layers:
        n = np.asarray(n, dtype=complex) * np.ones_like(m11)
        d = 2.0 * np.pi * n * t_nm / lam_nm
        c, s = np.cos(d), np.sin(d)
        a11, a12, a21, a22 = c, 1j * s / n, 1j * n * s, c
        m11, m12, m21, m22 = (m11 * a11 + m12 * a21, m11 * a12 + m12 * a22,
                              m21 * a11 + m22 * a21, m21 * a12 + m22 * a22)
    num = n_in * (m11 + m12 * n_sub) - (m21 + m22 * n_sub)
    den = n_in * (m11 + m12 * n_sub) + (m21 + m22 * n_sub)
    return np.abs(num / den) ** 2


def lorentz_index(E_eV, eps_b=6.0, f=0.8, E0=1.70, gamma=0.010):
    """Lorentz oscillator permittivity -> complex refractive index."""
    eps = eps_b + f / (E0 ** 2 - E_eV ** 2 - 1j * gamma * E_eV)
    return np.sqrt(eps.astype(complex))


def rc_of_stack(E, t_sio2_nm, f=0.8):
    n_res = lorentz_index(E, f=f)
    with_layer = [(2.1, 10.0), (n_res, 0.6), (2.1, 10.0), (1.46, t_sio2_nm)]
    without = [(2.1, 10.0), (2.1, 10.0), (1.46, t_sio2_nm)]
    R1 = reflectance_stack(E, with_layer)
    R0 = reflectance_stack(E, without)
    return (R1 - R0) / R0


def main():
    E = np.linspace(1.55, 1.85, 900)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # the oscillator itself: symmetric Lorentzian absorption
    n_res = lorentz_index(E)
    axes[0].plot(E, n_res.imag, lw=1.5, color="C3")
    axes[0].set(xlabel="energy (eV)", ylabel="Im(n) of resonant layer",
                title="the input: a symmetric Lorentz oscillator")

    # ...but the measured RC depends on the rest of the stack
    for t in (90.0, 285.0, 200.0):
        axes[1].plot(E, rc_of_stack(E, t) * 100, lw=1.4, label=f"SiO2 = {t:.0f} nm")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].axvline(1.70, color="0.6", ls=":", lw=1, label="oscillator E0")
    axes[1].set(xlabel="energy (eV)", ylabel="RC (%)",
                title="the output: sign, size and asymmetry\nall set by interference with the stack")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s1_04_tmm_simulation.png", dpi=150)
    plt.show()

    print("Takeaways:")
    print(" * an asymmetric / dispersive RC lineshape does NOT mean the")
    print("   underlying resonance is exotic -- check the stack first;")
    print(" * fitting RC with a Fano (or a TMM model directly) recovers E0;")
    print("   reading the extremum off the plot does not.")


if __name__ == "__main__":
    main()
