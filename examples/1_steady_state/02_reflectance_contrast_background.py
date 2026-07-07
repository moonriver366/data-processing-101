"""
1.02 — Reflectance contrast and the background problem
=======================================================

Reflectance contrast is the workhorse linear measurement:

    RC(E) = (R_sample(E) - R_reference(E)) / R_reference(E)

It removes the lamp spectrum, the spectrometer response, and every
other multiplicative factor — IF you have a reference. Two situations:

  A) There is an off-sample / bare-substrate region: take R_reference
     there, done. This is the clean case.

  B) There is no feature-free spot anywhere (e.g. a planar cavity whose
     "background" is itself thickness-dependent). Then the reference
     must be *estimated*. My current take — and this is an open
     question, better ideas welcome — is to mask the resonance and
     interpolate the background through the feature-free wavelengths
     (linear or low-order polynomial). This script shows exactly how
     much the answer depends on that choice, which is the honest reason
     to always report how the background was constructed.

All data synthetic.

Run:  python 02_reflectance_contrast_background.py
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=2)


# ----------------------------------------------------------------------
# Synthetic spectra
# ----------------------------------------------------------------------
def lamp_and_system(E):
    """Broad lamp spectrum x system response: smooth but not flat."""
    return (1.0 + 0.6 * np.exp(-0.5 * ((E - 1.62) / 0.18) ** 2)) * (2.2 - 0.5 * E)


def true_feature(E, E0=1.70, gamma=0.008, depth=0.12):
    """The physics we are after: a resonance dip in reflectance."""
    return 1.0 - depth * gamma ** 2 / ((E - E0) ** 2 + gamma ** 2)


def make_spectra():
    E = np.linspace(1.55, 1.85, 700)
    base = lamp_and_system(E)
    # gentle etalon-like wiggle so the background is not trivially linear
    wiggle = 1.0 + 0.02 * np.sin(2 * np.pi * (E - 1.55) / 0.11)
    R_ref = base * wiggle
    R_sam = R_ref * true_feature(E)
    noise = lambda: RNG.normal(1.0, 0.003, size=E.size)
    return E, R_sam * noise(), R_ref * noise()


# ----------------------------------------------------------------------
# Case B: estimate the reference from the sample spectrum itself
# ----------------------------------------------------------------------
def estimate_background(E, R, feature_window=(1.67, 1.73), order=1):
    """Mask the resonance region and fit a polynomial through the rest.

    order=1 is a straight line, order=2..3 can follow curvature — but
    every extra order can also start eating the feature's wings. There
    is no universally right answer; the useful discipline is to try a
    couple of orders and treat the spread as a systematic error bar.
    """
    m = (E < feature_window[0]) | (E > feature_window[1])
    coeffs = np.polyfit(E[m], R[m], order)
    return np.polyval(coeffs, E)


def main():
    E, R_sam, R_ref = make_spectra()

    # --- case A: real reference ------------------------------------------
    rc_true_ref = (R_sam - R_ref) / R_ref

    # --- case B: interpolated backgrounds of increasing order -------------
    rc_est = {}
    for order in (1, 2, 3):
        B = estimate_background(E, R_sam, order=order)
        rc_est[order] = (R_sam - B) / B

    truth = true_feature(E) - 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].plot(E, R_ref, color="0.55", lw=1, label="reference (off-sample)")
    axes[0].plot(E, R_sam, color="C0", lw=1, label="sample")
    axes[0].plot(E, estimate_background(E, R_sam, order=2), "r--", lw=1.2,
                 label="estimated background (order 2)")
    axes[0].axvspan(1.67, 1.73, color="0.9", label="masked feature window")
    axes[0].set(xlabel="energy (eV)", ylabel="counts (a.u.)",
                title="raw spectra: the feature rides on\nlamp x system response")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(E, truth, "k-", lw=2.2, alpha=0.35, label="true RC")
    axes[1].plot(E, rc_true_ref, color="C0", lw=1.1,
                 label="case A: measured reference")
    axes[1].set(xlabel="energy (eV)", ylabel="RC", title="with a real reference:\nclean recovery")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(E, truth, "k-", lw=2.2, alpha=0.35, label="true RC")
    for order, rc in rc_est.items():
        axes[2].plot(E, rc, lw=1.0, label=f"case B: interpolated, order {order}")
    axes[2].set(xlabel="energy (eV)", ylabel="RC",
                title="without a reference: answer depends\non the background model (open question!)")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s1_02_reflectance_contrast_background.png", dpi=150)
    plt.show()

    # quantify the systematic: dip depth vs background choice
    print("recovered dip depth (truth = 0.120):")
    print(f"  case A, measured reference : {-rc_true_ref.min():.4f}")
    for order, rc in rc_est.items():
        print(f"  case B, polynomial order {order}: {-rc.min():.4f}")


if __name__ == "__main__":
    main()
