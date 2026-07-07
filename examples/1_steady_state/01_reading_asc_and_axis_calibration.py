"""
1.01 — Reading an .asc image and calibrating the axes
=====================================================

The first file a new sample usually produces is an Andor .asc: comment
lines starting with '#', then a matrix whose first column is the
wavelength axis and whose remaining columns are the pixels along the
spectrometer slit. That second axis is where the thinking happens —
depending on the optics it is either

  * REAL SPACE (position on the sample), calibrated from magnification
    and cross-checked with a diffraction-limited laser spot, or
  * K SPACE (angle of emission), when the back focal plane of the
    objective is imaged onto the slit; the objective NA sets the edge.

This script writes a fake .asc to disk, reads it back the way I read
real ones, and builds both axis calibrations. All numbers synthetic.

Run:  python 01_reading_asc_and_axis_calibration.py
"""

import io
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=1)


# ----------------------------------------------------------------------
# Step 0: fabricate an .asc so the parser has something real to chew on
# ----------------------------------------------------------------------
def write_fake_asc():
    wl = np.linspace(700.0, 800.0, 340)
    npx = 200
    # a spot: Gaussian in space, one spectral feature
    px = np.arange(npx)
    img = (np.exp(-0.5 * ((wl[:, None] - 752) / 6.0) ** 2)
           * np.exp(-0.5 * ((px[None, :] - 96) / 9.0) ** 2) * 900.0 + 100.0)
    img = RNG.poisson(img).astype(float)
    buf = io.StringIO()
    buf.write("# Andor Solis export (synthetic)\n")
    buf.write("# exposure_s: 1.0\n# grating: 300 l/mm\n")
    for i in range(wl.size):
        buf.write("\t".join([f"{wl[i]:.4f}"] + [f"{v:.1f}" for v in img[i]]) + "\n")
    return buf.getvalue()


# ----------------------------------------------------------------------
# Step 1: the parser I actually use — skip '#', autodetect delimiter,
# first column is wavelength, the rest is the image
# ----------------------------------------------------------------------
def read_asc(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", "\t").split()
        rows.append([float(p) for p in parts])
    arr = np.asarray(rows)
    wavelength_nm = arr[:, 0]
    image = arr[:, 1:]                      # (n_wavelength, n_spatial_px)
    return wavelength_nm, image


# ----------------------------------------------------------------------
# Step 2a: REAL-SPACE calibration
# ----------------------------------------------------------------------
def real_space_um_per_px(camera_pitch_um=16.0, f_objective_mm=2.0,
                         f_tube_mm=200.0):
    """First-principles number: magnification M = f_tube / f_objective,
    so one camera pixel maps to pitch / M on the sample."""
    M = f_tube_mm / f_objective_mm
    return camera_pitch_um / M


def gaussian(x, A, x0, sigma, c):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


def check_with_laser_spot(profile_counts, wavelength_um=0.532, NA=0.8):
    """Sanity check the magnification number with a diffraction-limited
    laser spot imaged through the same optics.

    A well-focused Gaussian-ish spot has FWHM ~ 0.51 * lambda / NA on
    the sample. Fitting the imaged spot in *pixels* then gives an
    independent um-per-pixel estimate. Real spots carry aberrations, so
    treat this as an upper bound on the pixel size (the spot can only
    be bigger than the limit, never smaller).
    """
    px = np.arange(profile_counts.size)
    p0 = [profile_counts.max(), float(np.argmax(profile_counts)), 3.0,
          float(np.median(profile_counts))]
    popt, _ = curve_fit(gaussian, px, profile_counts, p0=p0)
    fwhm_px = 2.355 * abs(popt[2])
    fwhm_um_expected = 0.51 * wavelength_um / NA
    return fwhm_um_expected / fwhm_px, fwhm_px


# ----------------------------------------------------------------------
# Step 2b: K-SPACE calibration
# ----------------------------------------------------------------------
def k_space_angle_axis(n_px, edge_lo_px, edge_hi_px, NA=0.8):
    """When the back focal plane is imaged, the illuminated disk ends
    exactly at the NA of the objective: sin(theta_max) = NA. Position
    across the disk is linear in sin(theta), NOT in theta, so

        sin(theta)[px] = NA * (px - center) / radius
        theta[px]      = arcsin( ... )

    The disk edges (found from a white-light BFP image) give center and
    radius. Angle spacing therefore stretches toward the edge.
    """
    center = 0.5 * (edge_lo_px + edge_hi_px)
    radius = 0.5 * (edge_hi_px - edge_lo_px)
    px = np.arange(n_px)
    sin_theta = np.clip(NA * (px - center) / radius, -0.999, 0.999)
    return np.degrees(np.arcsin(sin_theta))


def main():
    wl, image = read_asc(write_fake_asc())
    print(f"parsed image: {image.shape[0]} wavelengths x {image.shape[1]} pixels")

    # ---- real space -----------------------------------------------------
    umpp_mag = real_space_um_per_px()
    # fake a diffraction-limited 532 nm spot imaged on the same camera:
    true_umpp = 0.16
    fwhm_um_true = 0.51 * 0.532 / 0.8 * 1.10          # 10% aberration
    px = np.arange(200)
    spot = gaussian(px, 3000, 101.3, fwhm_um_true / true_umpp / 2.355, 40)
    spot = RNG.poisson(np.clip(spot, 0, None)).astype(float)
    umpp_spot, fwhm_px = check_with_laser_spot(spot)
    print(f"um/px from magnification : {umpp_mag:.4f}")
    print(f"um/px from laser spot    : {umpp_spot:.4f} "
          f"(spot FWHM = {fwhm_px:.1f} px; upper bound, spot carries aberrations)")

    x_um = (np.arange(image.shape[1]) - image.shape[1] / 2) * umpp_mag

    # ---- k space --------------------------------------------------------
    theta_deg = k_space_angle_axis(image.shape[1], edge_lo_px=8, edge_hi_px=192)

    # ---- plots ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    im = axes[0].imshow(image, extent=[x_um[0], x_um[-1], wl[-1], wl[0]],
                        aspect="auto", cmap="magma")
    axes[0].set(xlabel="position on sample (um)", ylabel="wavelength (nm)",
                title="real-space calibration")
    fig.colorbar(im, ax=axes[0], label="counts")

    axes[1].plot(px, spot, ".", ms=3, color="0.6", label="laser spot (synthetic)")
    axes[1].plot(px, gaussian(px, *curve_fit(gaussian, px, spot,
                 p0=[3000, 100, 3, 40])[0]), "r-", lw=1.3, label="Gaussian fit")
    axes[1].set(xlabel="camera pixel", ylabel="counts",
                title="diffraction-limited spot cross-check")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(np.arange(image.shape[1]), theta_deg, lw=1.5)
    axes[2].set(xlabel="camera pixel", ylabel="emission angle (deg)",
                title="k-space axis: linear in sin(theta),\nstretched in theta near NA edge")
    axes[2].axhline(np.degrees(np.arcsin(0.8)), color="0.6", ls=":",
                    label="asin(NA) = 53.1 deg")
    axes[2].axhline(-np.degrees(np.arcsin(0.8)), color="0.6", ls=":")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s1_01_asc_axis_calibration.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
