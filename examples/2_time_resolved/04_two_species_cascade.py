"""
2.04 — Two species, one cascade: spectrally-resolved imaging
============================================================

If the spectrum has two emission bands, spectrally-resolved PL imaging
(a band-pass in front of the imaging path, or a spectrometer with an
imaging axis) gives one spatial profile PER SPECIES — and the two
profiles carry information the spectra alone don't.

Toy model I find genuinely instructive: pump creates only the HIGH
energy species H; H relaxes into the LOW energy species L; L has a
finite density of available states, so the transfer saturates:

    dn_H/dt = G(x, t) - n_H / tau_H - k * n_H * (1 - n_L / N_sat)
    dn_L/dt =           - n_L / tau_L + k * n_H * (1 - n_L / N_sat)

Predicted spatial signatures (all reproduced below):
  * H profile ~ follows the laser (narrow), maybe eaten at the center
    if the transfer is fast.
  * L profile is BROADER than the laser and FLAT-TOPPED at high pump:
    at the center n_L hits N_sat and clips, in the wings it doesn't.
    A "donut" or flat-top in one spectral window with a normal Gaussian
    in the other is a saturation fingerprint, not transport.
  * L dynamics at the center: slowed rise (blocked transfer) compared
    to the wings.

All data synthetic.

Run:  python 04_two_species_cascade.py
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_cascade(pump=6.0, N_sat=1.0, tau_H=1.0, tau_L=6.0, k=2.0,
                     sigma_L=0.8, L=12.0, nx=241, t_end=15.0, nt=3001):
    """Pulsed excitation at t=0, then free evolution on a 1D grid."""
    x = np.linspace(-L / 2, L / 2, nx)
    dt = t_end / (nt - 1)
    times = np.linspace(0, t_end, nt)
    nH = pump * np.exp(-0.5 * x ** 2 / sigma_L ** 2)
    nL = np.zeros_like(nH)
    snaps_t = np.linspace(0, t_end, 61)
    H_snaps, L_snaps, k_idx = [], [], 0
    H_int = np.zeros_like(nH)          # time-integrated emission per species
    L_int = np.zeros_like(nL)
    for i, t in enumerate(times):
        if k_idx < snaps_t.size and t >= snaps_t[k_idx] - 1e-9:
            H_snaps.append(nH.copy())
            L_snaps.append(nL.copy())
            k_idx += 1
        transfer = k * nH * np.clip(1.0 - nL / N_sat, 0.0, None)
        dH = -nH / tau_H - transfer
        dL = -nL / tau_L + transfer
        nH = np.clip(nH + dt * dH, 0, None)
        nL = np.clip(nL + dt * dL, 0, None)
        H_int += nH / tau_H * dt        # emission ~ population / tau_rad
        L_int += nL / tau_L * dt
    return x, snaps_t, np.array(H_snaps), np.array(L_snaps), H_int, L_int


def main():
    x, ts, H, Ls, H_int, L_int = simulate_cascade(pump=6.0, N_sat=1.0)
    _, _, _, _, H_lo, L_lo = simulate_cascade(pump=0.3, N_sat=1.0)

    laser = np.exp(-0.5 * x ** 2 / 0.8 ** 2)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # time-integrated profiles = what a spectrally-filtered camera sees
    axes[0].plot(x, laser, color="0.7", lw=2, label="laser")
    axes[0].plot(x, H_int / H_int.max(), color="C0", lw=1.6, label="species H (high E)")
    axes[0].plot(x, L_int / L_int.max(), color="C3", lw=1.6, label="species L (low E)")
    axes[0].set(xlabel="x (um)", ylabel="normalized integrated emission",
                title="high pump: L is flat-topped & broader\n(center clipped at N_sat)",
                xlim=(-5, 5))
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(x, laser, color="0.7", lw=2, label="laser")
    axes[1].plot(x, H_lo / H_lo.max(), color="C0", lw=1.6, label="species H")
    axes[1].plot(x, L_lo / L_lo.max(), color="C3", lw=1.6, label="species L")
    axes[1].set(xlabel="x (um)",
                title="low pump, same sample: both follow\nthe laser -> it was saturation",
                xlim=(-5, 5))
    axes[1].legend(frameon=False, fontsize=8)

    # dynamics of L at center vs wing: the blocked-rise fingerprint
    ic = np.argmin(np.abs(x))
    iw = np.argmin(np.abs(x - 1.6))
    axes[2].plot(ts, Ls[:, ic] / Ls[:, ic].max(), color="C3", lw=1.6,
                 label="L at center (rise blocked)")
    axes[2].plot(ts, Ls[:, iw] / Ls[:, iw].max(), color="C3", ls="--", lw=1.6,
                 label="L in the wing")
    axes[2].plot(ts, H[:, ic] / H[:, ic].max(), color="C0", lw=1.2,
                 label="H at center")
    axes[2].set(xlabel="time (ns)", ylabel="normalized population",
                title="same species, different positions:\nspatial + temporal beats either alone")
    axes[2].legend(frameon=False, fontsize=8)

    # MSD(t) of each species: L's apparent broadening is saturation, not transport
    def sigma2(prof):
        base = np.median(np.concatenate([prof[:8], prof[-8:]]))
        w = np.clip(prof - base, 0, None)
        if w.sum() <= 0:
            return np.nan
        mu = np.sum(x * w) / np.sum(w)
        return np.sum((x - mu) ** 2 * w) / np.sum(w)
    s2H = np.array([sigma2(H[i]) for i in range(len(ts))])
    s2L = np.array([sigma2(Ls[i]) for i in range(len(ts))])
    axes[3].plot(ts, s2H - np.nanmin(s2H), "o-", ms=3, color="C0", label="H")
    axes[3].plot(ts, s2L - np.nanmin(s2L), "s-", ms=3, color="C3", label="L")
    axes[3].set(xlabel="time (ns)", ylabel="MSD = sigma² - sigma0² (um²)",
                title="MSD(t): L 'spreads' as its center\nsaturates — no real transport")
    axes[3].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("s2_04_two_species_cascade.png", dpi=150)
    plt.show()

    # quantify the flat-top: ratio of FWHM to laser FWHM
    def fwhm(y):
        above = x[y > 0.5 * y.max()]
        return above.max() - above.min()
    print(f"FWHM / laser FWHM: H = {fwhm(H_int)/fwhm(laser):.2f}, "
          f"L = {fwhm(L_int)/fwhm(laser):.2f} (high pump)")
    print(f"                   H = {fwhm(H_lo)/fwhm(laser):.2f}, "
          f"L = {fwhm(L_lo)/fwhm(laser):.2f} (low pump)")


if __name__ == "__main__":
    main()
