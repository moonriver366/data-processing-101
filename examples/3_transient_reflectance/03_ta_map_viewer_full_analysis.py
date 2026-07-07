"""
3.03 — The full TA map analysis, in one interactive viewer
==========================================================

Everything from Parts 1-2 converges here. A spatially- and spectrally-
resolved TA stack (delay, y, wavelength) gets the standard treatment:

  * map + click-to-move crosshair + frame slider (the lab viewer layout)
  * SPATIAL CUT at the feature wavelength, fitted live with a Gaussian
    -> sigma(t)
  * DYNAMICS at the feature center, INCLUDING pre-time-zero points
    (they establish the baseline and catch pump scatter), fitted with a
    plain exponential after the rise. Why no IRF convolution here,
    unlike TRPL (2.01)? The time resolution of a pump-probe experiment
    is the optical cross-correlation of two ~100 fs pulses — orders of
    magnitude shorter than ps-ns dynamics — whereas a TCSPC detector
    IRF (tens-hundreds of ps) is often comparable to the lifetime.
    So here the rise is effectively a step, and fitting from just after
    it is fine.
  * MSD panel: sigma(t)^2 - sigma(0)^2 with a linear fit -> D.
    All the caveats of 2.03 apply: check power dependence before
    believing the D.
  * A trap worth noticing in the numbers: the fitted tau at the spot
    CENTER comes out shorter than the population lifetime (12 ps here),
    because diffusion carries signal out of the point being watched.
    Local decay != population decay whenever anything moves; transport
    belongs in the MSD panel, not in a single-point tau.

All data synthetic: a spectral dip, Gaussian in space, that decays
(tau = 12 ps) and diffuses (D = 0.35 um^2/ps).

Run:  python 03_ta_map_viewer_full_analysis.py   (interactive backend)
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

RNG = np.random.default_rng(seed=1)

D_TRUE, TAU_TRUE, SIG0 = 0.35, 12.0, 1.2   # um^2/ps, ps, um


def make_stack():
    delays = np.concatenate([np.linspace(-5, -0.5, 8),      # pre-t0 baseline
                             np.linspace(0, 5, 21),
                             np.linspace(5.5, 60, 28)])
    # field of view must stay much wider than the spot at the LAST delay
    # you intend to fit — otherwise truncation biases every sigma estimate
    y_um = np.linspace(-12, 12, 96)
    wl = np.linspace(700, 800, 128)
    stack = np.empty((delays.size, y_um.size, wl.size))
    for i, t in enumerate(delays):
        if t < 0:
            sig = np.zeros((y_um.size, wl.size))
        else:
            var = SIG0 ** 2 + 2 * D_TRUE * t
            spatial = np.exp(-0.5 * y_um ** 2 / var) * (SIG0 ** 2 / var) ** 0.5
            spectral = -np.exp(-0.5 * ((wl - 750) / 8) ** 2)
            sig = np.exp(-t / TAU_TRUE) * np.outer(spatial, spectral) * 2e-3
        stack[i] = sig + RNG.normal(0, 1.0e-5, size=(y_um.size, wl.size))
    return delays, y_um, wl, stack


def gaussian(x, A, x0, s, c):
    return A * np.exp(-0.5 * ((x - x0) / s) ** 2) + c


def fit_spatial_sigma(y_um, profile, noise=1e-5):
    """Gaussian fit of the (negative) spatial dip profile; None if too weak.

    Two gates that matter in practice: (1) require the dip to clear the
    noise floor by a solid factor — a fit to noise returns a confident
    garbage sigma; (2) pin the offset near zero: dR/R has no baseline by
    construction, and a free offset trades against width once the wings
    leave the field of view.
    """
    amp = profile.min()
    if amp > -8 * noise:
        return None
    try:
        popt, _ = curve_fit(gaussian, y_um, profile,
                            p0=[amp, 0.0, 1.5, 0.0],
                            bounds=([-np.inf, y_um.min(), 0.3, -3 * noise],
                                    [0.0, y_um.max(), 15.0, 3 * noise]),
                            maxfev=10000)
        return popt
    except RuntimeError:
        return None


def exp_decay(t, A, tau, c):
    return A * np.exp(-t / tau) + c


class TAViewer:
    def __init__(self):
        self.delays, self.y_um, self.wl, self.stack = make_stack()
        self.i_t = 12
        self.cur_y, self.cur_wl = 0.0, 750.0

        # precompute sigma(t) and center dynamics once (cheap enough live,
        # but this mirrors how the lab viewers cache derived quantities)
        self.sigmas = np.full(self.delays.size, np.nan)
        self.dip_amp = np.full(self.delays.size, np.nan)
        j750 = np.argmin(np.abs(self.wl - 750))
        for i, t in enumerate(self.delays):
            if t < 0:
                continue
            popt = fit_spatial_sigma(self.y_um, self.stack[i, :, j750])
            if popt is not None:
                self.sigmas[i] = abs(popt[2])
                self.dip_amp[i] = abs(popt[0])

        # exponential fit of the center dynamics, only after the rise
        iy0 = np.argmin(np.abs(self.y_um))
        self.dyn = self.stack[:, iy0, j750]
        m = self.delays > 1.0
        self.exp_popt, _ = curve_fit(exp_decay, self.delays[m], self.dyn[m],
                                     p0=[self.dyn[m].min(), 10.0, 0.0])

        # MSD linear fit -> D. Weight by the dip amplitude: sigma estimates
        # from late, weak frames are noisier, and because sigma^2 enters the
        # MSD, their scatter biases an unweighted fit upward.
        ok = np.isfinite(self.sigmas) & (self.delays > 0.5)
        msd = self.sigmas[ok] ** 2 - np.nanmin(self.sigmas[ok] ** 2)
        self.msd_slope, self.msd_icpt = np.polyfit(self.delays[ok], msd, 1,
                                                   w=self.dip_amp[ok])

        self._build()

    def _build(self):
        self.fig = plt.figure(figsize=(12.5, 8))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.5, 1],
                                   hspace=0.35, wspace=0.3)
        self.ax_map = self.fig.add_subplot(gs[0, :2])
        self.ax_spat = self.fig.add_subplot(gs[0, 2])
        self.ax_dyn = self.fig.add_subplot(gs[1, 0:2])
        self.ax_msd = self.fig.add_subplot(gs[1, 2])

        vmax = np.nanmax(np.abs(self.stack))
        self.im = self.ax_map.imshow(
            self.stack[self.i_t],
            extent=[self.wl[0], self.wl[-1], self.y_um[0], self.y_um[-1]],
            origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        self.fig.colorbar(self.im, ax=self.ax_map, label="dR/R")
        self.ch_h = self.ax_map.axhline(0, color="k", lw=0.8, ls="--")
        self.ch_v = self.ax_map.axvline(750, color="k", lw=0.8, ls="--")
        self.ax_map.set(xlabel="wavelength (nm)", ylabel="y (um)")

        (self.spat_pts,) = self.ax_spat.plot([], [], ".", ms=3, color="0.6")
        (self.spat_fit,) = self.ax_spat.plot([], [], "r-", lw=1.4)
        self.ax_spat.set(xlabel="y (um)", ylabel="dR/R",
                         title="spatial cut + Gaussian fit")

        (self.dyn_pts,) = self.ax_dyn.plot([], [], "o", ms=4, color="C0")
        (self.dyn_fit,) = self.ax_dyn.plot([], [], "r-", lw=1.4)
        self.dyn_tmark = self.ax_dyn.axvline(0, color="0.5", ls=":")
        self.ax_dyn.axhline(0, color="0.7", lw=0.7)
        self.ax_dyn.axvline(0, color="0.7", lw=0.7)
        self.ax_dyn.set(xlabel="delay (ps)", ylabel="dR/R",
                        title="dynamics at crosshair (pre-t0 shows the baseline; "
                              "exp fit after the rise)")

        ok = np.isfinite(self.sigmas)
        msd = self.sigmas ** 2 - np.nanmin(self.sigmas[ok] ** 2)
        self.ax_msd.plot(self.delays[ok], msd[ok], "s", ms=4, color="C2")
        tl = np.linspace(0, self.delays[-1], 50)
        self.ax_msd.plot(tl, self.msd_slope * tl + self.msd_icpt, "k--", lw=1,
                         label=f"slope/2 -> D = {self.msd_slope/2:.2f} um²/ps\n"
                               f"(truth {D_TRUE})")
        self.msd_tmark = self.ax_msd.axvline(0, color="0.5", ls=":")
        self.ax_msd.set(xlabel="delay (ps)", ylabel="MSD (um²)",
                        title="sigma(t)² - sigma0²")
        self.ax_msd.legend(frameon=False, fontsize=8)

        ax_sl = self.fig.add_axes([0.13, 0.015, 0.55, 0.02])
        self.slider = Slider(ax_sl, "delay idx", 0, self.delays.size - 1,
                             valinit=self.i_t, valstep=1)
        self.slider.on_changed(lambda v: (setattr(self, "i_t", int(v)),
                                          self.redraw()))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.redraw()

    def on_click(self, event):
        if event.inaxes is not self.ax_map or event.xdata is None:
            return
        self.cur_wl, self.cur_y = event.xdata, event.ydata
        self.redraw()

    def redraw(self):
        iy = int(np.argmin(np.abs(self.y_um - self.cur_y)))
        iw = int(np.argmin(np.abs(self.wl - self.cur_wl)))
        t = self.delays[self.i_t]
        frame = self.stack[self.i_t]

        self.im.set_data(frame)
        self.ch_h.set_ydata([self.cur_y] * 2)
        self.ch_v.set_xdata([self.cur_wl] * 2)
        self.ax_map.set_title(f"t = {t:.1f} ps")

        prof = frame[:, iw]
        self.spat_pts.set_data(self.y_um, prof)
        popt = fit_spatial_sigma(self.y_um, prof) if t >= 0 else None
        if popt is not None:
            self.spat_fit.set_data(self.y_um, gaussian(self.y_um, *popt))
            self.ax_spat.set_title(f"spatial cut: sigma = {abs(popt[2]):.2f} um")
        else:
            self.spat_fit.set_data([], [])
            self.ax_spat.set_title("spatial cut (no significant dip)")
        self.ax_spat.relim(); self.ax_spat.autoscale_view()

        dyn = self.stack[:, iy, iw]
        self.dyn_pts.set_data(self.delays, dyn)
        m = self.delays > 1.0
        try:
            popt_d, _ = curve_fit(exp_decay, self.delays[m], dyn[m],
                                  p0=[dyn[m].min(), 10.0, 0.0])
            tl = np.linspace(1.0, self.delays[-1], 100)
            self.dyn_fit.set_data(tl, exp_decay(tl, *popt_d))
            self.ax_dyn.set_title(f"dynamics at crosshair: tau = {popt_d[1]:.1f} ps "
                                  f"(plain exp — rise ~ pulse xcorr, no IRF needed)")
        except RuntimeError:
            self.dyn_fit.set_data([], [])
        self.dyn_tmark.set_xdata([t] * 2)
        self.msd_tmark.set_xdata([t] * 2)
        self.ax_dyn.relim(); self.ax_dyn.autoscale_view()
        self.fig.canvas.draw_idle()


def main():
    viewer = TAViewer()
    viewer.fig.savefig("s3_03_ta_viewer_full_analysis.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
