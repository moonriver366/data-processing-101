"""
Example 07 — Interactive viewer: heatmap + crosshair + linecuts + slider
========================================================================

The signature interactive layout used across all the lab viewers
(pyqtgraph in the Qt tools, matplotlib widgets in the quick-look
scripts), reproduced here in pure matplotlib so it runs anywhere:

    +-----------------+-------------------+
    |   2D map        |  spectral linecut |    <- row at crosshair
    |  (image + X)    |                   |
    +-----------------+-------------------+
    | vertical linecut|  time slider      |    <- column at crosshair
    +-----------------+-------------------+

Interactions:
  * click on the map        -> move the crosshair, update both linecuts
  * drag the delay slider   -> show a different time frame of the stack
  * the cursor position is remembered across frames (like the per-file
    cursor memory in the lab viewers)

The data is a synthetic 3D TA stack (delay, y, wavelength): a decaying,
diffusing Gaussian spot — no real data anywhere.

Run:  python 07_interactive_map_viewer.py   (needs an interactive backend)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

RNG = np.random.default_rng(seed=1)


# ----------------------------------------------------------------------
# Synthetic (delay, y, wavelength) stack: decaying + diffusing spot
# ----------------------------------------------------------------------
def make_stack():
    delays = np.concatenate([np.linspace(0, 5, 26), np.linspace(5.5, 60, 30)])
    y_um = np.linspace(-8, 8, 96)                       # spatial axis
    wl = np.linspace(700, 800, 128)                     # spectral axis
    D = 0.35                                            # um^2/ps diffusion
    sig0, tau = 1.2, 12.0

    stack = np.empty((delays.size, y_um.size, wl.size))
    for i, t in enumerate(delays):
        sig2 = sig0 ** 2 + 2 * D * t                    # MSD grows linearly
        spatial = np.exp(-0.5 * y_um ** 2 / sig2)
        spectral = -np.exp(-0.5 * ((wl - 750) / 8) ** 2)
        stack[i] = np.exp(-t / tau) * np.outer(spatial, spectral) * 1e-3
    stack += RNG.normal(0, 2.5e-5, size=stack.shape)
    return delays, y_um, wl, stack


class MapViewer:
    """Minimal clone of the lab heatmap+linecut viewer."""

    def __init__(self, delays, y_um, wl, stack):
        self.delays, self.y_um, self.wl, self.stack = delays, y_um, wl, stack
        self.i_t = 10
        self.cur_y, self.cur_wl = 0.0, 750.0            # crosshair position

        self.fig = plt.figure(figsize=(11.5, 7.5))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.5, 1],
                                   height_ratios=[1.7, 1], hspace=0.3, wspace=0.25)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        self.ax_vert = self.fig.add_subplot(gs[1, 0])
        self.ax_dyn = self.fig.add_subplot(gs[1, 1])

        vmax = np.nanmax(np.abs(stack))
        self.im = self.ax_map.imshow(
            stack[self.i_t], extent=[wl[0], wl[-1], y_um[0], y_um[-1]],
            origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        self.fig.colorbar(self.im, ax=self.ax_map, label="$\\Delta R/R$")
        # crosshair lines (dashed, like pg.InfiniteLine in the Qt viewers)
        self.ch_h = self.ax_map.axhline(self.cur_y, color="k", lw=0.8, ls="--")
        self.ch_v = self.ax_map.axvline(self.cur_wl, color="k", lw=0.8, ls="--")
        self.ax_map.set(xlabel="wavelength (nm)", ylabel="y ($\\mu$m)")

        (self.spec_line,) = self.ax_spec.plot([], [], color="C2", lw=1.3)
        self.ax_spec.set(xlabel="wavelength (nm)", ylabel="$\\Delta R/R$",
                         title="spectral cut (row at crosshair)")
        (self.vert_line,) = self.ax_vert.plot([], [], color="C4", lw=1.3)
        self.ax_vert.set(xlabel="y ($\\mu$m)", ylabel="$\\Delta R/R$",
                         title="spatial cut (column at crosshair)")
        (self.dyn_line,) = self.ax_dyn.plot([], [], "o-", ms=3, lw=1, color="C0")
        self.time_marker = self.ax_dyn.axvline(delays[self.i_t], color="0.5", ls=":")
        self.ax_dyn.set(xlabel="delay (ps)", ylabel="$\\Delta R/R$",
                        title="dynamics at crosshair")

        ax_sl = self.fig.add_axes([0.15, 0.015, 0.6, 0.025])
        self.slider = Slider(ax_sl, "delay index", 0, delays.size - 1,
                             valinit=self.i_t, valstep=1)
        self.slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.redraw()

    # -- event handlers ---------------------------------------------------
    def on_slider(self, val):
        self.i_t = int(val)
        self.redraw()

    def on_click(self, event):
        if event.inaxes is not self.ax_map or event.xdata is None:
            return
        self.cur_wl, self.cur_y = event.xdata, event.ydata
        self.redraw()

    # -- drawing ------------------------------------------------------------
    def redraw(self):
        iy = int(np.argmin(np.abs(self.y_um - self.cur_y)))
        iw = int(np.argmin(np.abs(self.wl - self.cur_wl)))
        frame = self.stack[self.i_t]

        self.im.set_data(frame)
        self.ch_h.set_ydata([self.cur_y, self.cur_y])
        self.ch_v.set_xdata([self.cur_wl, self.cur_wl])
        self.ax_map.set_title(f"frame {self.i_t}: t = {self.delays[self.i_t]:.1f} ps")

        self.spec_line.set_data(self.wl, frame[iy, :])
        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self.vert_line.set_data(self.y_um, frame[:, iw])
        self.ax_vert.relim(); self.ax_vert.autoscale_view()
        self.dyn_line.set_data(self.delays, self.stack[:, iy, iw])
        self.time_marker.set_xdata([self.delays[self.i_t]] * 2)
        self.ax_dyn.relim(); self.ax_dyn.autoscale_view()
        self.fig.canvas.draw_idle()


def main():
    delays, y_um, wl, stack = make_stack()
    viewer = MapViewer(delays, y_um, wl, stack)
    viewer.fig.savefig("07_interactive_map_viewer.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
