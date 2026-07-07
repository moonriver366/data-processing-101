# Toddler's Toolbox of Data Processing

A proposed pass for characterizing a new sample by optical spectroscopy, with the
data-processing and instrument details behind each step. These are notes on the working
logic I follow when a fresh sample lands on the optical table — not a textbook. If you
disagree with any of it, that's what issues and PRs are for; several points below are
genuinely open questions and marked as such.

Everything here uses **synthetic data with known ground truth** — no real measurements —
so every processing step can be checked against what was put in.

**Interactive version:** the
[GitHub Pages site](https://moonriver366.github.io/data-processing-101/) (source in
[`docs/index.html`](docs/index.html)) has live, slider-driven versions of every figure,
with the formulas and instrument schematics next to the Python.

---

## The pass, in two steps

Steady state first — it's cheap, it carries no time-resolution artifacts to misread, and it
tells you where to point everything that follows. Then time-resolved.

### Step 1 — Linear / steady-state spectrum (reflectance & PL)

| § | Question | Script |
|---|---|---|
| 1.1 | How to read the raw Andor `.asc` and calibrate the spatial axis: pixel → position (magnification, cross-checked with a diffraction-limited spot) or pixel → angle (objective NA sets the k-space edge) | [`01_reading_asc_and_axis_calibration.py`](examples/1_steady_state/01_reading_asc_and_axis_calibration.py) |
| 1.2 | Do you need k-space at all? Only if the feature disperses inside the light cone — set by the mode's effective mass (see the demo on the site) | — |
| 1.3a | Reflectance contrast, and where the background comes from — measured reference vs interpolated (open question when there's no off-sample region, e.g. a cavity that's everywhere) | [`02_reflectance_contrast_background.py`](examples/1_steady_state/02_reflectance_contrast_background.py) |
| 1.3b | Which lineshape — Lorentzian, Gaussian, Fano (with a background phase) — and how to set initial guesses and bounds so `curve_fit` converges | [`03_lineshapes_lorentzian_gaussian_fano.py`](examples/1_steady_state/03_lineshapes_lorentzian_gaussian_fano.py) |
| 1.3c | Why RC lineshapes look Fano even when the resonance is a plain Lorentz oscillator: interference with the stack (transfer-matrix demo) | [`04_tmm_thin_film_simulation.py`](examples/1_steady_state/04_tmm_thin_film_simulation.py) |
| 1.4 | What steady-state PL counts mean (time-integrated: rep-rate matters, intensity ∝ I₀·τ, so two peaks with intensity ratio *a* and lifetime ratio *b* show integrated ratio *ab*), and what the spatial profile means (diffusion during the lifetime: σ²_PL = σ²_laser + 2Dτ) | [`05_pl_and_spatial_profile.py`](examples/1_steady_state/05_pl_and_spatial_profile.py) |
| 1.5 | Sweeping power / temperature / gate: tracking a peak through a sweep with sequential constrained fits, and when that's the wrong thing to do | [`06_parameter_sweep_peak_tracking.py`](examples/1_steady_state/06_parameter_sweep_peak_tracking.py) |

### Step 2 — Time-resolved

**TRPL & PL imaging** — scripts in [`examples/2_time_resolved/`](examples/2_time_resolved/)

| § | Question | Script |
|---|---|---|
| 2.1 | What a TCSPC (PicoHarp) histogram is, what the IRF is, and why to fit the analytic IRF-convolved decay (the Origin `ConvolutedDecay4n` form) instead of deconvolving | [`01_trpl_irf_convolved_fit.py`](examples/2_time_resolved/01_trpl_irf_convolved_fit.py) |
| 2.2 | Two ways to a time-resolved spatial profile: point scan (stages + single-pixel detector) vs a 23-pixel SPAD array (sample the spot, 2D-Gaussian fit → σ) | [`02_spatial_profiles_scan_vs_array.py`](examples/2_time_resolved/02_spatial_profiles_scan_vs_array.py) |
| 2.3 | MSD analysis — and the four mechanisms that broaden a profile (diffusion, ballistic, annihilation, emission saturation), only some of which are transport | [`03_msd_transport_regimes.py`](examples/2_time_resolved/03_msd_transport_regimes.py) |
| 2.4 | Two species, one cascade: what spectrally-resolved imaging adds — the flat-top/saturation fingerprint (in profile and in MSD) vs real transport | [`04_two_species_cascade.py`](examples/2_time_resolved/04_two_species_cascade.py) |

**Transient reflectance (TA)** — scripts in [`examples/3_transient_reflectance/`](examples/3_transient_reflectance/)

| § | Question | Script |
|---|---|---|
| 2.5 (files) | The paired `para`/`sum` file format and how the `sum` blob reshapes into a (delay, position, spectral) stack | — |
| 2.5a | What actually makes a dR/R signal: amplitude vs shift vs broadening — and why TA is *not* automatically population (toggle a Lorentzian dip ↔ Fano) | [`01_dr_over_r_origins.py`](examples/3_transient_reflectance/01_dr_over_r_origins.py) |
| 2.5b | Wavelength calibration with long/shortpass filter edges, and chirp correction (polynomial t₀(λ) + column re-interpolation) | [`02_ta_map_chirp_correction.py`](examples/3_transient_reflectance/02_ta_map_chirp_correction.py) |
| 2.5c | The whole thing in one interactive viewer: map + crosshair, spectrum at the cursor, spatial-cut fit → σ(t), dynamics (pre-t₀ baseline, plain-exponential fit — why the IRF matters less here than in TRPL), and MSD → D | [`03_ta_map_viewer_full_analysis.py`](examples/3_transient_reflectance/03_ta_map_viewer_full_analysis.py) |

Rendered output of every script is in [`figures/`](figures/).

Planned but deferred: the k-space toolkit (dispersion extraction, coupled-oscillator fits,
Hopfield analysis) for strongly dispersive samples such as polaritons. It deserves its own step.

---

## Habits I try to follow

- **Synthesize before you analyze.** Known truth in, known truth out — then real data.
- **Bound every fit, gate every fit.** Physical bounds plus a signal/quality gate: a failed
  fit records NaN and leaves a visible gap, never a silent garbage point. A parameter pinned
  at its bound means the model or window is wrong.
- **Look at residuals, not R² alone.** Structure in the residuals means the model is wrong
  even when the number looks fine.
- **Time-integrated ≠ instantaneous.** Steady-state PL sees I₀·τ; pulsed excitation at the
  same average power changes per-pulse density with rep rate.
- **Broadening ≠ transport, dR/R ≠ population, extremum ≠ resonance position.** Three of the
  most tempting shortcuts in this business; Step 2 and the Fano/TMM sections each exist to
  kill one of them.

## Running the examples

```
pip install -r requirements.txt        # numpy, scipy, matplotlib
python examples/1_steady_state/01_reading_asc_and_axis_calibration.py
```

Every script is standalone and generates its own data. The viewer scripts (e.g.
`examples/3_transient_reflectance/03_ta_map_viewer_full_analysis.py`) want an interactive
matplotlib backend.

---

*Comments, corrections and contributions are all welcome — especially on the parts marked as
open questions. Last edited 2026-07-07 by Minxue Wang.*
