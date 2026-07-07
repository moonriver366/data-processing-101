# Data Processing 101

My take on going from raw optical-spectroscopy files to numbers you can
trust. This is not a textbook — it's the working logic I follow when a new
sample lands on the optical table, written down with runnable code. If you
disagree with any of it, that's what issues and PRs are for; several points
below are genuinely open questions and marked as such.

Everything here uses **synthetic data with known ground truth** — no real
measurements — so every processing step can be checked against what was put
in. That's also rule zero: *if a pipeline can't recover known parameters
from synthetic data, it has no business touching real data.*

**Interactive version:** [`docs/index.html`](docs/index.html) (or the GitHub
Pages site) has live, slider-driven versions of most figures next to the
Python code.

---

## The logic: a new sample just arrived

The pass I run, in order. Steady-state first — it's cheap, it can't be
misinterpreted by time-resolution artifacts, and it tells you where to look
with everything that comes after.

### Part 1 — Steady state (linear reflectance & PL)

| § | Question | Script |
|---|---|---|
| 1.1 | How do I read the raw file and what do the axes mean? Pixel → position (magnification, cross-checked with a diffraction-limited laser spot) or pixel → angle (objective NA sets the edge of k-space) | [`01_reading_asc_and_axis_calibration.py`](examples/1_steady_state/01_reading_asc_and_axis_calibration.py) |
| 1.2 | Do I need k-space at all? Only if the feature disperses inside the light cone — see the demo on the site | — |
| 1.3 | What is reflectance contrast, and where does the background come from — measured reference vs interpolated (open question when there's no off-sample region, e.g. a cavity that's everywhere) | [`02_reflectance_contrast_background.py`](examples/1_steady_state/02_reflectance_contrast_background.py) |
| 1.3 | Which lineshape — Lorentzian, Gaussian, Fano — and how to set initial guesses and bounds so `curve_fit` actually converges | [`03_lineshapes_lorentzian_gaussian_fano.py`](examples/1_steady_state/03_lineshapes_lorentzian_gaussian_fano.py) |
| 1.3 | Why RC lineshapes look Fano even when the resonance is a plain Lorentz oscillator: interference with the stack (transfer-matrix demo) | [`04_tmm_thin_film_simulation.py`](examples/1_steady_state/04_tmm_thin_film_simulation.py) |
| 1.4 | What steady-state PL counts actually mean (time-integrated! rep-rate matters, intensity ∝ n₀·τ), and what the spatial profile means (diffusion during the lifetime: σ²_PL = σ²_laser + 2Dτ) | [`05_pl_and_spatial_profile.py`](examples/1_steady_state/05_pl_and_spatial_profile.py) |
| 1.5 | Sweeping power / temperature / gate: tracking peaks through a sweep with sequential constrained fits, and when that's the wrong thing to do | [`06_parameter_sweep_peak_tracking.py`](examples/1_steady_state/06_parameter_sweep_peak_tracking.py) |

### Part 2 — Time-resolved: TRPL & PL imaging

| § | Question | Script |
|---|---|---|
| 2.1–2.2 | What a TCSPC histogram is, what the IRF is, and why to fit the analytic IRF-convolved decay instead of deconvolving | [`01_trpl_irf_convolved_fit.py`](examples/2_time_resolved/01_trpl_irf_convolved_fit.py) |
| 2.3 | Two ways to a time-resolved spatial profile: point scan (stages + single-pixel detector) vs detector array (sparse pixels + 2D Gaussian fit) | [`02_spatial_profiles_scan_vs_array.py`](examples/2_time_resolved/02_spatial_profiles_scan_vs_array.py) |
| 2.4 | MSD analysis — and the four mechanisms that broaden a profile (diffusion, ballistic, annihilation, emission saturation), only some of which are transport | [`03_msd_transport_regimes.py`](examples/2_time_resolved/03_msd_transport_regimes.py) |
| 2.5 | Two species, one cascade: what spectrally-resolved imaging adds — the flat-top/saturation fingerprint vs real transport | [`04_two_species_cascade.py`](examples/2_time_resolved/04_two_species_cascade.py) |

### Part 3 — Time-resolved: transient reflectance (TA)

| § | Question | Script |
|---|---|---|
| 3.1 | What actually makes a dR/R signal: amplitude vs shift vs broadening — and why TA is *not* automatically population | [`01_dr_over_r_origins.py`](examples/3_transient_reflectance/01_dr_over_r_origins.py) |
| 3.2 | Wavelength calibration with long/shortpass filter edges, and chirp correction (polynomial t₀(λ) + column re-interpolation) | [`02_ta_map_chirp_correction.py`](examples/3_transient_reflectance/02_ta_map_chirp_correction.py) |
| 3.3 | The whole thing in one interactive viewer: map + crosshair + spatial-cut fit + dynamics (with pre-t₀ baseline, plain-exponential fit — and why IRF matters less here than in TRPL) + MSD → D | [`03_ta_map_viewer_full_analysis.py`](examples/3_transient_reflectance/03_ta_map_viewer_full_analysis.py) |

Rendered output of every script is in [`figures/`](figures/).

Planned but deferred: the k-space toolkit (dispersion extraction, coupled-
oscillator fits, Hopfield analysis). It deserves its own part.

---

## Rules I try to follow

- **Synthesize before you analyze.** Known truth in, known truth out — then
  real data.
- **Bound every fit, gate every fit.** Physical bounds plus an SNR/R² gate:
  a failed fit records NaN and leaves a visible gap, never a silent garbage
  point. A parameter pinned at its bound means the model or window is wrong.
- **Look at residuals, not R² alone.** Structure in the residuals means the
  model is wrong even when the number looks fine.
- **Time-integrated ≠ instantaneous.** Steady-state PL sees n₀·τ; pulsed
  excitation at the same average power changes per-pulse density with rep
  rate.
- **Broadening ≠ transport, dR/R ≠ population, extremum ≠ resonance
  position.** The three most tempting shortcuts in this business; Parts 2–3
  and the Fano section each exist to kill one of them.
- **Keep the script with the data.** Copy the analysis script into the
  measurement folder it analyzes. Un-DRY, fully reproducible.

## Running the examples

```
pip install -r requirements.txt        # numpy, scipy, matplotlib
python examples/1_steady_state/01_reading_asc_and_axis_calibration.py
```

Every script is standalone and generates its own data. Script 3.03 (and
the other viewers) want an interactive matplotlib backend.

---

*Comments, corrections and contributions are all welcome — especially on the
parts marked as open questions.*
