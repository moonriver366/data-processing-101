# TMDC Optical Spectroscopy — Data Processing Tutorial

A self-contained tutorial distilled from the data-processing code used in our
lab for optical spectroscopy of 2D semiconductors (TMDCs): gated PL, power
dependence, time-resolved PL, transient absorption microscopy, and
exciton-polariton dispersion analysis.

**No real measurement data is included.** Every example generates its own
synthetic data with known ground truth, so you can check that each processing
step actually recovers what was put in.

**Interactive version:** open [`docs/index.html`](docs/index.html) in a
browser (or the GitHub Pages site) for live, slider-driven versions of every
figure alongside the Python code.

---

## The processing pipeline, conceptually

Across all our experiments the pipeline is the same five stages:

```
   raw files          calibrate            preprocess              fit                 visualize
┌─────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ .asc / .csv │   │ pixel → nm/eV  │   │ dark/baseline  │   │ bounded        │   │ heatmap +      │
│ .txt stacks │ → │ pixel → angle  │ → │ subtraction,   │ → │ physics-model  │ → │ crosshair +    │
│ metadata in │   │ E = 1240/λ     │   │ NaN-aware      │   │ least squares  │   │ linecuts +     │
│  filenames  │   │ power calib    │   │ smoothing,     │   │ + quality gate │   │ sliders        │
└─────────────┘   └────────────────┘   │ chirp corr.    │   └────────────────┘   └────────────────┘
                                       └────────────────┘
```

1. **Load & parse.** Data arrives as plain-text arrays (Andor `.asc` spectra,
   tab-delimited TA stacks, TCSPC histograms). Sweep metadata lives in
   filenames with a compact convention — `P1p5uW` = 1.5 µW, `Vfm2p5` =
   front gate −2.5 V (`p` = decimal point, `m` = minus) — parsed with regex.
   Measurement sessions are timestamped folders; the analysis script is
   copied into the folder it analyzes, so every dataset keeps a frozen,
   working copy of its own processing.

2. **Calibrate.** Pixel → wavelength is a two-point linear map; wavelength →
   energy is `E[eV] = 1239.84 / λ[nm]`; camera row → emission angle (k-space)
   is another linear map. Excitation power comes from interpolating a
   measured calibration table (filter-wheel angle → power).

3. **Preprocess.** Dark counts / baseline are estimated from signal-free
   regions (tail wavelengths, pre-pulse time bins) — often per row.
   Noise floors use the median absolute deviation (robust to peaks).
   Maps with missing points are smoothed with **NaN-aware Gaussian
   filtering** (normalized convolution). Pump-probe maps get **chirp
   correction**: a polynomial t₀(λ) is fit to the coherent artifact and each
   wavelength column is re-interpolated onto a common time axis.

4. **Fit physics models with bounded least squares.** All fitting is
   `scipy.optimize.curve_fit` / `least_squares` with explicit physical bounds
   and a **quality gate** (SNR, R², parameter sanity) — a bad fit returns
   `None`/NaN and leaves a gap rather than a garbage point. Sweeps reuse the
   previous point's fit as the next initial guess. The standard models:

   | Observable | Model |
   |---|---|
   | PL spectrum | Gaussian / Lorentzian peaks (exciton X⁰ + trion) fitted in windows, then refined globally |
   | PL vs power | Hill saturation `I = Imax·Pⁿ/(Kⁿ+Pⁿ) + c`; onset located via `dI/d(log P)` |
   | TRPL decay | multi-exponential **analytically convolved with a Gaussian IRF** (erfc form); model-free 1/e lifetime as cross-check |
   | TA transients | exponential decays with rise; spatial Gaussian σ²(t) for diffusion |
   | Reflectance contrast | `RC = (I_sample − I_bg) / I_bg` |
   | Polariton dispersion | 2×2 coupled oscillator → LP/UP branches, Rabi splitting 2g, Hopfield fractions; Fano/CMT lineshapes for full spectra |

5. **Visualize interactively.** The signature layout is a **2D heatmap with a
   click-to-move crosshair and two linecuts** (spectral cut + spatial/time
   cut), plus a frame slider for 3D stacks — built with pyqtgraph for the
   instrument-side viewers and matplotlib widgets for quick-look scripts.
   Diverging colormaps (RdBu) for signed ΔR/R, sequential (viridis/magma)
   for intensity. Results export to CSV with axes in the header row/column.

---

## Examples

Each script is standalone: `python examples/<name>.py` (needs numpy, scipy,
matplotlib — `pip install -r requirements.txt`).

| # | Script | What it teaches |
|---|---|---|
| 01 | [`01_pl_spectrum_two_peak_fit.py`](examples/01_pl_spectrum_two_peak_fit.py) | Baseline correction from signal-free windows, MAD noise floor, windowed Lorentzian fits with quality gates, sequential-then-global two-peak decomposition |
| 02 | [`02_power_dependence_saturation.py`](examples/02_power_dependence_saturation.py) | Filename-metadata parsing (`P1p5uW`), log-log power laws, Hill saturation fit, NaN-aware dI/d(log P) derivative |
| 03 | [`03_trpl_irf_convolved_decay.py`](examples/03_trpl_irf_convolved_decay.py) | Analytic IRF-convolved bi-exponential (erfc form), Poisson-weighted fitting, residual diagnostics, model-free 1/e lifetime |
| 04 | [`04_ta_map_chirp_correction.py`](examples/04_ta_map_chirp_correction.py) | Pump-probe ΔR/R maps, polynomial t₀(λ) chirp model, column-wise re-interpolation (`map_coordinates`), transient + spectral linecuts |
| 05 | [`05_gate_dependent_pl_map.py`](examples/05_gate_dependent_pl_map.py) | Energy-vs-gate colormaps, NaN-aware Gaussian smoothing, double-Gaussian species tracking (exciton → trion transfer) |
| 06 | [`06_polariton_dispersion_fit.py`](examples/06_polariton_dispersion_fit.py) | Angle-resolved R(θ,E) maps, per-row ridge extraction, coupled-oscillator fit of both branches, Rabi splitting + Hopfield fractions |
| 07 | [`07_interactive_map_viewer.py`](examples/07_interactive_map_viewer.py) | The lab's standard interactive viewer: heatmap + crosshair + linecuts + frame slider, with cursor memory |

Rendered output of every script is in [`figures/`](figures/).

---

## Habits worth copying

- **Synthesize before you analyze.** Every pipeline here is validated on
  synthetic data with known truth first. If the code can't recover known
  parameters, it has no business touching real data.
- **Bound every fit.** Unbounded `curve_fit` on noisy spectra will happily
  return a 0.001 nm-wide peak. Physical bounds + a quality gate turn silent
  failures into visible gaps.
- **Handle NaN as a first-class value.** Skipped sweep points, failed fits,
  and dead pixels all become NaN and propagate cleanly through normalized
  convolution and NaN-aware derivatives.
- **Keep the script with the data.** Copying the analysis script into each
  measurement folder trades DRY for reproducibility — the exact code that
  produced a figure is always sitting next to its data.
- **Prefer analytic convolution over deconvolution.** Fitting an
  IRF-convolved model is stable; deconvolving noisy counts is not.
