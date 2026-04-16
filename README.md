# DSE4231 Group Project 
# Heterogeneous Treatment Effect Estimation - Meta-Learner Comparison Study

Three simulation studies comparing R-, S-, T-, and X-learners (with lasso, boosting, and kernel base methods) across distinct data-generating processes that stress different aspects of causal inference: severe treatment imbalance, null treatment effects, and sign-changing conditional treatment effects.

## Study overview

Each case study is a self-contained simulation with a realistic clinical DGP, a pre-specified set of subgroups, and a common learner panel of 12 meta-learner × base-method combinations plus baselines (`ols_inter`, `zero_pred`, and — in the preventative study — a constant predictor).

| Case study | Core challenge | Treatment effect structure | Primary output |
|---|---|---|---|
| **Smoking and Longevity** | Severe treatment imbalance (~7.5% treated) | Heterogeneous τ(X), all non-negative | CSVs + propensity histograms |
| **Preventative Healthcare (SMS)** | Genuine null effect — tests whether learners hallucinate heterogeneity | τ(X) = 0.01 for everyone | JSON + summary CSVs |
| **Telemedicine** | Sign change in τ(X) — beneficial for some, harmful for others | ATE ≈ 0, σ(τ) large | JSON + long-format metrics + sign-recovery plot |

## Project structure

```
Rlearner/
├── rlearner/                      # rlearner package source (used as local install if CRAN version unavailable)
├── Rlearner.Rproj                 # RStudio project file — open this to set working directory
├── README.md
├── Latest Phase/
│   ├── Smoking and Longevity/
│   │   ├── code/
│   │   │   ├── 1_smoking_and_longevity.R               # single-iter exploratory (rlasso, rboost only)
│   │   │   ├── 2_smoking_and_longevity_all.R           # single-iter, all 14 learners, n ∈ {300, 500}
│   │   │   ├── 3_smoking_and_longevity_multi_iter.R    # 50 iterations, n = 500, parallel
│   │   │   └── 4_smoking_and_longevity_summary_stats.ipynb  # DGP summary table (Python)
│   │   └── results/
│   ├── Preventative Healthcare/
│   │   ├── code/
│   │   │   ├── 1_preventative.R                        # single-iter exploratory
│   │   │   ├── 2_preventative_all.R                    # single-iter, all learners
│   │   │   ├── 3_preventative_multi_iter_100.R         # pilot: 100 iterations
│   │   │   ├── 3_preventative_multi_iter_200.R         # primary: 200 iterations
│   │   │   ├── 3_preventative_multi_iter_summary_gen_100.R
│   │   │   └── 3_preventative_multi_iter_summary_gen_200.R
│   │   └── results/
│   └── Telemedicine/
│       ├── code/
│       │   ├── 1_telemed.R                             # single-iter exploratory
│       │   ├── 2_telemed_all.R                         # single-iter, all learners, n = 250, saves JSON
│       │   ├── 3_telemed_multi_iter.R                  # 200 iterations, n = 300
│       │   ├── 3_telemed_convert_json_mse.R            # adds raw_mse field to existing JSON (run once)
│       │   ├── 3_json_analysis.R                       # extracts long-format metrics CSVs from JSON
│       │   ├── 4_telemed_dgp.R                         # DGP-level diagnostics over 200 iterations
│       │   └── 4_telemed_plot_generation.R             # generates sign-recovery plot
│       └── results/
└── Old Phase/                     # earlier exploratory work — not needed for reproduction
```

## Requirements

- **R ≥ 4.3.0** (tested on 4.3.0 through 4.4.3)
- **Python ≥ 3.10** (only for the smoking summary-stats notebook)
- A machine with ≥ 4 cores recommended — multi-iteration scripts run in parallel via `furrr`

### R packages

From CRAN:

```r
install.packages(c(
  "MASS", "devtools", "jsonlite", "rjson",
  "future", "furrr", "tidyverse", "dplyr", "tidyr",
  "ggplot2", "scales"
))
```

For `rlearner` — install from GitHub (not on CRAN):

```r
devtools::install_github("xnie/rlearner")
```

### Installing KRLS2 (kernel base method) — platform-specific

**Windows:** CRAN install should work directly:

```r
install.packages("KRLS2")
```

**macOS:** `install.packages("KRLS2")` fails on recent builds. The package needs a Fortran compiler that matches the CRAN R binary. Workaround for CRAN R Big Sur builds (R 4.3.0 through 4.4.3):

1. Go to <https://mac.r-project.org/tools/> and download **`gfortran-12.2-universal.pkg`**.
2. Install it, then confirm in a terminal:
   ```bash
   gfortran --version
   ```
3. Restart RStudio.
4. Install KRLS2 from GitHub:
   ```r
   devtools::install_github("xnie/KRLS")
   library(KRLS2)
   ```

> This workaround is specific to CRAN R Big Sur builds used with `gfortran-12.2-universal.pkg` for R 4.3.0 up to 4.4.3. Other macOS R distributions may require a different gfortran version.

### Python packages (smoking summary notebook only)

```bash
pip install pandas numpy scipy jupyter
```

## Reproducing the results

### One-time setup

1. Clone / download the project and open `Rlearner.Rproj` in RStudio. This sets the working directory to the project root (`~/DSE4231/Rlearner` or wherever you place it). **All R paths in the scripts are relative to this root** — do not `source()` individual scripts from inside their `code/` subfolders without first setting the working directory to the project root.
2. Install the R packages listed above (including the KRLS2 workaround if on macOS).
3. Confirm the directory structure matches the layout shown above.

### Pipeline convention

Each case study follows the same numbered-prefix convention. Run in order:

- **`1_*.R`** — single-iteration exploratory run with 2 learners (`rlasso`, `rboost`). Quick sanity check on the DGP and produces one CSV + one propensity histogram.
- **`2_*.R`** — single-iteration run with the full learner panel (all 12 meta-learner × base-method combinations plus baselines).
- **`3_*.R`** — multi-iteration parallel simulation. Uses `future::multisession` with `availableCores() - 1` workers. Produces the primary results used in downstream analysis.
- **`4_*.R` / `.ipynb`** — post-processing: summary stats, DGP diagnostics, plot generation.

### Study 1: Smoking and Longevity

Working directory: project root.

```r
# Single-iteration exploratory (n = 500, rlasso + rboost)
source("Latest Phase/Smoking and Longevity/code/1_smoking_and_longevity.R")

# Full learner panel — change n in the script to 300 or 500
source("Latest Phase/Smoking and Longevity/code/2_smoking_and_longevity_all.R")

# Multi-iteration: 50 iterations at n = 500 (primary result)
source("Latest Phase/Smoking and Longevity/code/3_smoking_and_longevity_multi_iter.R")
```

Outputs (written to `Latest Phase/Smoking and Longevity/results/`):

- `1_smoking_500.csv`, `1_propensity_hist_500.png`
- `2_smoking_all_{300,500}.csv`, `2_propensity_hist_all_{300,500}.png`
- `3_smoking_multi_iter_n500_50iters.csv` — one row per iteration with per-learner metrics
- `3_smoking_multi_iter_summary_n500_50iters.csv` — aggregated across iterations

DGP summary table (Python, from the Jupyter notebook):

```bash
cd "Latest Phase/Smoking and Longevity/code"
jupyter notebook 4_smoking_and_longevity_summary_stats.ipynb
# Run all cells — expects ../results/3_smoking_multi_iter_n500_50iters.csv
```

### Study 2: Preventative Healthcare

```r
# Single-iteration exploratory
source("Latest Phase/Preventative Healthcare/code/1_preventative.R")

# Full learner panel, single iteration
source("Latest Phase/Preventative Healthcare/code/2_preventative_all.R")

# Multi-iteration — 200 is the primary result, 100 is the pilot
source("Latest Phase/Preventative Healthcare/code/3_preventative_multi_iter_200.R")

# Post-process JSON into summary CSVs
# Summary scripts expect the JSON to sit in the same directory they're run from —
# either setwd() to the results folder first, or move the JSON there.
source("Latest Phase/Preventative Healthcare/code/3_preventative_multi_iter_summary_gen_200.R")
```

Primary outputs (in `Latest Phase/Preventative Healthcare/results/`):

- `3_preventative_multi_iter_200iters.json` — full per-iteration results
- `3_preventive_multi_iter_summary_200iters.csv` — aggregated summary table

The 100-iteration version (`3_preventative_multi_iter_100.R` + `_summary_gen_100.R`) is a pilot used to validate the pipeline before the larger run. Running it is optional; outputs are `3_preventative_multi_iter_100iters.json` and `3_preventative_multi_iter_summary_100iters.csv`.

### Study 3: Telemedicine

```r
# Single-iteration exploratory
source("Latest Phase/Telemedicine/code/1_telemed.R")

# Full learner panel, n = 250, saves JSON
source("Latest Phase/Telemedicine/code/2_telemed_all.R")

# Multi-iteration: 200 iterations at n = 300 (primary)
source("Latest Phase/Telemedicine/code/3_telemed_multi_iter.R")

# One-time: back-fill raw_mse field in the JSON (only needed if running on
# old JSON outputs that predate the raw-MSE change; new runs already include it)
source("Latest Phase/Telemedicine/code/3_telemed_convert_json_mse.R")

# Extract long-format metrics CSVs from JSON
source("Latest Phase/Telemedicine/code/3_json_analysis.R")

# DGP-level diagnostics (tau distribution, ATT/ATC, SNR, overlap)
source("Latest Phase/Telemedicine/code/4_telemed_dgp.R")

# Final plot (sign recovery by subgroup)
source("Latest Phase/Telemedicine/code/4_telemed_plot_generation.R")
```

The 100-iteration JSON (`3_telemed_multi_iter_n300_100iters.json`) is from a pilot run and is not required for the main results.

Primary outputs (in `Latest Phase/Telemedicine/results/`):

- `2_telemed_all_250.json` — single-iteration results with all learners and subgroup truth
- `3_telemed_multi_iter_n300_200iters.json` — full per-iteration results
- `3_telemed_multi_iter_metrics_long.csv` — tidy long-format for ggplot
- `3_telemed_multi_iter_dgp_summary.csv` — DGP-level summary (one row)
- `3_telemed_table_ready.csv` — formatted comparison table
- `4_telemed_dgp_iter_diagnostics.csv` — per-iteration DGP diagnostics
- `3_sign_recovery_plot.png` — final sign-recovery figure

## Runtime notes

Multi-iteration scripts (`3_*.R`) use parallel backends and can take a while depending on hardware. Approximate wall-clock times on a machine with 8 physical cores:

| Script | Iterations | Approx. runtime |
|---|---|---|
| `3_smoking_and_longevity_multi_iter.R` | 50 | 15–30 min |
| `3_preventative_multi_iter_200.R` | 200 | 45–90 min |
| `3_telemed_multi_iter.R` | 200 | 60–120 min |

The kernel base methods (`*kern`) and the T-learner with boosting (`tboost`) dominate runtime. Reducing `ntrees_max` and `num_search_rounds` in the `boost_args` list at the top of each script trades accuracy for speed if you need a quicker sanity-check run.

## Seeds and reproducibility

All scripts set `set.seed(42)` (telemedicine, smoking) or `set.seed(1)` (preventative) at the top. Multi-iteration scripts use a derived per-iteration seed (`base_seed + i`) to ensure each iteration is reproducible independently. Parallel execution via `future::multisession` is seed-stable given the same worker count, but exact numerical agreement across different core counts is not guaranteed for the boosting learners; the aggregate summary statistics are robust to this.

## Troubleshooting

- **`could not find function "<learner>"` from `rlearner`** — confirm you installed from GitHub (`xnie/rlearner`), not CRAN.
- **`KRLS2` not found on macOS** — follow the gfortran workaround above. `install.packages("KRLS2")` alone will not work on recent macOS R builds.
- **Summary-gen scripts can't find the JSON** — the summary scripts read the JSON from the current working directory, not from `results/`. Either `setwd()` into the results folder before sourcing, or move the JSON file.
- **Parallel workers hang** — check that each worker has the required packages installed. The `3_*.R` scripts call `library()` inside `run_one_iteration` for this reason; if you modify them, keep those inner calls.
