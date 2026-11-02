# EpiEstim: Python Replication and Extension

This repository contains a Python replication and extension of the EM-based method for estimating time-varying epidemic reproduction numbers (\(R_t\)) from temporally aggregated incidence data, proposed by Nash et al. (2023).

## Project Objectives

1. **Replication:** Reimplement the original EM-based incidence reconstruction and reproduction number estimation method in Python.
2. **Validation:** Reproduce and evaluate the main results of the original study using simulated and real-world epidemic data.
3. **Extension:** Investigate the robustness of the method under different case reporting mechanisms, particularly time-varying reporting rates.

## Original Research

Nash, R. K., Bhatt, S., Cori, A., & Nouvellet, P. (2023). Estimating the epidemic reproduction number from temporally aggregated incidence data: A statistical modelling approach and software tool. *PLOS Computational Biology, 19*(8), e1011439.

**Paper:** https://doi.org/10.1371/journal.pcbi.1011439

**Original analysis repository:** https://github.com/rebeccanash/EM_EpiEstim_Nash2023

**Original EpiEstim R package:** https://github.com/mrc-ide/EpiEstim

**Archived original code:** https://doi.org/10.5281/zenodo.8091058

## Project Status

This is an independent educational replication and extension project for a Stochastic Processes course. It is not an official Python implementation of EpiEstim.

## Contributors

Three undergraduate students working collaboratively on model replication, simulation experiments, and statistical evaluation.

## Repository Structure

```
├── src/epiestim/              # Core engine: estimate_R, estimate_R_agg (EM),
│                              #   aggregate_inc, discr_si, simulate_incidence,
│                              #   loess.py (port of R's loess), rds_reader.py,
│                              #   reporting.py (Binomial thinning + weekly aggregation)
├── experiments/replication/   # Full replication of the paper
│   ├── reproduce_paper.py     #   Figures 2-4 (flu / covid cases / covid deaths)
│   ├── sim_study.py           #   All 10 S1 Appendix studies + supplementary analyses
│   ├── validate_vs_R.py       #   engine vs the paper's stored RDS results (≤ 1e-14)
│   ├── validate_loess_vs_pdf.py  # LOESS pipeline vs the paper's PDF vector curves
│   └── test_si_variant.py     #   serial-interval discretisation variants
├── experiments/extension/     # Reporting-model robustness simulation (Nash 2023)
│   ├── run_reporting_nash2023_pilot.py    #   small-scale pilot (10 trajectories)
│   ├── run_reporting_nash2023_formal.py   #   100-trajectory formal run (S0–S3)
│   ├── run_reporting_experiments.py       #   per-scenario entry point
│   └── constant_reporting.py / sudden_change.py / gradual_change.py
├── experiments/analysis/      # Statistical evaluation (bias / RMSE / coverage)
│   ├── analyze_reporting.py   #   figures + tables for S0–S3
│   ├── stat_metrics.py        #   shared metric + summary helpers
│   └── robustness_constant_p.py  # constant-underreporting sweep (20/40/60%)
├── EM_EpiEstim_Nash2023-main/ # Original R implementation (data + stored results
│                              #   used as the validation benchmark; trimmed copy —
│                              #   the full archive is at the Zenodo DOI below)
├── data/raw/                  # Daily incidence datasets (flu, covid cases, deaths)
├── results/figures/           # Generated figures (committed as reference)
├── results/reporting_nash2023/  # Reporting-extension outputs
│   ├── formal/                #   S0–S3 estimates, metadata, check report
│   ├── constant_p/            #   constant-underreporting robustness sweep
│   └── analysis/              #   analysis_report.md
├── results/tables/            # reporting_summary.csv
├── docs/
│   ├── CONVENTIONS.md         # ⚠ group-wide parameter conventions (seeds, time
│   │                          #   indexing, SI variants, array shapes, metrics)
│   ├── replication_notes.md   # bugs & semantics discovered in the original code
│   ├── reporting_conventions_diff.md  # reporting-extension conventions (scenarios, seeds)
│   ├── reporting_results_schema.md    # result CSV schema
│   ├── reporting_robustness_theory.md # renewal-process theory of underreporting
│   ├── nash2023_engine_provenance.md  # engine provenance vs A's commit
│   └── 说明.md                # data & code provenance (Chinese)
└── tests/                     # test_reporting.py (reporting model); notebooks/ reserved
```

## Reporting-Model Extension

The extension studies how robust the EM estimator is when reported case counts are thinned by a time-varying reporting probability: `Y_t ~ Binomial(I_t, p_t)` with `I_t ~ Poisson(R_t Λ_t)`.

- **Model** (`src/epiestim/reporting.py`): Binomial thinning + aligned weekly aggregation. Four scenarios — S0 (perfect, p=1), S1 (constant p=0.8), S2 (sudden 0.8→0.4 at day 35), S3 (gradual 0.8→0.4 over days 35–56).
- **Simulation** (`experiments/extension/`): a 10-trajectory pilot and a 100-trajectory formal run with fixed seeds and exact reproducibility checks; outputs land in `results/reporting_nash2023/formal/`.
- **Analysis** (`experiments/analysis/`): bias / RMSE / 95% credible-interval coverage of daily-reported vs weekly-EM estimates (`analyze_reporting.py`), plus a constant-underreporting robustness sweep at 20/40/60% (`robustness_constant_p.py`).
- **Theory** (`docs/reporting_robustness_theory.md`): constant thinning preserves the epidemic growth rate, so R_t stays unbiased; only *time-varying* p_t biases it — which is exactly what S2/S3 vs S1 confirm numerically.

Run the extension:

```bash
cd experiments/extension
python run_reporting_nash2023_formal.py      # 100-trajectory formal run (S0–S3)
python run_reporting_experiments.py          # per-scenario entry point

cd ../analysis
python analyze_reporting.py                  # bias / coverage figures + tables
python robustness_constant_p.py              # constant-underreporting sweep
```

## Quick Start

```bash
pip install -r requirements.txt      # numpy, pandas, scipy, matplotlib, pyreadr
                                     # (no R installation needed)

cd experiments/replication
python reproduce_paper.py            # main paper, Figures 2-4
python sim_study.py                  # all supplementary studies
python sim_study.py loess            # or any subset: constant_rt, tv_sudden,
                                     #   tv_gradual, weekend_effects, diff_aggs,
                                     #   mid_agg, loess, weekday, zika, appendix_rt
```

Outputs go to `results/figures/`. The per-simulation EM runs are cached in
`experiments/replication/_est_cache/` (not committed; delete to force
re-estimation — a full fresh run takes ~1-2 h).

## Validation

- `python validate_vs_R.py` — every stored RDS result of the paper's R
  pipeline is reproduced to ≤ 1e-14.
- `python validate_loess_vs_pdf.py` — the LOESS port matches the curves and
  CrI ribbons in the stored `loess_rt.pdf` to < 0.005 PDF points.
- Simulation studies reproduce the paper's reported coverage ranges exactly
  (e.g. sudden change: 95.0/90.4/87.9% vs [87-96%]).

## For Contributors

All group members: read [docs/CONVENTIONS.md](docs/CONVENTIONS.md) before
writing code that extends the replication — every parameter convention in it
is fixed unless the whole group agrees to change it. The reporting extension
adds its own conventions in [docs/reporting_conventions_diff.md](docs/reporting_conventions_diff.md).
See [docs/replication_notes.md](docs/replication_notes.md) for the known bugs in
the original R scripts and the EpiEstim version differences that the
replication handles.
