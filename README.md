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
│                              #   loess.py (port of R's loess), rds_reader.py
├── experiments/replication/   # Full replication of the paper
│   ├── reproduce_paper.py     #   Figures 2-4 (flu / covid cases / covid deaths)
│   ├── sim_study.py           #   All 10 S1 Appendix studies + supplementary analyses
│   ├── validate_vs_R.py       #   engine vs the paper's stored RDS results (≤ 1e-14)
│   ├── validate_loess_vs_pdf.py  # LOESS pipeline vs the paper's PDF vector curves
│   └── test_si_variant.py     #   serial-interval discretisation variants
├── EM_EpiEstim_Nash2023-main/ # Original R implementation (data + stored results
│                              #   used as the validation benchmark; trimmed copy —
│                              #   the full archive is at the Zenodo DOI below)
├── data/raw/                  # Daily incidence datasets (flu, covid cases, deaths)
├── results/figures/           # Generated figures (25 PDFs, committed as reference)
├── docs/
│   ├── CONVENTIONS.md         # ⚠ group-wide parameter conventions (seeds, time
│   │                          #   indexing, SI variants, array shapes, metrics)
│   ├── replication_notes.md   # bugs & semantics discovered in the original code
│   └── 说明.md                # data & code provenance (Chinese)
└── tests/ notebooks/          # reserved for the extension work
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
is fixed unless the whole group agrees to change it. See
[docs/replication_notes.md](docs/replication_notes.md) for the known bugs in
the original R scripts and the EpiEstim version differences that the
replication handles.
