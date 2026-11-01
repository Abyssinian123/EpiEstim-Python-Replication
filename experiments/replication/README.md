# Replication — Nash et al. (2023)

Complete Python replication of

> Nash, R. K., Bhatt, S., Cori, A., & Nouvellet, P. (2023). Estimating the
> epidemic reproduction number from temporally aggregated incidence data.
> *PLoS Computational Biology*, 19(8): e1011439.

covering the three main-paper figures (Figures 2-4), all simulation studies
of the S1 Appendix, and the remaining supplementary analyses (LOESS, weekday
reporting, zika workaround, flu/covid appendix R-estimate figures). Every
figure mirrors the R scripts in `../../EM_EpiEstim_Nash2023-main/` and is
written to `../../results/figures/*.pdf`. Group-wide parameter conventions
are in [`../../docs/CONVENTIONS.md`](../../docs/CONVENTIONS.md); findings
about the original implementation are in
[`../../docs/replication_notes.md`](../../docs/replication_notes.md).

## Run

```bash
# from the repository root, once:
pip install -r requirements.txt     # numpy, pandas, scipy, matplotlib, pyreadr (no R)

cd experiments/replication
python reproduce_paper.py            # main paper, Figures 2-4
python sim_study.py                  # all supplementary studies
python sim_study.py tv_sudden        # or any subset, names from --help output:
                                     # constant_rt, tv_sudden, tv_gradual,
                                     # weekend_effects, diff_aggs, mid_agg,
                                     # loess, weekday, zika, appendix_rt
```

The slow part is the per-simulation EM; results are cached in
`experiments/replication/_est_cache/` (not committed — delete the directory
to force re-estimation, a full fresh run takes ~1-2 h).

## Layout

| File | Purpose |
|---|---|
| `reproduce_paper.py` | Figures 2-4 (flu / covid cases / covid deaths) |
| `sim_study.py` | All S1 Appendix simulation studies + the remaining supplementary analyses |
| `validate_vs_R.py` | Engine validation against the paper's stored RDS results (1e-14) |
| `validate_loess_vs_pdf.py` | LOESS pipeline validation against the stored `loess_rt.pdf` vector curves |
| `decode_weekend_pdf.py` | Decodes the paper's `weekend_effects.pdf` (see replication notes) |
| `extract_loess_pdf.py` | Extracts the plotted curves from `loess_rt.pdf` (helper) |
| `test_si_variant.py` | Documents the serial-interval variant discovery |

The core engine lives in `../../src/epiestim/` (`epiestim.py`, `loess.py`,
`rds_reader.py`).

## Validation

1. **Estimation engine, 1e-14 against R** (`validate_vs_R.py`): every RDS
   result stored by the paper's R pipeline (`r_rep_result.rds`,
   `r_agg_result.rds`, ... for flu, covid cases and deaths) is reproduced to
   ≤ 1e-14 (the R scripts' `signif(..., 5)` rounding).
2. **LOESS port against the published PDF** (`validate_loess_vs_pdf.py`):
   `loess.py` is a line-by-line port of R's `src/library/stats/src/loessf.f`
   (tricube weights, median-split k-d tree, equilibrated local polynomial
   fits with the 2^-52 singular-value cutoff, Hermite blending). The full
   pipeline (naive weekly incidence → LOESS degrees 0/1/2 → `estimate_R`)
   matches every curve and CrI ribbon in the stored `loess_rt.pdf` to
   < 0.005 PDF points (~3e-5 in R units).
3. **Simulation studies vs the paper's reported numbers**: e.g. the sudden
   change study reproduces the paper's true-R coverage ranges exactly
   (increasing: 95.0/90.4/87.9% vs [87-96%]; excluding windows around the
   change point: 94.5-97.3% vs [95-97%]); the gradual study gives
   77.6-95.8% (inc) / 61.5-93.6% (dec) vs [77-96%] / [61-94%]. The loess
   study's EM reconstruction matches the stored one to 5.9e-12.
