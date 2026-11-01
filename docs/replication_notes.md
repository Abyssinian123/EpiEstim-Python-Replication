# Replication Notes

Findings about the original R implementation, discovered while building
the Python replication. These are all handled (reproduced or worked
around) by the code in `src/epiestim/` and
`experiments/replication/`.

## The serial-interval variant (`test_si_variant.py`)

### The serial-interval variant (`test_si_variant.py`)

Inside `estimate_R_agg`, the R\* → r inversion (`get_r_from_R`) uses a
serial interval that differs between EpiEstim builds:

- the build that produced the paper's stored results uses the **centred
  discretisation** `d(k) = F(k+0.5) − F(k−0.5)` of the unshifted Gamma over
  support 0..1000 (R `distcrete("gamma", w = 0.5)` + `epitrix::r2R0`);
- current EpiEstim GitHub main uses `discr_si` truncated at 1e-6.

The two agree exactly for declining epidemics (R\* < 1) but differ for
rising ones, where the link is most sensitive. `estimate_R_agg()` defaults to
`si_variant = "distcrete"` (the paper's results; see docs/CONVENTIONS.md §3).

### R-script bugs and quirks (all handled here)

- **weekend_effects.R incidence panels**: `result[[4]]$I` is the daily
  simulation input, not an EM reconstruction, so the darkgreen
  "Reconstructed data" line in the paper's Figure S7 incidence row is
  actually the simulated daily series plotted a second time (the two lines
  coincide visually only because `add_weekend_effect` is applied to the
  aggregate). Decoded by fitting the affine PDF→data map
  (`decode_weekend_pdf.py`); the figure layout is columns constant / sudden
  / gradual, rows incidence / non-sliding / weekly sliding / two-week
  sliding, with the darkgrey reference filtered to
  `t_end >= min(agg t_end)` and the black dashed line the paper's idealised
  true-R step at t_end 1..70.
- **gdata::combine() name mapping**: with 3 data frames and explicit names,
  `names[1]` labels the 2nd argument, `names[2]` the 1st, `names[3]` the
  3rd; with 2 arguments the mapping is in order; with no names the source
  column is the deparsed argument name. (Determines which ggplot colour
  applies to which series in the R figures.)
- **loess_smoothing.R:15** reads `R_agg_sliding_result.rds` (capital R)
  while `main_analysis/flu.R:55` saves `r_agg_sliding_result.rds` — works
  only on case-insensitive filesystems.
- **zika.R** hardcodes the NA prefixes `rep(NA, 14)` / `rep(NA, 20)` for the
  two series; correct only because the window counts work out (the old
  2-week windows start at t_end 3, the EM weekly windows at t_end 21 after
  the `t_end > mean_si = 20` drop).
- **covid_cases.R:157 / covid_deaths.R:160** pair `r_agg` with
  `r_rep[3:length(r_rep$R$t_end)]` for the `lm()`; the lengths coincide
  (671 vs 671), but the intended pairing differs from the one used in the
  paper's scatter panels (flu pairs agg t_end 9..35 with rep t_end 8..34 via
  `[6:32]`). The classification-agreement percentages quoted in the paper
  text (74.4% covid cases daily; 81.8/44.4% flu; 94.9/85.8/93.3%) were
  computed by code **not included in the repository** — the aligned
  recomputation here reproduces all of them except covid cases daily, which
  gives 74.5% (a one-window difference).
- **covid_deaths_appendix.R** copies `coord_cartesian(xlim = c(5, 679))`
  from the cases appendix although the deaths series has 672 days.
- The covid deaths dataset has **672 days** (2020-03-02 .. 2022-01-02), not
  679 as some copied constants assume.

### `estimate_R_agg` window semantics

- Windows for R\* are the weekly aggregation windows starting on day 8.
- Output windows of length `dt_out` slide **daily** from `t_start = 8`
  (so the first t_end is 8 + dt_out − 1: day 9 for `dt_out = 2`, day 14 for
  `dt_out = 7`), then windows with `t_end <= mean_si` are dropped (raw mean,
  the paper's EpiEstim build). This is why the zika EM starts at t_end 21.
- `estimate_R` defaults to `t_start = seq(2, T-6)`, `t_end = seq(8, T)` and
  applies the same `t_end > mean_si` drop rule to custom windows.

### Simulations

`simulate_incidence` was verified against the CRAN source of
`projections::project` (`R_fix_within = FALSE`, `instantaneous_R = TRUE`):
7 seed days of 10 cases, day d's incidence ~ Poisson(R_t[d−1] · renewal),
change points advance the R value **after** the change row, and
`time_change` is offset by the seed days. The RNG differs by necessity
(R's Mersenne-Twister streams are not reproducible cross-language): the
convention (docs/CONVENTIONS.md §1) is `np.random.default_rng(6)` for constant
and `default_rng(5)` for time-varying scenarios, so individual trajectories
differ from the stored R runs while the generating process, parameters and
all distributional summaries (coverage, mean ribbons) match the paper's
reported ranges.

## Known approximations (documented, intentional)

- R's `signif(x, 5)` rounding is applied wherever the R scripts applied it.
- ggplot2 `theme_ipsum()` typography (font, sizes) is approximated by the
  shared matplotlib style in `sim_study.py`; all geometry, limits, colours,
  breaks and labels follow the R calls exactly.
- Jitter offsets in `incidence_weekday.pdf` use seed 5 but a different RNG
  (the crossbars — the content of the figure — are exact).
- The paper's plotted "true R" lines are step idealisations (see
  docs/CONVENTIONS.md §2); the studies reproduce the idealisations, not the
  per-window ground truth.
