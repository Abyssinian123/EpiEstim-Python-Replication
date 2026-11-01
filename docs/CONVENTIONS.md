# Group Project Conventions

Shared parameter and data conventions for all team members building on the
Python reproduction of Nash et al. (2023), PLoS Comput Biol 19(8): e1011439,
"Estimating the epidemic reproduction number from temporally aggregated
incidence data".

> ⚠ All quantities below are fixed. Any deviation must be agreed by the whole
> group and recorded here.

## 1. Random seeds

| Scenario                        | Seed |
|---------------------------------|------|
| Constant Rt simulations         | 6    |
| Time-varying Rt (sudden/gradual)| 5    |

In Python use `np.random.default_rng(seed)` (R's `set.seed()` streams are
generator-specific and not reproducible across languages; the convention is
that **all team members use the same Python generator and seed** so that
everyone's simulations are mutually comparable).

## 2. Time indexing

- **1-based daily index**: day 1 = first day of observed incidence.
- `t_start` / `t_end` in every output are **1-based inclusive** day indices.
- Default (reported-data) estimation windows: `t_start = 2..(T-6)`,
  `t_end = 8..T` (7-day sliding windows). The main analyses additionally use
  custom 2-day windows `t_start = 2..(T-1)`, `t_end = t_start + 1`.
- **Drop rule** (paper's EpiEstim build): any window with
  `t_end <= mean_si` (the raw mean, not the truncated SI mean) is dropped
  from the output entirely. With the default windows nothing is dropped for
  the paper's SI parameters; with the custom 2-day windows this is why the
  flu daily results start at t_end = 4 and the covid ones at t_end = 7.
- Aggregation windows are non-overlapping, aligned to day 1:
  window j = days `[(j-1)*dt + 1 .. j*dt]`. The incidence length T must be an
  exact multiple of dt (no remainder days are kept).
- Simulation timeline: `n_days = 70`, seeded with **7 days of 10 cases**,
  `n_sim = 100`. For a sudden change, `time_change = 34` means days 1-35 use
  the old Rt and days 36-70 the new Rt (R convention: `time_change` = last
  day of the old Rt). For a gradual change with `time_change = seq(19, 49)`
  and a length-32 ramp R: days 1-20 use R[0], days 21-50 use R[1..30],
  days 51-70 use R[31] (the 32-value ramp covers days 20-51; the endpoints
  hold the rest).
- The paper plots its "true R" lines as step idealisations over the day
  ranges above (e.g. `rep(a, 35), rep(b, 35)` for a sudden change at day 35
  and weekly-repeat patterns for the mid-aggregation study); the fine
  semantics above are what the simulations actually use.

## 3. Serial interval (SI)

⚠ **Two different SI discretisations appear in the pipeline — do not mix
them up.**

**A. Posterior / infectivity SI (`discr_si`)** — parametric **shifted Gamma
discretisation** `discr_si(k)` for k = 0, 1, 2, ... (Cori et al. 2013,
`EpiEstim::discr_si`; support starts at day 0, so w_0 ≈ 0). Used by
`estimate_R()` for the posterior / overall infectivity.

**B. Growth-rate link SI (`distcrete` gamma, w = 0.5)** — inside the EM, the
R* → r inversion (`get_r_from_R`) uses the **centred discretisation**
`d(k) = F(k+0.5) − F(k−0.5)` of the **unshifted** Gamma(shape = 1/cv²,
scale = μ·cv²) over k = 0..1000 (R `distcrete("gamma", w = 0.5)` +
`epitrix::r2R0`, no truncation). This is what the EpiEstim build that
generated the paper's stored results used (see `vignette/functions.R` in the
reproduction repo) and reproduces them to 1e-10. In Python it is
`si_variant = "distcrete"`, the default of `estimate_R_agg()`.
`si_variant = "discr_si"` instead reproduces current EpiEstim GitHub main
(`discr_si` truncated at 1e-6); the two agree exactly for declining
epidemics (R* < 1) but differ for rising ones, so **all team members must
use the default `"distcrete"`**.

- **Influenza** (all simulations + flu dataset): `mean_si = 3.6`, `sd_si = 1.6`.
- **COVID-19** (cases and deaths datasets): `mean_si = 6.3`, `sd_si = 4.2`.
- Simulation SI: `discr_si(0:30)`, normalised, then drop w_0
  (`si_one` = lags 1..30).
- Estimation SI (posterior): `discr_si(0..T-1)` over the full observation period.

## 4. Input array shapes

- Daily incidence: **1-D float array, shape (T,)**.
- Aggregated incidence: **1-D float array, shape (n_dt,)**, with
  `T = n_dt * dt` (flu: T = 35, n_dt = 5 at dt = 7).
- `simulate_incidence(Rt, si_one, n_days=70, n_sim=100)` returns
  **shape (n_sim, n_days) = (100, 70)**.
- Rt input: scalar (constant), 2 values (sudden change), or 32 values
  (gradual ramp).

## 5. Output fields

`estimate_R()` returns a DataFrame with exactly these columns (same as
EpiEstim's R output):

```
t_start, t_end, Mean(R), Std(R),
Quantile.0.025(R), Quantile.0.05(R), Quantile.0.25(R), Median(R),
Quantile.0.75(R), Quantile.0.95(R), Quantile.0.975(R)
```

`estimate_R_agg()` returns a dict:

| Key         | Meaning                                    | Shape      |
|-------------|--------------------------------------------|------------|
| `R`         | final Rt estimates (same 11 columns)       | (W, 11)    |
| `I`         | reconstructed daily incidence              | (T,)       |
| `I_local`   | local part (I with day 1 moved out)        | (T,)       |
| `I_imported`| imported part (day 1 only, by convention)  | (T,)       |
| `iterations`| per-EM-iteration reconstructions           | (T, iter)  |
| `si_distr`  | SI pmf used                                 | (T,)       |

## 6. Other fixed settings

| Setting                 | Value                                   |
|-------------------------|-----------------------------------------|
| Prior on R              | Gamma(mean = 5, sd = 5)                 |
| Estimation method       | `parametric_si` (analytic posterior)    |
| Quantiles               | 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975 |
| EM iterations           | `iter = 10`                             |
| EM tolerance            | `tol = 1e-6`                            |
| Reconstruction option   | `recon_opt = "naive"` or `"match"`      |
| Growth-rate grid        | precision 0.001, r ∈ [-1, 1], widened ×5 when any R* hits an edge |
| Aggregation             | `dt = 7` (weekly); output `dt_out = 7` (sliding weekly) or `dt_out = 2` |
| First-day incidence     | treated as imported (`I_local[0] = 0`)  |
| Trend classification    | increasing if `Q2.5 > 1`; declining if `Q97.5 < 1`; else uncertain |
| Weekend effect (S1 App.)| **80%** of Sat/Sun cases moved to the following Mon/Tue |
| Simulation SI           | `mean_si = 3.6, sd_si = 1.6`            |

## 7. Evaluation metrics (used in every comparison)

- R² and correlation between **reported-data** and **reconstructed-data**
  Rt estimates (Mean(R)).
- R² / correlation of **95% CrI widths** (Q97.5 − Q2.5), reported vs
  reconstructed.
- Relative squared error of reconstructed vs true daily incidence
  (simulations).
- Trend classification accuracy (increasing / declining / uncertain).
