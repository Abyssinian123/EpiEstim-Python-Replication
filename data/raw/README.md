# Raw data

Daily incidence series used in the paper, copied from
`EM_EpiEstim_Nash2023-main/real_data/` (originally from the paper's public
analysis repository and its Zenodo archive, see `docs/说明.md`).

| File | Content | Days |
|---|---|---|
| `daily_flu.rds` | 2009 H1N1 influenza A/H1N1 pandemic, U.S. (2009-12-11 .. 2010-01-14) | 35 |
| `daily_covid_cases.rds` | COVID-19 cases, U.K. (2020-02-21 .. 2021-12-31) | 679 |
| `daily_covid_deaths.rds` | COVID-19 deaths, U.K. (2020-03-02 .. 2022-01-02) | 672 |

RDS files are read with `pyreadr` (no R installation needed).
