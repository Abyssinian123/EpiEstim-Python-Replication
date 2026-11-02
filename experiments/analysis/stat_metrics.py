# -*- coding: utf-8 -*-
"""Statistical-evaluation metrics for the reporting-rate extension (Member C).

These consume the long-form ``estimates_all.csv`` produced by Member B's
formal runner and compute, per window / scenario / estimation method, the
Bias, RMSE and 95% credible-interval coverage of the Rt estimates, plus a
trend-classification check. The true Rt is constant at 1.25 in the formal
batch, so every window's truth is 1.25 (see docs/reporting_results_schema.md).

The one thing to *not* read into these numbers: ``t_end`` is the end of the
sliding estimation window, not the information cutoff of the observation used.
The weekly EM reconstructs a whole 7-day aggregation window retrospectively,
so a window whose ``t_end`` falls *before* a reporting-rate change can already
be influenced by later days inside that same week. Any early deflection around
``t_end = 34`` in S2/S3 is that retrospective leakage, not a time-index error
and not predictive ability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_RT = 1.25

R_MEAN = "Mean(R)"
R_LOW = "Quantile.0.025(R)"
R_HIGH = "Quantile.0.975(R)"


def add_error_columns(estimates: pd.DataFrame, true_col: str = "true_rt") -> pd.DataFrame:
    """Attach per-row error columns to a copy of the estimates frame.

    Columns added: ``bias``, ``sq_err``, ``covered`` (95% CrI), ``cr_width``,
    ``wrong_declining`` (true Rt = 1.25 is "increasing", so a window is misread
    as "declining" when its upper 95% bound is below 1).
    """
    df = estimates.copy()
    df["bias"] = df[R_MEAN] - df[true_col]
    df["sq_err"] = df["bias"] ** 2
    df["covered"] = (df[R_LOW] <= df[true_col]) & (df[true_col] <= df[R_HIGH])
    df["cr_width"] = df[R_HIGH] - df[R_LOW]
    df["wrong_declining"] = df[R_HIGH] < 1.0
    return df


def rmse(errors) -> float:
    """Root-mean-square error of an array of signed errors."""
    errors = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.nanmean(np.square(errors))))


def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Group a frame carrying the error columns by scenario and estimation
    method, returning mean bias, RMSE, coverage and wrong-declining rate."""
    return df.groupby(["reporting_scenario", "estimation_method"]).agg(
        n=("bias", "size"),
        mean_bias=("bias", "mean"),
        rmse=("bias", lambda s: rmse(s)),
        coverage=("covered", "mean"),
        wrong_declining=("wrong_declining", "mean"),
        mean_cr_width=("cr_width", "mean"),
    ).reset_index()


def paired_difference(errors: pd.DataFrame, scenario_from: str,
                      scenario_ref: str = "S1") -> pd.DataFrame:
    """Per-window paired difference of a metric between two scenarios.

    Both scenarios share the same true-incidence matrix, so rows are paired on
    ``simulation_id`` (== ``paired_truth_id``) and ``t_end``. The returned frame
    has, for each ``t_end``, the mean paired difference and its SE and 95% CI
    (normal approximation over the 100 trajectories).

    ``errors`` must be the wide/long frame with an ``error`` column (e.g. bias,
    or a mean estimate) and columns ``simulation_id``, ``t_end``,
    ``estimation_method``, ``reporting_scenario``.
    """
    def _pivot(scenario):
        sub = errors[errors["reporting_scenario"] == scenario]
        return sub.pivot_table(index=["simulation_id", "t_end"],
                               columns="estimation_method",
                               values="error")

    base = _pivot(scenario_ref)
    comp = _pivot(scenario_from)
    diff = comp - base

    rows = []
    for method in diff.columns:
        d = diff[method].groupby("t_end")
        n = d.count()
        mean = d.mean()
        se = d.std(ddof=1) / np.sqrt(n)
        rows.append(pd.DataFrame({
            "t_end": mean.index.to_numpy(),
            "estimation_method": method,
            "paired_diff_mean": mean.to_numpy(),
            "paired_diff_se": se.to_numpy(),
            "paired_diff_ci_lo": mean.to_numpy() - 1.96 * se.to_numpy(),
            "paired_diff_ci_hi": mean.to_numpy() + 1.96 * se.to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)
