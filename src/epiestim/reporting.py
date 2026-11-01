"""Case-reporting observation model and aligned temporal aggregation."""

from __future__ import annotations

import numpy as np


def make_reporting_rates(scenario: str, n_days: int = 70, change_day: int = 35) -> np.ndarray:
    """Return daily reporting probabilities; day numbers in the API are 1-based."""
    if not isinstance(n_days, (int, np.integer)) or n_days < 1:
        raise ValueError("n_days must be a positive integer")
    if not isinstance(change_day, (int, np.integer)) or not 1 <= change_day <= n_days:
        raise ValueError("change_day must be a 1-based day within n_days")
    if scenario in ("S0", "perfect"):
        rates = np.ones(n_days)
    elif scenario in ("S1", "constant"):
        rates = np.full(n_days, 0.8)
    elif scenario in ("S2", "sudden"):
        rates = np.full(n_days, 0.8)
        rates[change_day - 1:] = 0.4
    elif scenario in ("S3", "gradual"):
        if n_days != 70:
            raise ValueError("S3 gradual scenario is defined for n_days=70")
        rates = np.full(n_days, 0.4)
        rates[:34] = 0.8
        rates[34:56] = np.linspace(0.8, 0.4, 22)
    else:
        raise ValueError("scenario must be S0/S1/S2/S3 or perfect/constant/sudden/gradual")
    return rates


def apply_reporting(true_incidence, reporting_rates, seed=None) -> np.ndarray:
    """Draw reported integer counts from Binomial(true incidence, p)."""
    incidence = np.asarray(true_incidence)
    if incidence.ndim not in (1, 2):
        raise ValueError("true_incidence must be one- or two-dimensional")
    if not np.issubdtype(incidence.dtype, np.number) or not np.all(np.isfinite(incidence)):
        raise ValueError("true_incidence must contain finite numeric counts")
    if np.any(incidence < 0) or np.any(incidence != np.floor(incidence)):
        raise ValueError("true_incidence must contain nonnegative integer counts")
    rates = np.asarray(reporting_rates, dtype=float)
    if rates.ndim == 0:
        rates = np.full(incidence.shape[-1], rates.item())
    if rates.ndim != 1 or rates.size != incidence.shape[-1]:
        raise ValueError("reporting_rates must be scalar or match the final incidence dimension")
    if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
        raise ValueError("reporting probabilities must be finite and between 0 and 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.binomial(incidence.astype(np.int64), rates)


def aggregate_incidence(daily_incidence, dt: int = 7) -> np.ndarray:
    """Sum aligned, non-overlapping windows along the final axis; reject remainder days."""
    values = np.asarray(daily_incidence)
    if values.ndim not in (1, 2):
        raise ValueError("daily_incidence must be one- or two-dimensional")
    if not isinstance(dt, (int, np.integer)) or dt < 1:
        raise ValueError("dt must be a positive integer")
    if values.shape[-1] % dt:
        raise ValueError("incidence length must be an exact multiple of dt")
    return values.reshape(*values.shape[:-1], -1, dt).sum(axis=-1)
