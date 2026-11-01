# -*- coding: utf-8 -*-
"""Test the hypothesis: the stored R results were generated with an EpiEstim
build whose get_r_from_R used a distcrete gamma SI (w = 0.5, centred
discretisation, support 0..1000) instead of the current discr_si + 1e-6
truncation. See vignette/functions.R in the paper repo (the paper's own EM
copy)."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import epiestim
from epiestim import estimate_R_agg, aggregate_inc, estimate_R  # noqa: F401
from epiestim import discr_si, _r2r0  # noqa: F401
from epiestim.rds_reader import read_rds, list_to_dict, RObject, Altrep  # noqa: F401
from scipy.special import gammainc

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main")


def gamma_mucv2shapescale(mu, sd):
    """epitrix::gamma_mucv2shapescale(mu, cv = sd/mu)."""
    cv = sd / mu
    return 1.0 / cv ** 2, mu * cv ** 2


def si_distcrete_gamma(mu, sd, n=1001):
    """distcrete('gamma', interval = 1, shape, scale, w = 0.5):
    d(k) = F(k + 0.5) - F(k - 0.5), k = 0..n-1 (F = gamma CDF)."""
    shape, scale = gamma_mucv2shapescale(mu, sd)
    k = np.arange(n)
    x = (k + 0.5) / scale
    xm = (k - 0.5) / scale
    F = lambda v: np.where(v > 0, gammainc(shape, np.where(v > 0, v, 0)), 0.0)
    return F(x) - F(xm)


def r2r0_distcrete(r, w):
    """epitrix::r2R0 with the exp-overflow guard, w = distcrete pmf.
    t = c(0, seq(0.5, length(w) - 0.5, 1)); if exp(-r * t_last) == Inf -> 0."""
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r, dtype=float)
    near0 = np.abs(r) < 1e-14
    out[near0] = 1.0
    if np.all(near0):
        return out
    t = np.concatenate([[0.0],
                        np.arange(0.5, len(w) - 0.5 + 1e-12, 1.0)])
    rr = r[~near0]
    t_last = t[-1]
    with np.errstate(over="ignore", invalid="ignore"):
        denom = -(w * np.diff(np.exp(-rr[:, None] * t), axis=1) / np.diff(t))
        val = rr / denom.sum(axis=1)
    # epitrix: if (exp(-e * t[length(t)]) == Inf) R0 <- 0
    val = np.where(np.isinf(np.exp(-rr * t_last)), 0.0, val)
    out[~near0] = val
    return out


def get_r_from_R_distcrete(R_vals, mean_si, std_si, total_t, grid):
    """Vignette-style get_r_from_R: distcrete gamma SI, no truncation.
    (total_t is unused - the old version always used the full distcrete SI.)"""
    precision, gmin, gmax = grid["precision"], grid["min"], grid["max"]
    w = si_distcrete_gamma(mean_si, std_si)
    w = w / w.sum()

    r_grid = np.arange(gmin, gmax + precision / 2, precision)
    R_grid = r2r0_distcrete(r_grid, w)
    idx = np.array([np.argmin(np.abs(R_grid - Rv)) for Rv in R_vals])

    guard = 0
    while (np.any(idx == 0) or np.any(idx == len(r_grid) - 1)) and guard < 8:
        mult = 5
        gmax = mult * gmax if gmax > 0 else -gmax
        gmin = mult * gmin if gmin < 0 else -gmin
        r_grid = np.arange(gmin, gmax + precision / 2, precision)
        R_grid = r2r0_distcrete(r_grid, w)
        idx = np.array([np.argmin(np.abs(R_grid - Rv)) for Rv in R_vals])
        guard += 1
    return r_grid[idx]


def r_df(res):
    d = list_to_dict(res)
    out = {}
    for k, v in d.items():
        if isinstance(v, RObject) and "data.frame" in (v.attrs.get("class") or []):
            cols = v.attrs["names"]
            data = {c: (x.val if isinstance(x, RObject) else x)
                    for c, x in zip(cols, v.val)}
            out[k] = pd.DataFrame(data)
        elif isinstance(v, Altrep):
            vals = v.state.val if isinstance(v.state, RObject) else v.state
            n, first, incr = (float(x) for x in vals)
            out[k] = np.arange(n) * incr + first
        elif isinstance(v, RObject):
            out[k] = np.asarray(v.val, dtype=float)
        else:
            out[k] = v
    return out


import pandas as pd  # noqa: E402


def run(name, cfg):
    print(f"=== {name} ===")
    import pyreadr
    df = pyreadr.read_r(f"{REPO}/{cfg['rds']}")[None]
    incid = df["Incidence"].to_numpy(dtype=float)
    weekly = aggregate_inc(incid, dt=7)

    ref = r_df(read_rds(
        f"{REPO}/supplementary_analysis/{name}/r_agg_sliding_result.rds"))
    rI = np.asarray(ref["I"], dtype=float)
    rR = ref["R"].set_index("t_start")

    # --- current port (discr_si) ---
    py = estimate_R_agg(weekly, dt=7, dt_out=7, recon_opt="match",
                        mean_si=cfg["mean_si"], std_si=cfg["std_si"])
    print(f"  discr_si : max|dI| = {np.max(np.abs(py['I'] - rI)):.6g}")

    # --- distcrete variant ---
    epiestim._get_r_from_R = get_r_from_R_distcrete
    py2 = estimate_R_agg(weekly, dt=7, dt_out=7, recon_opt="match",
                         mean_si=cfg["mean_si"], std_si=cfg["std_si"])
    print(f"  distcrete: max|dI| = {np.max(np.abs(py2['I'] - rI)):.6g}")
    rR2 = py2["R"].set_index("t_start")
    print(f"  distcrete: max|d Mean(R)| = "
          f"{np.max(np.abs(rR2['Mean(R)'] - rR['Mean(R)'])):.6g}")

    # where do the two variants differ?
    d = np.abs(py["I"] - py2["I"])
    idx = np.where(d > 1e-6)[0]
    if len(idx):
        print(f"  variant-vs-variant I diffs at days {idx[:10]}...")
    print(f"  day1: discr_si={py['I'][0]:.4f}  distcrete={py2['I'][0]:.4f}  "
          f"R={rI[0]:.4f}")


if __name__ == "__main__":
    # restore the original function reference first
    orig = epiestim._get_r_from_R
    run("covid_cases", dict(rds="real_data/daily_covid_cases.rds",
                            mean_si=6.3, std_si=4.2))
    epiestim._get_r_from_R = orig
    run("covid_deaths", dict(rds="real_data/daily_covid_deaths.rds",
                             mean_si=6.3, std_si=4.2))
    epiestim._get_r_from_R = orig
    run("flu", dict(rds="real_data/daily_flu.rds",
                    mean_si=3.6, std_si=1.6))
