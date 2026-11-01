# -*- coding: utf-8 -*-
"""Compare the Python port against the R results stored in the reproduction
repo (supplementary_analysis/**/r_*_result.rds). Max absolute differences of
Mean(R)/quantiles and reconstructed incidence are printed per analysis.

The EM's R->r link uses the distcrete-gamma serial interval (si_variant =
"distcrete", the default) — the EpiEstim build that generated the stored
results (paper repo vignette/functions.R); the R posterior uses discr_si."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from epiestim import estimate_R, estimate_R_agg, aggregate_inc  # noqa: E402
from epiestim.rds_reader import read_rds, list_to_dict, RObject, Altrep  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main")
DATASETS = {
    "flu": dict(rds="real_data/daily_flu.rds", mean_si=3.6, std_si=1.6),
    "covid_cases": dict(rds="real_data/daily_covid_cases.rds",
                        mean_si=6.3, std_si=4.2),
    "covid_deaths": dict(rds="real_data/daily_covid_deaths.rds",
                         mean_si=6.3, std_si=4.2),
}


def r_df(res):
    """R result object -> dict {name: DataFrame / array / scalar}"""
    d = list_to_dict(res)
    out = {}
    for k, v in d.items():
        if isinstance(v, RObject) and "data.frame" in (v.attrs.get("class") or []):
            cols = v.attrs["names"]
            data = {c: (x.val if isinstance(x, RObject) else x)
                    for c, x in zip(cols, v.val)}
            out[k] = pd.DataFrame(data)
        elif isinstance(v, Altrep):
            # compact_intseq: state = [n, first, incr] -> arange
            vals = v.state.val if isinstance(v.state, RObject) else v.state
            n, first, incr = (float(x) for x in vals)
            out[k] = np.arange(n) * incr + first
        elif isinstance(v, RObject):
            out[k] = np.asarray(v.val, dtype=float)
        else:
            out[k] = v
    return out


def compare(name, py_R, r_R, tag):
    common = sorted(set(py_R["t_start"]).intersection(r_R["t_start"]))
    p = py_R.set_index("t_start").loc[common]
    r = r_R.set_index("t_start").loc[common]
    diffs = {}
    for col in ["Mean(R)", "Std(R)", "Quantile.0.025(R)", "Quantile.0.975(R)"]:
        diffs[col] = np.max(np.abs(p[col] - r[col]))
    print(f"  {tag}: {len(common)} windows aligned")
    for col, d in diffs.items():
        print(f"    max|d {col}| = {d:.3g}")


def main():
    for name, cfg in DATASETS.items():
        print(f"=== {name} ===")
        import pyreadr
        df = pyreadr.read_r(f"{REPO}/{cfg['rds']}")[None]
        incid = df["Incidence"].to_numpy(dtype=float)
        weekly = aggregate_inc(incid, dt=7)

        # R reference results
        ref = {}
        for tag in ["r_rep_sliding_result", "r_rep_result",
                    "r_agg_sliding_result", "r_agg_result"]:
            obj = read_rds(f"{REPO}/supplementary_analysis/{name}/{tag}.rds")
            ref[tag] = r_df(obj)

        # --- reported daily, weekly sliding windows (plain estimate_R) ---
        py = estimate_R(incid, mean_si=cfg["mean_si"], std_si=cfg["std_si"])
        compare(name, py["R"], ref["r_rep_sliding_result"]["R"],
                "reported daily / weekly sliding")

        # --- reported daily, 2-day sliding windows ---
        t_start = np.arange(2, len(incid))
        py = estimate_R(incid, mean_si=cfg["mean_si"], std_si=cfg["std_si"],
                        t_start=t_start, t_end=t_start + 1)
        compare(name, py["R"], ref["r_rep_result"]["R"], "reported daily / daily")

        # --- aggregated weekly, EM reconstruction, weekly sliding ---
        py = estimate_R_agg(weekly, dt=7, dt_out=7, recon_opt="match",
                            mean_si=cfg["mean_si"], std_si=cfg["std_si"])
        compare(name, py["R"], ref["r_agg_sliding_result"]["R"],
                "aggregated weekly / weekly sliding")
        rI = np.asarray(ref["r_agg_sliding_result"]["I"], dtype=float)
        print(f"    max|d reconstructed I| = "
              f"{np.max(np.abs(py['I'] - rI)):.3g}")

        # --- aggregated weekly, EM reconstruction, 2-day sliding ---
        py = estimate_R_agg(weekly, dt=7, dt_out=2, recon_opt="match",
                            mean_si=cfg["mean_si"], std_si=cfg["std_si"])
        compare(name, py["R"], ref["r_agg_result"]["R"],
                "aggregated weekly / daily")


if __name__ == "__main__":
    main()
