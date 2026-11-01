# -*- coding: utf-8 -*-
"""Python reproduction of the Nash et al. (2023) S1 Appendix simulation
studies and remaining supplementary analyses.

Studies (each writes ../../results/figures/*.pdf, mirroring the paper's
EM_EpiEstim_Nash2023-main/supplementary_figures/*.pdf):

  constant_rt            -> Figure S1  (constant_rt.pdf)
  tv_sudden              -> Figures S2/S3 (tv_rt_sudden_increase/decrease.pdf,
                             tv_rt_sudden_midwindow.pdf)
  tv_gradual             -> Figures S4/S5 (tv_rt_gradual_*.pdf)
  weekend_effects        -> weekend_effects.pdf
  diff_aggs              -> different_aggregations_aligned.pdf /
                             different_aggregations_misaligned.pdf
  mid_agg                -> mid_window_variability.pdf
  loess                  -> loess_reconstructions.pdf, loess_rt.pdf,
                             loess_rt_bias.pdf
  weekday                -> incidence_weekday.pdf
  zika                   -> zika_rt.pdf
  appendix_rt            -> covid_cases_rt.pdf, covid_rt_bias.pdf,
                             covid_deaths_rt.pdf, covid_deaths_rt_bias.pdf,
                             flu_rt.pdf, flu_rt_bias.pdf

Simulations follow projections::project(x, R, si, n_days, n_sim,
time_change, instantaneous_R = TRUE) exactly (renewal equation, Poisson);
the RNG is numpy's default_rng per the group conventions in CONVENTIONS.md
(R's Mersenne-Twister stream is not reproducible cross-language), so
individual trajectories differ from the stored R sims but the generating
process and all study parameters are identical.

Run:  PYTHONPATH="." python -P sim_study.py [study ...]
"""

import hashlib
import os
import pickle
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from epiestim import estimate_R, estimate_R_agg, aggregate_inc, discr_si  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "figures")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main")
_CACHEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_est_cache")


def _sims_key(sims):
    """Content hash of a simulation matrix (n_sim x n_days) so identical
    data -> identical estimation results across runs."""
    a = np.ascontiguousarray(np.asarray(sims, dtype=float))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def _cached(key, compute):
    """Pickle cache for the (slow) per-sim estimation runs.

    A cache entry that cannot be unpickled (corrupt file, numpy-version
    mismatch across machines) is treated as a miss and recomputed."""
    path = os.path.join(_CACHEDIR, key + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(path)
    obj = compute()
    os.makedirs(_CACHEDIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj

MEAN_SI, SD_SI = 3.6, 1.6
N_DAYS, N_SIM = 70, 100
SEED_CONST, SEED_TV = 6, 5          # group convention (CONVENTIONS.md sec. 1)

# shared plot style --------------------------------------------------------
C_DAILY, C_DAILY_FILL = "darkgrey", "grey"
C_WEEKLY, C_WEEKLY_FILL = "darkgreen", "mediumseagreen"
C_TRUE = "black"
C_COV_DAILY, C_COV_TRUE = "#1b82e6", "#061d33"
C_BIAS, C_BIAS_FILL = "darkgreen", "mediumseagreen"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "lightgrey", "axes.linewidth": 0.5,
    "axes.grid": False, "figure.dpi": 150,
})


def style_ax(ax, xlim=None, ylim=None, xlabel="Time", ylabel="",
             label=None):
    """theme_ipsum-like minimal axes; label = panel letter A..P."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if label:
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold",
                     pad=10)
    ax.tick_params(colors="black")


def panel_label(ax, letter):
    ax.set_title(letter, loc="left", fontsize=12, fontweight="bold",
                 pad=10)


# ---- simulation core -----------------------------------------------------

def sim_si():
    """Serial interval used in all simulations: discr_si(0:30) normalised,
    w_0 dropped (si_one = lags 1..30), as in simulate_incidence.R."""
    si = discr_si(np.arange(0, 31), MEAN_SI, SD_SI)
    si_zero = si / si.sum()
    return si_zero[1:]


def simulate_incidence(Rt, time_change=None, n_days=N_DAYS, n_sim=N_SIM,
                       seed=None, si=None):
    """projections::project(instantaneous_R = TRUE) equivalent.

    Rt: scalar (constant), length-2 vector (sudden change, time_change = 34),
    or length-k vector with k-1 change points (gradual: 32-value ramp,
    time_change = seq(19, 49); mid-aggregation: 20 values, 19 change points).

    Semantics (verified against the paper's stored sims):
    - 7 seed days of 10 cases; 70 simulated days follow.
    - new cases at simulated day d ~ Poisson(R_t[d] * sum_s w_s * I_{d-s}),
      with R_t[d] the value governing day d.
    - sudden time_change = 34 -> days 1-35 use Rt[0], days 36-70 Rt[1].
    - gradual time_change = seq(19, 49) -> days 1-20 Rt[0], days 21-50
      Rt[1:31], days 51-70 Rt[31].
    - arbitrary time_change -> R value advances *after* each change row.

    Returns array of shape (n_sim, n_days) (CONVENTIONS.md sec. 4).
    """
    if si is None:
        si = sim_si()
    si = np.asarray(si, dtype=float)
    Rt = np.atleast_1d(np.asarray(Rt, dtype=float))
    rng = np.random.default_rng(seed)

    t_max = n_days + 7 - 1                    # 76: last absolute time index
    # absolute change rows (projections: time_change <- 8 + time_change - 1)
    if time_change is None:
        tc = np.array([], dtype=int)
    else:
        tc = np.atleast_1d(np.asarray(time_change, dtype=int)) + 7

    R_t = np.empty(t_max)
    p = 0
    for i in range(1, t_max + 1):
        R_t[i - 1] = Rt[p]
        if i in tc:
            p += 1

    w = np.zeros(t_max)
    w[:len(si)] = si
    out = np.zeros((t_max + 1, n_sim))
    out[:7, :] = 10.0
    for i in range(7, t_max + 1):
        lam = (w[:i][::-1][:, None] * out[:i, :]).sum(axis=0) * R_t[i - 1]
        out[i, :] = rng.poisson(lam)
    return out[7:, :].T                          # (n_sim, n_days)


def add_weekend_effect(sims):
    """Move 80% of Sat/Sun cases (days 6, 7 of each week) to the following
    Mon/Tue (weekend_effects.R lines 51-66)."""
    out = sims.copy()
    mon = np.arange(0, out.shape[1], 7)
    tue = np.arange(1, out.shape[1], 7)
    sat = np.arange(5, out.shape[1], 7)
    sun = np.arange(6, out.shape[1], 7)
    mt = np.sort(np.concatenate([mon, tue]))
    ss = np.sort(np.concatenate([sat, sun]))
    prop = out[:, ss] * 0.8
    out[:, ss] -= prop
    out[:, mt] += prop
    return out


def sim_cache():
    """All scenario matrices, generated once with the group-convention seeds
    so that every study uses identical simulated data (as the R pipeline
    does, sharing the stored .rds files)."""
    cache = {}
    for rt in (1.0, 1.25, 1.5, 1.75):
        cache[f"const_{rt}"] = simulate_incidence(rt, seed=SEED_CONST)
    for a, b in [(1, 1.25), (1.25, 1.5), (1.5, 1.75),
                 (1.25, 0.75), (1.25, 1), (1.5, 1.25), (1.75, 1.5)]:
        cache[f"sud_{a}to{b}"] = simulate_incidence([a, b], 34, seed=SEED_TV)
    for a, b in [(1, 1.25), (1.25, 1.5), (1.5, 1.75),
                 (1.25, 0.75), (1.25, 1), (1.5, 1.25), (1.75, 1.5)]:
        ramp = np.linspace(a, b, 32)
        cache[f"grad_{a}to{b}"] = simulate_incidence(
            ramp, np.arange(19, 50), seed=SEED_TV)
    # mid-aggregation: alternating weekly R, changes at days 4 and 6 of each
    # week (mid_aggregation_variations.R lines 41-61)
    s1 = np.arange(4, 71, 7)
    s2 = np.arange(6, 71, 7)
    idx_tc = np.sort(np.concatenate([s1, s2]))[:-1]
    for a, b in [(1.25, 1.5), (1.5, 1.25)]:
        rt = np.resize(np.tile([a, b], 10), len(idx_tc) + 1)
        cache[f"mid_{a}to{b}"] = simulate_incidence(
            rt, idx_tc, seed=SEED_TV)
    return cache


# ---- estimation engine ---------------------------------------------------

def run_agg(sims, dt=7, dt_out=7, verbose=False):
    """estimate_R_agg(match) on each simulation row (weekly aggregated).
    Returns list (per sim) of R DataFrames."""
    def compute():
        out = []
        for s in sims:
            res = estimate_R_agg(aggregate_inc(s, dt=dt), dt=dt,
                                 dt_out=dt_out, recon_opt="match",
                                 mean_si=MEAN_SI, std_si=SD_SI)
            out.append(res["R"])
        return out
    return _cached(f"agg_{_sims_key(sims)}_{dt}_{dt_out}", compute)


def run_daily(sims, t_start=None, t_end=None):
    """estimate_R (parametric, reported daily) on each simulation row.
    Default: weekly sliding windows."""
    def compute():
        out = []
        for s in sims:
            res = estimate_R(s, mean_si=MEAN_SI, std_si=SD_SI,
                             t_start=t_start, t_end=t_end)
            out.append(res["R"])
        return out
    cfg = hashlib.sha1(repr(t_start).encode() + repr(t_end).encode()
                       ).hexdigest()[:8]
    return _cached(f"day_{_sims_key(sims)}_{cfg}", compute)


def summarize(ests, t_end=None):
    """R process_res(): per t_end, mean over sims of Mean(R) / Q2.5 / Q97.5.
    Returns DataFrame with t_end, mean_r, lower, upper."""
    all_r = pd.concat(ests, ignore_index=True)
    g = all_r.groupby("t_end", sort=True)
    out = pd.DataFrame({
        "mean_r": g["Mean(R)"].mean(),
        "lower": g["Quantile.0.025(R)"].mean(),
        "upper": g["Quantile.0.975(R)"].mean(),
    }).reset_index()
    return out


def coverage(ests, ref):
    """Per t_end: proportion of sims whose 95% CrI contains ref[t_end]
    (ref indexed by t_end)."""
    all_r = pd.concat(ests, ignore_index=True)
    inside = ((all_r["Quantile.0.025(R)"] <=
               all_r["t_end"].map(ref)) &
              (all_r["Quantile.0.975(R)"] >=
               all_r["t_end"].map(ref)))
    all_r = all_r.assign(inside=inside.astype(float))
    g = all_r.groupby("t_end", sort=True)
    return pd.DataFrame({"t_end": g["inside"].mean().index,
                         "coverage": g["inside"].mean().to_numpy()})


def r_plot(ax, ests_sum, true=None, source_colors=None, ylim=(0.7, 2),
           xlim=(14, 70), ylabel="Weekly sliding R estimate", hline=1):
    """The paper's standard R-estimate panel: mean line + mean-quantile
    ribbon for 'daily' and 'weekly' sources, dashed 'true' line."""
    for name, color, fill in (("daily", C_DAILY, C_DAILY_FILL),
                              ("weekly", C_WEEKLY, C_WEEKLY_FILL)):
        if name not in ests_sum:
            continue
        d = ests_sum[name]
        ax.plot(d["t_end"], d["mean_r"], color=color, lw=1.2, label=name)
        ax.fill_between(d["t_end"], d["lower"], d["upper"], color=fill,
                        alpha=0.2, lw=0)
    if true is not None:
        t = np.arange(len(true)) + (true.index[0] if hasattr(true, "index")
                                    else 14)
        ax.plot(t, np.asarray(true, dtype=float), color=C_TRUE, ls="--",
                lw=1.0)
    if hline is not None:
        ax.axhline(hline, color="grey", ls=":", lw=1)
    style_ax(ax, xlim=xlim, ylim=ylim, ylabel=ylabel)


def legend(ax, names=("daily", "weekly", "true"), loc="upper left"):
    handles = [
        plt.Line2D([], [], color=C_DAILY, lw=1.2, label="Daily data"),
        plt.Line2D([], [], color=C_WEEKLY, lw=1.2, label="Reconstructed data"),
        plt.Line2D([], [], color=C_TRUE, ls="--", lw=1.0, label="True R"),
    ]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9)


# =========================================================================
# Study 1: constant Rt (16-panel Figure S1)
# =========================================================================

def study_constant_rt(cache=None):
    print("=== constant_rt ===")
    if cache is None:
        cache = sim_cache()
    rts = [1.0, 1.25, 1.5, 1.75]
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for i, rt in enumerate(rts):
        sims = cache[f"const_{rt}"]
        weekly_ests = run_agg(sims)
        daily_ests = run_daily(sims)
        wsum = summarize(weekly_ests)               # t_end 14..70
        dsum = summarize(daily_ests)                # t_end 8..70
        te = wsum["t_end"].to_numpy()

        # panel 1: R estimates
        ax = axes[0, i]
        r_plot(ax, {"daily": dsum, "weekly": wsum},
               true=pd.Series(rt, index=te), ylim=(0.7, 2))
        if i == 3:
            legend(ax, loc=(0.02, 0.62))

        # panel 2: bias (weekly mean - true)
        ax = axes[1, i]
        bias = wsum["mean_r"].to_numpy() - rt
        lo = wsum["lower"].to_numpy() - rt
        hi = wsum["upper"].to_numpy() - rt
        ax.plot(te, bias, color=C_BIAS, lw=0.7)
        ax.fill_between(te, lo, hi, color=C_BIAS_FILL, alpha=0.3, lw=0)
        ax.axhline(0, ls="--", color="black", lw=0.8)
        style_ax(ax, ylim=(-1, 1), ylabel="Bias")

        # panel 3: uncertainty (mean CrI width)
        ax = axes[2, i]
        unc = wsum["upper"].to_numpy() - wsum["lower"].to_numpy()
        ax.plot(te, unc, color="#061d33", lw=0.7)
        style_ax(ax, ylim=(0, 0.8), ylabel="Uncertainty")

        # panel 4: coverage
        ax = axes[3, i]
        dref = pd.Series(dsum["mean_r"].to_numpy(), index=dsum["t_end"])
        cov_d = coverage(weekly_ests, dref)
        cov_t = coverage(weekly_ests, pd.Series(rt, index=te))
        ax.plot(cov_d["t_end"], cov_d["coverage"], color=C_COV_DAILY, lw=0.6,
                marker="o", ms=2)
        ax.plot(cov_t["t_end"], cov_t["coverage"], color=C_COV_TRUE, lw=0.6,
                marker="o", ms=2)
        ax.axhline(0.95, ls="--", color="black", lw=0.8)
        style_ax(ax, ylim=(0, 1), ylabel="95% Coverage")
        if i == 3:
            ax.legend(handles=[plt.Line2D([], [], color=C_COV_DAILY, lw=0.6,
                                          label="Daily data R"),
                               plt.Line2D([], [], color=C_COV_TRUE, lw=0.6,
                                          label="True R")],
                      loc=(0.05, 0.02), frameon=False, fontsize=9)

    for row, letters in enumerate("ABCD EFGH IJKL MNOP".split()):
        for j, letter in enumerate(letters):
            panel_label(axes[row, j], letter)
    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    fig.savefig(f"{FIGDIR}/constant_rt.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/constant_rt.pdf")


# =========================================================================
# Studies 2-3: time-varying Rt, sudden / gradual change (Figures S2-S5)
# =========================================================================
# Both studies share this structure (time_varying_rt_sudden.R /
# time_varying_rt_gradual.R):
#   7 scenarios (3 increasing, 4 decreasing), 100 sims, seed 5;
#   weekly agg estimates (t_end 14..70, 57) vs daily-data reference
#   (default windows, t_end 8..70, 63);
#   the plotted "true" line is the paper's idealisation on the weekly
#   t_end grid, not the per-window ground truth:
#     sudden:  R1 on t_end 14..35, R2 on 36..70 (vline 35);
#     gradual: R1 14..18, linear ramp 19..50, R2 51..70 (vlines 20, 50);
#   mid-window variant relabels t_end -> t_end - 3.5 with
#     sudden:  R1 on mid 10.5..34.5; gradual: R1 10.5..17.5,
#     ramp 18.5..49.5, R2 50.5..66.5.

TV_SCENARIOS = [(1, 1.25), (1.25, 1.5), (1.5, 1.75),
                (1.25, 0.75), (1.25, 1), (1.5, 1.25), (1.75, 1.5)]


def tv_true(kind, a, b):
    if kind == "sudden":
        return np.r_[np.repeat(a, 22), np.repeat(b, 35)]
    return np.r_[np.repeat(a, 5), np.linspace(a, b, 32), np.repeat(b, 20)]


def tv_mid_true(kind, a, b):
    if kind == "sudden":
        return np.r_[np.repeat(a, 25), np.repeat(b, 32)]
    return np.r_[np.repeat(a, 8), np.linspace(a, b, 32), np.repeat(b, 17)]


def tv_scenario(sims, kind, a, b):
    """All per-scenario summaries: weekly agg + daily ref estimates, the
    idealised true line, and the two coverage series (vs daily-data mean R
    and vs true R, per t_end over the 100 sims)."""
    weekly = run_agg(sims)
    daily = run_daily(sims)
    wsum = summarize(weekly)
    dsum = summarize(daily)
    te = wsum["t_end"].to_numpy()
    true = tv_true(kind, a, b)
    cov_t = coverage(weekly, pd.Series(true, index=te))
    cov_d = coverage(weekly, pd.Series(dsum["mean_r"].to_numpy(),
                                       index=dsum["t_end"]))
    return dict(wsum=wsum, dsum=dsum, te=te, true=true,
                cov_d=cov_d, cov_t=cov_t)


def _tv_bias_panel(ax, sc, vlines):
    te, true = sc["te"], sc["true"]
    w = sc["wsum"]
    bias = w["mean_r"].to_numpy() - true
    lo = w["lower"].to_numpy() - true
    hi = w["upper"].to_numpy() - true
    for v in vlines:
        ax.axvline(v, ls="--", color="lightgrey", lw=0.8)
    ax.plot(te, bias, color=C_BIAS, lw=0.7)
    ax.fill_between(te, lo, hi, color=C_BIAS_FILL, alpha=0.3, lw=0)
    ax.axhline(0, ls="--", color="black", lw=0.8)
    style_ax(ax, ylim=(-1, 1), ylabel="Bias")


def _tv_unc_panel(ax, sc, vlines):
    for v in vlines:
        ax.axvline(v, ls="--", color="lightgrey", lw=0.8)
    w = sc["wsum"]
    unc = w["upper"].to_numpy() - w["lower"].to_numpy()
    ax.plot(sc["te"], unc, color="#061d33", lw=0.7)
    style_ax(ax, ylim=(0, 0.8), ylabel="Uncertainty")


def _tv_cov_panel(ax, sc, vlines, legend_pos=None, mid=False):
    """Coverage series: CrI of the weekly estimates contains the daily-data
    mean R (#1b82e6) or the true R (#061d33); with mid=True the same
    proportions are drawn at the window mid-points and the end-window true
    series (grey) is added, as in the midwindow figures."""
    x = sc["te"] - 3.5 if mid else sc["te"]
    for v in vlines:
        ax.axvline(v, ls="--", color="lightgrey", lw=0.8)
    ax.plot(x, sc["cov_d"]["coverage"], color=C_COV_DAILY, lw=0.6,
            marker="o", ms=2)
    ax.plot(x, sc["cov_t"]["coverage"], color=C_COV_TRUE, lw=0.6,
            marker="o", ms=2)
    if mid:
        ax.plot(sc["te"], sc["cov_t"]["coverage"], color="grey", lw=0.6,
                marker="o", ms=2)
    ax.axhline(0.95, ls="--", color="black", lw=0.8)
    style_ax(ax, ylim=(0, 1), ylabel="95% Coverage")
    if legend_pos:
        handles = [plt.Line2D([], [], color=C_COV_DAILY, lw=0.6,
                              label="Daily data R")]
        if mid:
            handles += [plt.Line2D([], [], color=C_COV_TRUE, lw=0.6,
                                   label="True R (mid-window)"),
                        plt.Line2D([], [], color="grey", lw=0.6,
                                   label="True R (end of window)")]
        else:
            handles += [plt.Line2D([], [], color=C_COV_TRUE, lw=0.6,
                                   label="True R")]
        ax.legend(handles=handles, loc=legend_pos, frameon=False, fontsize=9)


def study_tv(kind, cache=None):
    """Figures S2/S3 (sudden) or S4/S5 (gradual): increase, decrease and
    midwindow. Prints the mean over time of the true-R coverage (paper
    comments: sudden inc [87-96%] dec [85-90%], excl t_end 36:42
    [95-97%], mid inc [88-96%] dec [85-93%]; gradual inc [77-96%]
    dec [61-94%], mid inc [94-97%] dec [88-96%])."""
    print(f"=== tv_{kind} ===")
    if cache is None:
        cache = sim_cache()
    key = "sud" if kind == "sudden" else "grad"
    scenarios = [tv_scenario(cache[f"{key}_{a}to{b}"], kind, a, b)
                 for a, b in TV_SCENARIOS]
    vlines = [35] if kind == "sudden" else [20, 50]
    inc, dec = scenarios[:3], scenarios[3:]

    def _cov_report(tag, scs, excl=None):
        vals = [s["cov_t"]["coverage"][~np.isin(s["te"], excl)]
                if excl is not None else s["cov_t"]["coverage"]
                for s in scs]
        print(f"  {tag}: " + "  ".join(
            f"{v.mean()*100:.1f}%" for v in vals))

    _cov_report("true-R coverage (inc)", inc)
    _cov_report("true-R coverage (dec)", dec)
    if kind == "sudden":
        _cov_report("excl t_end 36:42 (inc)", inc, excl=np.arange(36, 43))
        _cov_report("excl t_end 36:42 (dec)", dec, excl=np.arange(36, 43))

    # ---- increase / decrease figures (4 rows x 3 or 4 cols) ----
    def _fig(subs, outname, r_legend_idx, cov_legend_idx, cov_loc):
        fig, axes = plt.subplots(4, len(subs), figsize=(12, 12))
        for i, sc in enumerate(subs):
            ax = axes[0, i]
            r_plot(ax, {"daily": sc["dsum"], "weekly": sc["wsum"]},
                   true=pd.Series(sc["true"], index=sc["te"]),
                   ylim=(0.5, 2))
            if i == r_legend_idx:
                legend(ax, loc=(0.80, 0.37))
            _tv_bias_panel(axes[1, i], sc, vlines)
            _tv_unc_panel(axes[2, i], sc, vlines)
            _tv_cov_panel(axes[3, i], sc, vlines,
                          legend_pos=cov_loc if i == cov_legend_idx else None)
        letters = np.array(list("ABCDEFGHIJKLMNOP"))[:4 * len(subs)]
        for k, letter in enumerate(letters):
            panel_label(axes[k // len(subs), k % len(subs)], letter)
        fig.tight_layout(w_pad=1.6, h_pad=2.0)
        fig.savefig(f"{FIGDIR}/{outname}.pdf")
        plt.close(fig)
        print(f"  -> {FIGDIR}/{outname}.pdf")

    _fig(inc, f"tv_rt_{kind}_increase", 2, 2 if kind == "sudden" else 0,
         (0.90, 0.30) if kind == "sudden" else (0.80, 0.37))
    _fig(dec, f"tv_rt_{kind}_decrease", 3, 3 if kind == "sudden" else 0,
         (0.90, 0.30) if kind == "sudden" else (0.80, 0.37))

    # ---- midwindow figure ----
    # top band: 3 increasing scenarios (R + coverage) with a legends column;
    # bottom band: 4 decreasing scenarios (R + coverage).
    mid_ylims_inc = {"sudden": (0.7, 2), "gradual": (0.75, 1.8)}[kind]
    xmin = 10.5 if kind == "sudden" else 10
    fig = plt.figure(figsize=(12, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
    gs_inc = outer[0].subgridspec(2, 4, width_ratios=[1, 1, 1, 0.45])
    gs_dec = outer[1].subgridspec(2, 4)
    for i, sc in enumerate(inc):
        ax = fig.add_subplot(gs_inc[0, i])
        r_plot(ax, {"daily": sc["dsum"], "weekly": sc["wsum"]},
               true=pd.Series(sc["true"], index=sc["te"]),
               ylim=mid_ylims_inc, xlim=(xmin, 70))
        ax = fig.add_subplot(gs_inc[1, i])
        _tv_cov_panel(ax, sc, vlines, mid=True)
    for j, sc in enumerate(dec):
        ax = fig.add_subplot(gs_dec[0, j])
        r_plot(ax, {"daily": sc["dsum"], "weekly": sc["wsum"]},
               true=pd.Series(sc["true"], index=sc["te"]),
               ylim=(0.5, 2), xlim=(xmin, 70))
        ax = fig.add_subplot(gs_dec[1, j])
        _tv_cov_panel(ax, sc, vlines, mid=True)
    axl = fig.add_subplot(gs_inc[:, 3])
    axl.axis("off")
    h1 = [plt.Line2D([], [], color=C_DAILY, lw=1.2, label="Daily data"),
          plt.Line2D([], [], color=C_WEEKLY, lw=1.2,
                     label="Reconstructed data"),
          plt.Line2D([], [], color=C_TRUE, ls="--", lw=1.0, label="True R"),
          plt.Line2D([], [], color=C_COV_DAILY, lw=0.6,
                     label="Daily data R"),
          plt.Line2D([], [], color=C_COV_TRUE, lw=0.6,
                     label="True R (mid-window)"),
          plt.Line2D([], [], color="grey", lw=0.6,
                     label="True R (end of window)")]
    axl.legend(handles=h1, loc="center", frameon=False, fontsize=10)
    letters = list("ABCDEF") + list("GHIJKLMN")
    for k, letter in enumerate(letters):
        if k < 6:
            ax = fig.add_subplot(gs_inc[k % 2, k // 2])
        else:
            ax = fig.add_subplot(gs_dec[(k - 6) % 2, (k - 6) // 2])
        panel_label(ax, letter)
    fig.savefig(f"{FIGDIR}/tv_rt_{kind}_midwindow.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/tv_rt_{kind}_midwindow.pdf")
    print("  mid true-R coverage: inc " + "  ".join(
        f"{s['cov_t']['coverage'].mean()*100:.1f}%" for s in inc)
        + "  dec " + "  ".join(
        f"{s['cov_t']['coverage'].mean()*100:.1f}%" for s in dec))


def study_tv_sudden(cache=None):
    study_tv("sudden", cache)


def study_tv_gradual(cache=None):
    study_tv("gradual", cache)


# =========================================================================
# Study 4: weekend effects (weekend_effects.pdf)
# =========================================================================
# 80% of Sat/Sun cases are moved to the following Mon/Tue in each of the
# three scenario sims; estimation on the aggregated weekly data with
# non-sliding (dt_out = 2), weekly sliding (default) and two-week sliding
# (dt_out = 14) windows, each compared with the matching daily-data
# reference (nonsliding: t_end 4..70; sliding: 8..70; two-week: 15..70).
# Columns of the figure are const / sud / grad; the daily reference curve
# is filtered to t_end >= min(agg t_end) (weekend_effects.R lines 374-381).
# NOTE: the paper's incidence panels plot the same daily simulation twice
# ("Simulated" and "Reconstructed" — result[[4]]$I *is* the daily input,
# not the EM reconstruction), so we reproduce that faithfully and flag it
# in the README.

def study_weekend_effects(cache=None):
    print("=== weekend_effects ===")
    if cache is None:
        cache = sim_cache()
    base = {"const": cache["const_1.5"],
            "sud": cache["sud_1.5to1.25"],
            "grad": cache["grad_1.25to1.5"]}
    true = {"const": np.repeat(1.5, 70),
            "sud": np.r_[np.repeat(1.5, 35), np.repeat(1.25, 35)],
            "grad": np.r_[np.repeat(1.25, 18),
                          np.linspace(1.25, 1.5, 32), np.repeat(1.5, 20)]}

    ests = {}
    for name in ("const", "sud", "grad"):
        sims = add_weekend_effect(base[name])
        ns_d = run_daily(sims, t_start=np.arange(2, 70),
                         t_end=np.arange(3, 71))
        sl_d = run_daily(sims)
        tw_d = run_daily(sims, t_start=np.arange(2, 58),
                         t_end=np.arange(15, 71))
        ests[name] = {
            "ns": (summarize(run_agg(sims, dt_out=2)), summarize(ns_d)),
            "sl": (summarize(run_agg(sims)), summarize(sl_d)),
            "tw": (summarize(run_agg(sims, dt_out=14)), summarize(tw_d)),
        }
        # daily reference filtered to t_end >= min(agg t_end)
        for k in ("ns", "sl", "tw"):
            w, d = ests[name][k]
            ests[name][k] = (w, d[d["t_end"] >= w["t_end"].min()])

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    cols = ["const", "sud", "grad"]
    row_specs = [("ns", (0, 5), "Daily R estimate"),
                 ("sl", (0.9, 2), "Weekly sliding R estimate"),
                 ("tw", (1, 2), "Two week sliding R estimate")]

    # row 1: incidence (the daily simulation plotted twice — see note above)
    for j, name in enumerate(cols):
        ax = axes[0, j]
        sim = base[name][0, :]
        ax.plot(np.arange(1, 71), sim, color="darkgrey", alpha=0.8, lw=0.8,
                label="Simulated")
        ax.plot(np.arange(1, 71), sim, color="#297a4d", lw=0.8,
                label="Reconstructed")
        style_ax(ax, ylabel="Daily incidence")
        if j == 2:
            ax.legend(loc=(0.30, 0.87), frameon=False, fontsize=10)
    # rows 2-4: R estimates
    for i, (kind, ylim, ylab) in enumerate(row_specs):
        for j, name in enumerate(cols):
            ax = axes[i + 1, j]
            w, d = ests[name][kind]
            r_plot(ax, {"daily": d, "weekly": w},
                   true=pd.Series(true[name], index=np.arange(1, 71)),
                   ylim=ylim, xlim=(0, 70), ylabel=ylab, hline=None)
            if i == 2 and j == 2:                # legend on grad tw panel
                legend(ax, loc=(0.35, 0.90))
    for row, letters in enumerate(("ABC", "DEF", "GHI", "JKL")):
        for j, letter in enumerate(letters):
            panel_label(axes[row, j], letter)
    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    fig.savefig(f"{FIGDIR}/weekend_effects.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/weekend_effects.pdf")


# =========================================================================
# Study 5: different temporal aggregations, aligned / misaligned
# =========================================================================
# 3 scenarios (const/sud/grad), aggregated at dt = 3/7/10/14 days. The
# aligned variant drops the first 2/0/5/7 days so aggregation windows line
# up with the step change; the misaligned variant uses days 1:69 / 3:65 /
# 1:70 / 1:70. Weekly (7-day) sliding windows on the EM reconstruction for
# every dt ("weekly"), plus dt-day windows for 10/14 days ("match", navy).
# The t_end axis is shifted back by the number of dropped days, and
# grey96 polygons cover the excluded edge periods.

def study_diff_aggs(cache=None):
    print("=== diff_aggs ===")
    if cache is None:
        cache = sim_cache()
    base = {"const": cache["const_1.5"],
            "sud": cache["sud_1.5to1.25"],
            "grad": cache["grad_1.25to1.5"]}
    true = {"const": np.repeat(1.5, 70),
            "sud": np.r_[np.repeat(1.5, 35), np.repeat(1.25, 35)],
            "grad": np.r_[np.repeat(1.25, 18),
                          np.linspace(1.25, 1.5, 32), np.repeat(1.5, 20)]}
    names = ["const", "sud", "grad"]
    # per aggregation window: (dt, first day 0-based, last day 0-based
    # exclusive, t_end shift back to original day coordinates)
    configs = {"aligned": [(3, 2, 68, 2), (7, 0, 70, 0), (10, 5, 65, 5),
                           (14, 7, 63, 7)],
               "misaligned": [(3, 0, 69, 0), (7, 2, 65, 2), (10, 0, 70, 0),
                              (14, 0, 70, 0)]}
    guides = {"aligned": {3: np.arange(5, 69, 3), 7: np.arange(7, 71, 7),
                          10: np.arange(15, 66, 10),
                          14: np.arange(21, 64, 14)},
              "misaligned": {3: np.arange(3, 70, 3), 7: np.arange(9, 66, 7),
                             10: np.arange(10, 71, 10),
                             14: np.arange(14, 71, 14)}}
    excl = {"aligned": {3: [(1, 3), (68, 70)], 7: [],
                        10: [(1, 6), (65, 70)], 14: [(1, 8), (63, 70)]},
            "misaligned": {3: [(69, 70)], 7: [(1, 3), (65, 70)], 10: [],
                           14: []}}
    legend_loc = {"aligned": (0.67, 0.87), "misaligned": (0.75, 0.87)}

    for kind, cfgs in configs.items():
        data = {}
        for dt, s0, s1, shift in cfgs:
            for name in names:
                sims = base[name][:, s0:s1]
                w = summarize(run_agg(sims, dt=dt))
                w = w.assign(t_end=w["t_end"] + shift)
                entry = {"w": w}
                if dt in (10, 14):
                    m = summarize(run_agg(sims, dt=dt, dt_out=dt))
                    m = m.assign(t_end=m["t_end"] + shift)
                    entry["m"] = m
                data[(dt, name)] = entry

        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        for row, (dt, s0, s1, shift) in enumerate(cfgs):
            for col, name in enumerate(names):
                ax = axes[row, col]
                entry = data[(dt, name)]
                for v in guides[kind][dt]:
                    ax.axvline(v, color="darkgrey", ls=":", lw=0.7)
                w = entry["w"]
                ax.plot(w["t_end"], w["mean_r"], color=C_WEEKLY, lw=1.0)
                ax.fill_between(w["t_end"], w["lower"], w["upper"],
                                color=C_WEEKLY_FILL, alpha=0.2, lw=0)
                ylab = "Weekly sliding R estimate"
                if "m" in entry:
                    m = entry["m"]
                    ax.plot(m["t_end"], m["mean_r"], color="navy", lw=1.0)
                    ax.fill_between(m["t_end"], m["lower"], m["upper"],
                                    color="dodgerblue", alpha=0.2, lw=0)
                    ylab = "Sliding R estimate"
                ax.plot(np.arange(1, 71), true[name], color=C_TRUE, ls="--",
                        lw=1.0)
                for x0, x1 in excl[kind][dt]:
                    ax.axvspan(x0, x1, 0.95, 2.25, color="#f5f5f5", zorder=3)
                style_ax(ax, xlim=(1, 70), ylim=(1, 2), ylabel=ylab)
                if row >= 2 and col == 0:
                    handles = [
                        plt.Line2D([], [], color=C_WEEKLY, lw=1.0,
                                   label="7 day window"),
                        plt.Line2D([], [], color="navy", lw=1.0,
                                   label=f"{dt} day window"),
                        plt.Line2D([], [], color=C_TRUE, ls="--", lw=1.0,
                                   label="True R")]
                    ax.legend(handles=handles, loc=legend_loc[kind],
                              frameon=False, fontsize=9)
        for row, letters in enumerate(("ABC", "DEF", "GHI", "JKL")):
            for j, letter in enumerate(letters):
                panel_label(axes[row, j], letter)
        fig.tight_layout(w_pad=1.6, h_pad=2.0)
        fig.savefig(f"{FIGDIR}/different_aggregations_{kind}.pdf")
        plt.close(fig)
        print(f"  -> {FIGDIR}/different_aggregations_{kind}.pdf")


# =========================================================================
# Study 6: mid-aggregation window variations (mid_window_variability.pdf)
# =========================================================================
# R alternates weekly (days 1-4 and 7 at R1, days 5-6 at R2, change points
# at days 4 and 6 of each week, seed 5); estimation from weekly aggregated
# data with weekly sliding and non-sliding (dt_out = 2) windows vs daily
# references. 4 panels: A/B non-sliding (ylim 0.7-2.1), C/D sliding
# (ylim 1-1.7). The true line is the paper's idealisation
# rep(5*R1, 2*R2) x8 on the weekly t_end grid (57 values).

def study_mid_agg(cache=None):
    print("=== mid_agg ===")
    if cache is None:
        cache = sim_cache()
    scenarios = [("1.25to1.5", cache["mid_1.25to1.5"], 1.25, 1.5),
                 ("1.5to1.25", cache["mid_1.5to1.25"], 1.5, 1.25)]
    out = {}
    for key, sims, a, b in scenarios:
        sl = summarize(run_agg(sims))
        sl_d = summarize(run_daily(sims))
        ns = summarize(run_agg(sims, dt_out=2))
        ns_d = summarize(run_daily(sims, t_start=np.arange(2, 70),
                                   t_end=np.arange(3, 71)))
        true_vec = np.tile(np.r_[np.repeat(a, 5), np.repeat(b, 2)], 8)
        true_vec = np.r_[true_vec[0], true_vec]          # 57 values
        out[key] = dict(sl=sl, sl_d=sl_d, ns=ns, ns_d=ns_d,
                        true=true_vec)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, (key, sims, a, b) in enumerate(scenarios):
        d = out[key]
        # top row: non-sliding
        ax = axes[0, j]
        r_plot(ax, {"daily": d["ns_d"], "weekly": d["ns"]},
               true=pd.Series(d["true"], index=d["sl"]["t_end"].to_numpy()),
               ylim=(0.7, 2.1), ylabel="Daily R estimate", hline=None)
        if j == 0:
            legend(ax, loc=(0.80, 0.90))
        # bottom row: weekly sliding
        ax = axes[1, j]
        r_plot(ax, {"daily": d["sl_d"], "weekly": d["sl"]},
               true=pd.Series(d["true"], index=d["sl"]["t_end"].to_numpy()),
               ylim=(1, 1.7))
    for row, letters in enumerate(("AB", "CD")):
        for j, letter in enumerate(letters):
            panel_label(axes[row, j], letter)
    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    fig.savefig(f"{FIGDIR}/mid_window_variability.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/mid_window_variability.pdf")


# =========================================================================
# Real-data supplementary studies (no simulations)
# =========================================================================

LOESS_COLORS = {0: "red", 1: "dodgerblue", 2: "magenta"}

# ---- Study: LOESS smoothing of naive weekly incidence (flu) ----

def study_loess(cache=None):
    """loess_smoothing.R: smooth naive weekly incidence (weekly agg / 7)
    with R loess() degrees 0/1/2, anchored at the end (opt1: days
    7/14/21/28/35) or middle (opt2: 3/10/17/24/31) of each week, then
    estimate_R (t_end shifted by +6 / +2 to align with the weekly
    aggregation). 3 figures: loess_reconstructions.pdf, loess_rt.pdf,
    loess_rt_bias.pdf."""
    import pyreadr
    from epiestim.loess import r_loess
    from epiestim.rds_reader import read_rds, list_to_dict

    print("=== loess ===")
    flu = pyreadr.read_r(f"{REPO}/real_data/daily_flu.rds")[None]
    incid = flu["Incidence"].to_numpy(float)
    weekly = aggregate_inc(incid, dt=7)
    naive = weekly / 7
    agg = estimate_R_agg(weekly, dt=7, dt_out=7, recon_opt="match",
                         mean_si=3.6, std_si=1.6)
    recon = agg["I"]

    # cross-check vs the paper's stored EM reconstruction
    stored_em = list_to_dict(read_rds(
        f"{REPO}/supplementary_analysis/flu/R_agg_sliding_result.rds"))
    stored_I = np.array([float(x) for x in stored_em["I"]])
    print(f"  recon vs stored EM I: max|diff| = "
          f"{np.max(np.abs(recon - stored_I)):.2e}")

    r_rep_sl = estimate_R(incid, mean_si=3.6, std_si=1.6)["R"]
    r_agg_sl = agg["R"]

    opt1 = np.arange(7.0, 36.0, 7.0)
    opt2 = np.arange(3.0, 32.0, 7.0)
    curves = {}
    for key, opt, days, off in (("opt1", opt1, np.arange(7.0, 36.0), 6),
                                ("opt2", opt2, np.arange(3.0, 32.0), 2)):
        curves[key] = {}
        for d in (0, 1, 2):
            y = r_loess(opt, naive, d, days)
            R = estimate_R(y, mean_si=3.6, std_si=1.6)["R"].copy()
            R["t_end"] = R["t_end"] + off
            curves[key][d] = (days, y, R)

    # Figure 1: incidence reconstructions (A opt1, B opt2)
    x = np.arange(1, 36)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for k, (ax, key) in enumerate(zip(axes, ("opt1", "opt2"))):
        ax.plot(x, incid, color="grey", lw=1.2)
        ax.plot(x, recon, color="black", lw=1.2)
        opt = opt1 if key == "opt1" else opt2
        ax.plot(opt, naive, "o", mfc="none", mec="black", ms=4, ls="none")
        for d in (0, 1, 2):
            days, y, _ = curves[key][d]
            ax.plot(days, y, color=LOESS_COLORS[d], lw=1.2)
        style_ax(ax, xlim=(1, 35), xlabel="" if k == 0 else "Day",
                 ylabel="Incidence")
        panel_label(ax, "AB"[k])
    axes[0].legend(handles=[
        plt.Line2D([], [], color="grey", lw=1.2, label="Reported daily"),
        plt.Line2D([], [], color="black", lw=1.2,
                   label="EM Reconstruction"),
        plt.Line2D([], [], color="red", lw=1.2, label="LOESS (d = 0)"),
        plt.Line2D([], [], color="dodgerblue", lw=1.2, label="LOESS (d = 1)"),
        plt.Line2D([], [], color="magenta", lw=1.2, label="LOESS (d = 2)"),
    ], loc=(0.42, 0.85), frameon=False, fontsize=8)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(f"{FIGDIR}/loess_reconstructions.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/loess_reconstructions.pdf")

    # Figure 2: R estimates + CrI (A opt1, B opt2)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for k, (ax, key) in enumerate(zip(axes, ("opt1", "opt2"))):
        ax.axhline(1, color="black", ls=":", lw=1)
        ax.plot(r_rep_sl["t_end"], r_rep_sl["Mean(R)"], color="grey",
                lw=1.2)
        ax.fill_between(r_rep_sl["t_end"], r_rep_sl["Quantile.0.025(R)"],
                        r_rep_sl["Quantile.0.975(R)"], color="grey",
                        alpha=0.3, lw=0)
        ax.plot(r_agg_sl["t_end"], r_agg_sl["Mean(R)"], color="black",
                lw=1.2)
        ax.fill_between(r_agg_sl["t_end"], r_agg_sl["Quantile.0.025(R)"],
                        r_agg_sl["Quantile.0.975(R)"], color="black",
                        alpha=0.3, lw=0)
        for d in (0, 1, 2):
            _, _, R = curves[key][d]
            c = LOESS_COLORS[d]
            ax.plot(R["t_end"], R["Mean(R)"], color=c, lw=1.2)
            ax.fill_between(R["t_end"], R["Quantile.0.025(R)"],
                            R["Quantile.0.975(R)"], color=c, alpha=0.3, lw=0)
        style_ax(ax, ylim=(0.5, 2), xlabel="" if k == 0 else "Day",
                 ylabel="R estimate")
        panel_label(ax, "AB"[k])
    axes[0].legend(handles=[
        plt.Line2D([], [], color="grey", lw=1.2, label="Reported daily"),
        plt.Line2D([], [], color="black", lw=1.2,
                   label="EM reconstruction"),
        plt.Line2D([], [], color="red", lw=1.2, label="LOESS (d = 0)"),
        plt.Line2D([], [], color="dodgerblue", lw=1.2, label="LOESS (d = 1)"),
        plt.Line2D([], [], color="magenta", lw=1.2, label="LOESS (d = 2)"),
    ], loc=(0.9, 0.8), frameon=False, fontsize=9)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(f"{FIGDIR}/loess_rt.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/loess_rt.pdf")

    # Figure 3: bias = R(method) - R(reported), per shared t_end
    rep = dict(zip(r_rep_sl["t_end"].to_numpy(),
                   r_rep_sl["Mean(R)"].to_numpy()))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for k, (ax, key, tmin, tmax) in enumerate(zip(axes, ("opt1", "opt2"),
                                                  (15, 14), (35, 31))):
        ax.axhline(0, color="lightgrey", ls="--", lw=1)
        te = r_agg_sl["t_end"].to_numpy()
        m = (te >= tmin) & (te <= tmax)
        ax.plot(te[m], r_agg_sl["Mean(R)"].to_numpy()[m]
                - np.array([rep[t] for t in te[m]]), color="black", lw=1.2)
        for d in (0, 1, 2):
            _, _, R = curves[key][d]
            R = R.loc[R["t_end"].between(tmin, tmax)]
            ax.plot(R["t_end"].to_numpy(),
                    R["Mean(R)"].to_numpy()
                    - np.array([rep[t] for t in R["t_end"]]),
                    color=LOESS_COLORS[d], lw=1.2)
        style_ax(ax, xlim=(14, 35), ylim=(-0.6, 0.6),
                 xlabel="" if k == 0 else "Day",
                 ylabel="R (reported data) - \n R (reconstructed data)")
        panel_label(ax, "AB"[k])
    axes[0].legend(handles=[
        plt.Line2D([], [], color="black", lw=1.2, label="EM reconstruction"),
        plt.Line2D([], [], color="red", lw=1.2, label="LOESS (d = 0)"),
        plt.Line2D([], [], color="dodgerblue", lw=1.2, label="LOESS (d = 1)"),
        plt.Line2D([], [], color="magenta", lw=1.2, label="LOESS (d = 2)"),
    ], loc=(0.7, 0.8), frameon=False, fontsize=9)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(f"{FIGDIR}/loess_rt_bias.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/loess_rt_bias.pdf")


# ---- Study: relative difference in reporting by weekday ----

GGPLOT7 = ["#F8766D", "#C49A00", "#53B400", "#00C094", "#00B6EB",
           "#A58AFF", "#FB61D7"]          # ggplot2 default 7-level fill
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def _weekday_rel(rds, week_shift=None, extra_rm=()):
    """incidence_weekday.R: keep only complete ISO weeks, then per day
    compute (I - weekly mean) / weekly mean. Returns (weekday idx, rel)."""
    import pyreadr
    df = pyreadr.read_r(f"{REPO}/{rds}")[None]
    dates = pd.to_datetime(df["Date"], dayfirst=True)
    wd = dates.dt.weekday.to_numpy()                     # 0 = Monday
    wk = dates.dt.isocalendar().week.to_numpy().astype(int)
    if week_shift is not None:        # 1-based R slices [lo:hi] -> 0-based
        wk[week_shift[0]:week_shift[1]] += 53
    counts = pd.Series(wk).value_counts()
    rm = set(counts.index[counts.to_numpy() != 7]) | set(extra_rm)
    keep = np.array([w not in rm for w in wk])
    wd, wk = wd[keep], wk[keep]
    inc = df["Incidence"].to_numpy(float)[keep]
    means = {w: inc[wk == w].mean() for w in np.unique(wk)}
    m = np.array([means[w] for w in wk])
    return wd, (inc - m) / m


def study_weekday(cache=None):
    print("=== weekday ===")
    data = [
        _weekday_rel("real_data/daily_flu.rds"),
        _weekday_rel("real_data/daily_covid_cases.rds",
                     week_shift=(318, 679), extra_rm=(9,)),
        _weekday_rel("real_data/daily_covid_deaths.rds",
                     week_shift=(371, 672)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5))
    for k, (ax, (wd, rel)) in enumerate(zip(axes, data)):
        ax.axhline(0, color="grey", ls="--", lw=1)
        rng = np.random.default_rng(5)   # ggplot2 position_jitter(seed = 5)
        for i in range(7):
            vals = rel[wd == i]
            ax.plot([i - 0.375, i + 0.375], [vals.mean()] * 2,
                    color="black", lw=2, zorder=5)      # crossbar of means
            off = rng.uniform(-0.1, 0.1, len(vals))
            ax.scatter(i + off, vals, s=12, facecolor=GGPLOT7[i],
                       edgecolor="black", linewidth=0.5, zorder=3)
        style_ax(ax, xlim=(-0.6, 6.6),
                 ylabel="Relative difference from weekly mean",
                 xlabel="Weekday" if k == 2 else "")
        ax.set_xticks(range(7))
        ax.set_xticklabels(WEEKDAYS)
        ax.tick_params(axis="x", length=0)
        panel_label(ax, "ABC"[k])
    fig.tight_layout(h_pad=2.0)
    fig.savefig(f"{FIGDIR}/incidence_weekday.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/incidence_weekday.pdf")


# ---- Study: zika (Rio de Janeiro) workaround vs EM ----

def study_zika(cache=None):
    """zika.R: weekly incidence over 22 weeks; 'old' workaround = 2-week
    sliding windows with the Ferguson 2016 SI (mean 20, sd 7.4) rescaled
    to the weekly timescale (/7), each estimate repeated 7x on the daily
    grid (first 14 days NA); 'new' = estimate_R_agg (dt = dt_out = 7,
    match) with the daily-scale SI (first 20 days NA)."""
    print("=== zika ===")
    dat = pd.read_csv(f"{REPO}/supplementary_analysis/zika/"
                      f"Brazil_RioJaneiro.csv")
    incid = dat["incidence"].to_numpy(float)
    nweeks = len(incid)
    old = estimate_R(incid, mean_si=20.0 / 7, std_si=7.4 / 7,
                     t_start=np.arange(2, nweeks),
                     t_end=np.arange(3, nweeks + 1))["R"]
    new = estimate_R_agg(incid, dt=7, dt_out=7, recon_opt="match",
                         mean_si=20.0, std_si=7.4)["R"]
    assert len(new) == nweeks * 7 - 20 and new["t_end"].min() == 21

    overall = np.arange(1, nweeks * 7 + 1)
    series = {}
    for name, R, na, rep in (("old", old, 14, 7), ("new", new, 20, 1)):
        for col in ("r", "upper", "lower"):
            series.setdefault(name, {})[col] = np.full(nweeks * 7, np.nan)
        series[name]["r"][na:] = np.repeat(R["Mean(R)"].to_numpy(), rep)
        series[name]["upper"][na:] = np.repeat(
            R["Quantile.0.975(R)"].to_numpy(), rep)
        series[name]["lower"][na:] = np.repeat(
            R["Quantile.0.025(R)"].to_numpy(), rep)

    t = overall[overall > 35]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(1, color="black", ls=":", lw=1)
    for name, col, fill, fa in (("old", "grey", "grey", 0.18),
                                ("new", "darkgreen", "#2E8B57", 0.15)):
        ax.plot(t, series[name]["r"][overall > 35], color=col, lw=1.2)
        ax.fill_between(t, series[name]["lower"][overall > 35],
                        series[name]["upper"][overall > 35], color=fill,
                        alpha=fa, lw=0)
    style_ax(ax, ylim=(0, 4), xlabel="Time", ylabel="R estimate")
    x_dates = pd.Timestamp("2015-11-14")
    ax.set_xticks([60, 90, 120, 150])
    ax.set_xticklabels([(x_dates + pd.Timedelta(days=int(b) - 1))
                        .strftime("%d %b %y") for b in (60, 90, 120, 150)])
    ax.legend(handles=[
        plt.Line2D([], [], color="grey", lw=1.2, label="Workaround method"),
        plt.Line2D([], [], color="darkgreen", lw=1.2, label="EM algorithm"),
    ], loc=(0.8, 0.85), frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/zika_rt.pdf")
    plt.close(fig)
    print(f"  -> {FIGDIR}/zika_rt.pdf")


# ---- Study: appendix R-estimate figures (flu / covid cases / deaths) ----

APPENDIX = {
    "flu": dict(start="2009-12-11", breaks=[10, 20, 30],
                xlim_rt=(9, 35), xlim_b=(1, 35), ylim_rt=(0, 3.75),
                inc_ylab="Daily incidence by \n date of presentation at clinic",
                inc_log_ylab="Daily incidence by date \n of presentation "
                             "at clinic (log scale)",
                inc_legend=(0.45, 0.87), rt_legend=(0.2, 0.75),
                rep_fill=("grey", 0.6)),
    "covid_cases": dict(start="2020-02-21", breaks=[1, 200, 400, 600],
                        xlim_rt=(5, 679), xlim_b=(1, 679), ylim_rt=(0, 4),
                        inc_ylab="Daily incidence \n by date of specimen",
                        inc_log_ylab="Daily incidence by \n date of specimen "
                                     "(log scale)",
                        inc_legend=(0.73, 0.87), rt_legend=(0.85, 0.75),
                        rep_fill=("grey", 0.0)),
    "covid_deaths": dict(start="2020-03-02", breaks=[1, 200, 400, 600],
                         xlim_rt=(5, 679), xlim_b=(1, 672), ylim_rt=(0, 4),
                         inc_ylab="Daily incidence \n by date of death",
                         inc_log_ylab="Daily incidence by \n date of death "
                                      "(log scale)",
                         inc_legend=(0.73, 0.77), rt_legend=(0.85, 0.75),
                         rep_fill=("darkgrey", 0.7)),
}


def _appendix_res(name):
    """The four main-analysis estimations (reported/aggregated x
    sliding/daily windows), recomputed with the validated engine and
    pickled (same machinery as analyze() in reproduce_paper.py)."""
    from reproduce_paper import analyze, DATASETS
    return _cached(f"analyze_{name}",
                   lambda: analyze(name, DATASETS[name]))


def _appendix_dates(cfg):
    start = pd.Timestamp(cfg["start"])
    return [f"{(start + pd.Timedelta(days=int(b) - 1)).strftime('%d %b %y')}"
            for b in cfg["breaks"]]


def _rt_figure(name, cfg, res, outname):
    """covid_{cases,deaths}_rt.pdf / flu_rt.pdf: A daily (non-sliding)
    and B weekly sliding R estimates, reported vs EM-reconstructed."""
    labels = _appendix_dates(cfg)
    rep_d = res["r_rep_d"].iloc[5:32] if name == "flu" else res["r_rep_d"].iloc[2:]
    rep_s = res["r_rep_sl"].iloc[6:28] if name == "flu" else res["r_rep_sl"].iloc[6:]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5))
    panels = ((rep_d, res["r_agg_d"], "Daily R estimate"),
              (rep_s, res["r_agg_sl"], "Weekly sliding R estimate"))
    for k, (ax, (rep, agg, ylab)) in enumerate(zip(axes, panels)):
        ax.axhline(1, color="black", ls=":", lw=1)
        ax.plot(rep["t_end"], rep["Mean(R)"], color="darkgrey", lw=1.2)
        rf = cfg["rep_fill"]
        ax.fill_between(rep["t_end"], rep["Quantile.0.025(R)"],
                        rep["Quantile.0.975(R)"], color=rf[0],
                        alpha=rf[1] * 0.3, lw=0)
        ax.plot(agg["t_end"], agg["Mean(R)"], color="darkgreen", lw=1.2)
        ax.fill_between(agg["t_end"], agg["Quantile.0.025(R)"],
                        agg["Quantile.0.975(R)"], color="#2E8B57",
                        alpha=0.15, lw=0)
        style_ax(ax, xlim=cfg["xlim_rt"], ylim=cfg["ylim_rt"],
                 xlabel="" if k == 0 else "Date", ylabel=ylab)
        ax.set_xticks(cfg["breaks"])
        ax.set_xticklabels(labels)
        panel_label(ax, "AB"[k])
    axes[0].legend(handles=[
        plt.Line2D([], [], color="darkgrey", lw=1.2, label="Reported data"),
        plt.Line2D([], [], color="darkgreen", lw=1.2,
                   label="Reconstructed data"),
    ], loc=cfg["rt_legend"], frameon=False, fontsize=10)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(f"{FIGDIR}/{outname}")
    plt.close(fig)


def _bias_figure(name, cfg, res, outname):
    """covid{,_deaths}_rt_bias.pdf / flu_rt_bias.pdf: A incidence,
    B log10 incidence, C daily R difference, D weekly sliding R
    difference (reported - reconstructed)."""
    labels = _appendix_dates(cfg)
    rep_d = res["r_rep_d"].iloc[5:32] if name == "flu" else res["r_rep_d"].iloc[2:]
    rep_s = res["r_rep_sl"].iloc[6:28] if name == "flu" else res["r_rep_sl"].iloc[6:]
    diff_d = (rep_d["Mean(R)"].to_numpy()
              - res["r_agg_d"]["Mean(R)"].to_numpy())
    diff_s = (rep_s["Mean(R)"].to_numpy()
              - res["r_agg_sl"]["Mean(R)"].to_numpy())
    x = np.arange(1, len(res["incid"]) + 1)

    fig, axes = plt.subplots(4, 1, figsize=(7, 9))
    ax = axes[0]
    ax.plot(x, res["incid"], color="darkgrey", lw=1.2, alpha=0.8)
    ax.plot(x, res["recon_inc"], color="#297a4d", lw=1.2)
    style_ax(ax, xlim=cfg["xlim_b"], xlabel="", ylabel=cfg["inc_ylab"])
    ax.set_xticks(cfg["breaks"])
    ax.set_xticklabels(labels)
    ax.legend(handles=[
        plt.Line2D([], [], color="darkgrey", lw=1.2, alpha=0.8,
                   label="Reported"),
        plt.Line2D([], [], color="#297a4d", lw=1.2, label="Reconstructed"),
    ], loc=cfg["inc_legend"], frameon=False, fontsize=10)
    panel_label(ax, "A")

    ax = axes[1]
    ax.plot(x, res["incid"], color="darkgrey", lw=1.2, alpha=0.8)
    ax.plot(x, res["recon_inc"], color="#297a4d", lw=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    style_ax(ax, xlim=cfg["xlim_b"], xlabel="", ylabel=cfg["inc_log_ylab"])
    ax.set_xticks(cfg["breaks"])
    ax.set_xticklabels(labels)
    panel_label(ax, "B")

    diff_panels = (
        (res["r_agg_d"]["t_end"].to_numpy(), diff_d,
         "Daily R (reported data) \n - Daily R (reconstructed data)", ""),
        (res["r_agg_sl"]["t_end"].to_numpy(), diff_s,
         "Weekly R (reported data) \n - Weekly R (reconstructed data)",
         "Date"))
    for k, (ax, (day, diff, ylab, xlab)) in enumerate(
            zip(axes[2:], diff_panels)):
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.plot(day, diff, color="black", lw=1.2)
        style_ax(ax, xlim=cfg["xlim_b"], xlabel=xlab, ylabel=ylab)
        ax.set_xticks(cfg["breaks"])
        ax.set_xticklabels(labels)
        panel_label(ax, "CD"[k])
    fig.tight_layout(h_pad=1.6)
    fig.savefig(f"{FIGDIR}/{outname}")
    plt.close(fig)


def study_appendix_rt(cache=None):
    print("=== appendix_rt ===")
    outnames = {
        "flu": ("flu_rt.pdf", "flu_rt_bias.pdf"),
        "covid_cases": ("covid_cases_rt.pdf", "covid_rt_bias.pdf"),
        "covid_deaths": ("covid_deaths_rt.pdf", "covid_deaths_rt_bias.pdf"),
    }
    for name in ("flu", "covid_cases", "covid_deaths"):
        cfg = APPENDIX[name]
        res = _appendix_res(name)
        rt, bias = outnames[name]
        _rt_figure(name, cfg, res, rt)
        _bias_figure(name, cfg, res, bias)
        print(f"  -> {FIGDIR}/{rt}, {FIGDIR}/{bias}")


# =========================================================================
# dispatch
# =========================================================================

STUDIES = {
    "constant_rt": study_constant_rt,
    "tv_sudden": study_tv_sudden,
    "tv_gradual": study_tv_gradual,
    "weekend_effects": study_weekend_effects,
    "diff_aggs": study_diff_aggs,
    "mid_agg": study_mid_agg,
    "loess": study_loess,
    "weekday": study_weekday,
    "zika": study_zika,
    "appendix_rt": study_appendix_rt,
}

DEFAULT_STUDIES = ["constant_rt", "tv_sudden", "tv_gradual",
                   "weekend_effects", "diff_aggs", "mid_agg",
                   "loess", "weekday", "zika", "appendix_rt"]


def main():
    args = sys.argv[1:] or DEFAULT_STUDIES
    unknown = [a for a in args if a not in STUDIES]
    if unknown:
        print(f"unknown study: {unknown} (available: {', '.join(STUDIES)})")
        sys.exit(1)
    os.makedirs(FIGDIR, exist_ok=True)
    cache = sim_cache()          # one shared simulated dataset per run
    for name in args:
        STUDIES[name](cache)


if __name__ == "__main__":
    main()
