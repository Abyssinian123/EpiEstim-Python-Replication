# -*- coding: utf-8 -*-
"""
Reproduce Figures 2-4 and the headline statistics of

  Nash RK, Bhatt S, Cori A, Nouvellet P (2023). Estimating the epidemic
  reproduction number from temporally aggregated incidence data.
  PLoS Comput Biol 19(8): e1011439.

Everything is computed from scratch with the validated Python port
(epiestim.py); no stored R results are read. The figures mirror the paper's
ggplot code in EM_EpiEstim_Nash2023-main/main_analysis/{flu,covid_cases,
covid_deaths}.R, including its colours, panel layout, and axis limits.

Usage:  python reproduce_paper.py            # figures/Figure_{2,3,4}.pdf
                                             # + printed stats tables
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from epiestim import estimate_R, estimate_R_agg, aggregate_inc  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "figures")

# ---- dataset-specific settings (all taken from the paper's R scripts) -----
DATASETS = {
    "flu": dict(
        rds="real_data/daily_flu.rds", mean_si=3.6, std_si=1.6,
        start="2009-12-11", breaks=[10, 20, 30], figure="Figure_2.pdf",
        ylab="Daily incidence by presentation at clinic",
        xlim=(0, 35), figsize=(11, 7),
        corr1_lim=((0.6, 1.5), (0.6, 1.5)),   # panel C xlim, ylim
        corr2_lim=((0, 3.6), (0, 3.6)),       # panel E
        unc1_lim=((0.028, 0.049), (0.028, 0.049)),   # panel D
        unc2_lim=((0, 0.18), (0, 0.18)),      # panel F
        b_ylim=None,
        r2_text_pos=dict(c=(1.3, 0.8), d=(0.044, 0.0315),
                         e=(3.1, 0.5), f=(0.15, 0.027)),
        r2_dec=dict(c=2, d=2, e=2, f=2)),
    "covid_cases": dict(
        rds="real_data/daily_covid_cases.rds", mean_si=6.3, std_si=4.2,
        start="2020-02-21", breaks=[200, 400, 600], figure="Figure_3.pdf",
        ylab="Daily incidence by date of specimen",
        xlim=(0, 679), figsize=(12, 7),
        corr1_lim=((0.4, 3.2), (0.4, 3.2)),
        corr2_lim=((0.4, 3.2), (0.4, 3.2)),
        unc1_lim=((0, 0.125), (0, 0.125)),
        unc2_lim=((0, 0.18), (0, 0.18)),
        b_ylim=(0, 0.5),
        r2_text_pos=dict(c=(2.1, 1.4), d=(0.1025, 0.03),
                         e=(2.9, 1.6), f=(0.15, 0.045)),
        r2_dec=dict(c=2, d=3, e=2, f=2)),
    "covid_deaths": dict(
        rds="real_data/daily_covid_deaths.rds", mean_si=6.3, std_si=4.2,
        start="2020-03-02", breaks=[200, 400, 600], figure="Figure_4.pdf",
        ylab="Daily incidence by date of death",
        xlim=(0, 672), figsize=(12, 7),
        corr1_lim=((0.45, 3.0), (0.45, 3.0)),
        corr2_lim=((0.45, 3.0), (0.45, 3.0)),
        unc1_lim=((0, 0.65), (0, 0.65)),
        unc2_lim=((0, 1.4), (0, 1.4)),
        b_ylim=(0, 0.5),
        r2_text_pos=dict(c=(2.2, 0.7), d=(0.56, 0.18),
                         e=(2.6, 0.65), f=(1.15, 0.4)),
        r2_dec=dict(c=2, d=3, e=2, f=2)),
}

# colours copied from the paper's ggplot calls
C_REPORTED, C_RECON = "darkgrey", "#297a4d"
C_WEEKLY_R, C_DAILY_R = "#e35874", "#ff6600"
PHASE_COLORS = {"early": "#B3E5FC", "mid": "dodgerblue", "late": "#01579B"}


def signif(x, d=5):
    """R's signif(): round to d significant digits (half-to-even)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    nz = x != 0
    dec = (d - 1 - np.floor(np.log10(np.abs(x[nz])))).astype(int)
    out[nz] = [round(v, int(dd)) for v, dd in zip(x[nz], dec)]
    return out


def corr2(a, b):
    """R^2 = squared Pearson correlation (as in the paper's R scripts)."""
    return np.corrcoef(a, b)[0, 1] ** 2


def phase_interval(T, name):
    """The inc_dat$interval vector of the paper's R scripts (day 1 .. T).
    day1 = 9 (first daily-window t_end), day2 = 14 (first weekly-window t_end)."""
    day1, day2 = 9, 14
    if name == "flu":
        int_ = len(range(day2, T + 1)) / 3           # 22/3 -> each = 7
        each = int(int_)                             # R rep() floors 'each'
        v = (["no_est"] * (day1 - 1) + ["early_daily"] * (day2 - day1)
             + sum(([p] * each for p in ["early", "mid", "late"]), []))
        return v + ["late"]                          # R: c(..., 'late')
    if name == "covid_cases":
        each = len(range(day2, T + 1)) // 3          # 666/3 = 222 exact
        return (["no_est"] * (day1 - 1) + ["early_daily"] * (day2 - day1)
                + sum(([p] * each for p in ["early", "mid", "late"]), []))
    # covid_deaths: round(int) and drop the overflow (intervals[-(ndays+1)])
    each = round(len(range(day2, T + 1)) / 3)        # 220
    v = (["no_est"] * (day1 - 1) + ["early_daily"] * (day2 - day1)
         + sum(([p] * each for p in ["early", "mid", "late"]), []))
    return v[:T]


def phase_of(t_end, interval):
    """Phase label of a window ending on 1-based day t_end (daily windows'
    'early_daily' recoded to 'early', as in the R scripts)."""
    p = np.array([interval[t - 1] for t in t_end])
    p[p == "early_daily"] = "early"
    return p


def load_incidence(rds):
    import pyreadr
    df = pyreadr.read_r(f"{REPO}/{rds}")[None]
    return df["Incidence"].to_numpy(dtype=float)


def analyze(name, cfg):
    """Run all four estimation setups of the paper and collect panel data."""
    incid = load_incidence(cfg["rds"])
    T = len(incid)
    weekly = aggregate_inc(incid, dt=7)

    r_rep_sl = estimate_R(incid, mean_si=cfg["mean_si"], std_si=cfg["std_si"])["R"]
    t_d = np.arange(2, T)                          # daily (2-day) windows
    r_rep_d = estimate_R(incid, mean_si=cfg["mean_si"], std_si=cfg["std_si"],
                         t_start=t_d, t_end=t_d + 1)["R"]
    r_agg_sl = estimate_R_agg(weekly, dt=7, dt_out=7, recon_opt="match",
                              mean_si=cfg["mean_si"], std_si=cfg["std_si"])["R"]
    agg = estimate_R_agg(weekly, dt=7, dt_out=2, recon_opt="match",
                         mean_si=cfg["mean_si"], std_si=cfg["std_si"])
    r_agg_d = agg["R"]
    recon_inc = agg["I"]
    interval = phase_interval(T, name)

    # ---- panel C / D: weekly sliding scatter (aligned on t_end) ----
    ts = r_agg_sl["t_end"].to_numpy()
    i_rep = r_rep_sl["t_end"].to_numpy()
    if name == "flu":
        rep_sel = r_rep_sl.iloc[6:28]              # R: [7:28] -> t_end 14..35
    else:                                          # covid: drop t_end <= 29
        rep_sel = r_rep_sl.loc[r_rep_sl["t_end"] >= 30]
        ts = ts[ts >= 30]
        # keep only rows whose t_end survives (covid: t_end 30..T)
    i_rep = rep_sel["t_end"].to_numpy()
    if name == "flu":
        ts = ts[:len(i_rep)]                       # agg t_end 14..35
    weekly = dict(
        agg=r_agg_sl.loc[r_agg_sl["t_end"].isin(ts)].reset_index(drop=True),
        rep=rep_sel.reset_index(drop=True),
        col=phase_of(ts, interval))

    # ---- panel E / F: daily scatter ----
    td = r_agg_d["t_end"].to_numpy()
    if name == "flu":                              # R: rep[6:32] -> t_end 8..34
        rep_sel = r_rep_d.iloc[5:32].reset_index(drop=True)
    else:                                          # covid: drop t_end <= 29
        rep_sel = r_rep_d.loc[r_rep_d["t_end"] >= 30].reset_index(drop=True)
        td = td[td >= 30]
    if name == "flu":
        td = td[:len(rep_sel)]
    daily = dict(
        agg=r_agg_d.loc[r_agg_d["t_end"].isin(td)].reset_index(drop=True),
        rep=rep_sel,
        col=phase_of(td, interval))

    # ---- panel B: squared error of the lm fit ----
    # weekly sliding: R lm(agg_sl ~ rep_sl[7:n]) -> aligned t_end 14..T
    a = r_agg_sl["Mean(R)"].to_numpy()
    b = r_rep_sl["Mean(R)"].to_numpy()
    if name == "flu":
        a, b = a[:len(a)], b[6:28]                 # agg t_end 14..35
    else:
        m = r_agg_sl["t_end"].to_numpy() >= 14
        a, b = a[m], b[np.isin(r_rep_sl["t_end"], r_agg_sl["t_end"][m])]
    res1 = signif(_lm_resid(b, a), 5)
    sq_sl = res1 ** 2
    if name == "flu":
        x_sl = r_agg_sl["t_end"].to_numpy()          # t_end 14..35
    else:
        x_sl = r_agg_sl["t_end"].to_numpy()[r_agg_sl["t_end"] >= 14]

    # daily: paper's scripts pair agg t_end 9..T with rep t_end 8..T-1 for flu
    # (R: rep[6:32]) and intended an aligned pairing for covid (their [3:n]
    # index makes lm() error as committed; aligned is used here and matches
    # the published figure's scatter panels).
    a = r_agg_d["Mean(R)"].to_numpy()
    if name == "flu":
        b = r_rep_d["Mean(R)"].to_numpy()[5:32]    # t_end 8..34
    else:
        te = r_agg_d["t_end"].to_numpy()
        b = r_rep_d["Mean(R)"].to_numpy()[np.isin(r_rep_d["t_end"], te)]
    res2 = signif(_lm_resid(b, a), 5)
    sq_d = res2 ** 2
    x_d = r_agg_d["t_end"].to_numpy()[:len(sq_d)]

    return dict(name=name, cfg=cfg, incid=incid, recon_inc=recon_inc,
                interval=interval, r_rep_sl=r_rep_sl, r_rep_d=r_rep_d,
                r_agg_sl=r_agg_sl, r_agg_d=r_agg_d,
                weekly=weekly, daily=daily,
                sq_sl=sq_sl, x_sl=x_sl, sq_d=sq_d, x_d=x_d)


def _lm_resid(x, y):
    """Residuals of y ~ x (lm with intercept), x on the x-axis (reported)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return y - (intercept + slope * x)


def date_labels(cfg):
    start = pd.Timestamp(cfg["start"])
    return [f"{(start + pd.Timedelta(days=b - 1)).strftime('%d %b %y')}"
            for b in cfg["breaks"]]


def draw_figure(res):
    """Six-panel figure mirroring the paper's plot_grid layout:
    row 1: A incidence, C mean weekly R, E mean daily R
    row 2: B squared error, D weekly CrI width, F daily CrI width"""
    cfg = res["cfg"]
    T = len(res["incid"])
    fig, axes = plt.subplots(2, 3, figsize=cfg["figsize"],
                             gridspec_kw={"width_ratios": [1.25, 1, 1]
                                          if "covid" in res["name"] else [1, 1, 1]})
    axA, axC, axE = axes[0]
    axB, axD, axF = axes[1]
    lbl = date_labels(cfg)
    for ax, panel in zip((axA, axC, axE, axB, axD, axF), "ACEBDF"):
        ax.text(-0.09, 1.06, panel, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top")

    # ---- panel A: incidence + phase strip ----
    x = np.arange(1, T + 1)
    # phase strip: thin band under the data, coloured by phase (the paper's
    # polygons sit at the bottom of the incidence panel)
    iv = np.array(res["interval"])
    ymax = res["incid"].max()
    axA.plot(x, res["incid"], color=C_REPORTED, alpha=0.8, lw=1,
             label="Reported")
    axA.plot(x, res["recon_inc"], color=C_RECON, lw=1, label="Reconstructed")
    strip_y = (-0.085 * ymax, -0.02 * ymax)
    for ph, col in [("early", PHASE_COLORS["early"]),
                    ("mid", PHASE_COLORS["mid"]),
                    ("late", PHASE_COLORS["late"])]:
        m = iv == ph
        if m.any():
            days = np.where(m)[0] + 1
            axA.fill_between([days[0], days[-1] + 1], *strip_y, color=col,
                             alpha=0.75, lw=0)
    axA.set_ylim(strip_y[0], ymax * 1.04)
    axA.set_xlim(*cfg["xlim"])
    axA.set_ylabel(cfg["ylab"])
    axA.set_xticks(cfg["breaks"])
    axA.set_xticklabels(lbl)
    axA.legend(loc="upper center", frameon=False, ncol=2)
    axA.grid(False)

    # ---- panel B: squared error ----
    axB.plot(res["x_sl"], res["sq_sl"], color=C_WEEKLY_R, lw=1,
             label="Weekly sliding R")
    axB.plot(res["x_d"], res["sq_d"], color=C_DAILY_R, lw=1, alpha=0.3,
             label="Daily R")
    if cfg["b_ylim"]:
        axB.set_ylim(*cfg["b_ylim"])
    axB.set_xlim(*cfg["xlim"])
    axB.set_ylabel("Squared error of R estimates\n"
                   "using reported and reconstructed data")
    axB.set_xlabel("Date")
    axB.set_xticks(cfg["breaks"])
    axB.set_xticklabels(lbl)
    axB.legend(loc="upper right", frameon=False)
    axB.grid(False)

    # ---- panels C/D/E/F: scatters ----
    def scatter(ax, dat, xcol, ycol, panel, show_legend=False):
        with_errors = panel in "CE"   # R: error bars only on the mean-R panels
        for ph, col in PHASE_COLORS.items():
            m = dat["col"] == ph
            if not m.any():
                continue
            kw = dict(fmt="o", ms=2.5, color=col, alpha=0.7, lw=0.4,
                      markeredgewidth=0, label=ph.capitalize())
            if with_errors:
                ax.errorbar(dat["rep"][xcol][m], dat["agg"][ycol][m],
                            yerr=[dat["agg"][ycol][m] - dat["agg"]["Quantile.0.025(R)"][m],
                                  dat["agg"]["Quantile.0.975(R)"][m] - dat["agg"][ycol][m]],
                            xerr=[dat["rep"][xcol][m] - dat["rep"]["Quantile.0.025(R)"][m],
                                  dat["rep"]["Quantile.0.975(R)"][m] - dat["rep"][xcol][m]],
                            **kw)
            else:
                kw.pop("fmt")
                ax.plot(dat["rep"][xcol][m], dat["agg"][ycol][m], marker="o",
                        **kw)
        # x = y dashed; lm fit solid; R = 1 dotted lines (C/E only)
        lo = min(dat["rep"][xcol].min(), dat["agg"][ycol].min())
        hi = max(dat["rep"][xcol].max(), dat["agg"][ycol].max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="black", alpha=0.5, lw=1)
        s, i = np.polyfit(dat["rep"][xcol], dat["agg"][ycol], 1)
        xx = np.linspace(lo, hi, 50)
        ax.plot(xx, i + s * xx, color="black", lw=0.7)
        if panel in "CE":
            ax.axhline(1, ls=":", color="black", lw=0.8)
            ax.axvline(1, ls=":", color="black", lw=0.8)
        r2 = corr2(dat["agg"][ycol], dat["rep"][xcol])
        dec = cfg["r2_dec"][panel.lower()]
        ax.text(*cfg["r2_text_pos"][panel.lower()],
                f"$R^2 = {r2:.{dec}f}$", fontsize=10)
        if show_legend:
            ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(False)
        return r2

    r2c = scatter(axC, res["weekly"], "Mean(R)", "Mean(R)", "C",
                  show_legend=True)
    xlim, ylim = cfg["corr1_lim"]
    axC.set_xlim(*xlim)
    axC.set_ylim(*ylim)
    axC.set_ylabel("Mean weekly R (reconstructed data)")
    axC.set_xlabel("Mean weekly R (reported data)")

    r2e = scatter(axE, res["daily"], "Mean(R)", "Mean(R)", "E")
    xlim, ylim = cfg["corr2_lim"]
    axE.set_xlim(*xlim)
    axE.set_ylim(*ylim)
    axE.set_ylabel("Mean daily R (reconstructed data)")
    axE.set_xlabel("Mean daily R (reported data)")

    r2d = scatter(axD, res["weekly"], "width", "width", "D")
    xlim, ylim = cfg["unc1_lim"]
    axD.set_xlim(*xlim)
    axD.set_ylim(*ylim)
    axD.set_ylabel("95% CrI width for weekly R (reconstructed data)")
    axD.set_xlabel("95% CrI width for weekly R (reported data)")

    r2f = scatter(axF, res["daily"], "width", "width", "F")
    xlim, ylim = cfg["unc2_lim"]
    axF.set_xlim(*xlim)
    axF.set_ylim(*ylim)
    axF.set_ylabel("95% CrI width for daily R (reconstructed data)")
    axF.set_xlabel("95% CrI width for daily R (reported data)")

    for ax in (axA, axB, axC, axD, axE, axF):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(w_pad=2.5, h_pad=2.8)
    out = f"{FIGDIR}/{cfg['figure']}"
    fig.savefig(out)
    plt.close(fig)
    return dict(c=r2c, d=r2d, e=r2e, f=r2f), out


def classify(R):
    """Trend classification: increasing (Q2.5 > 1), declining (Q97.5 < 1),
    uncertain otherwise (S1 Appendix section 2e)."""
    qlo, qhi = R["Quantile.0.025(R)"].to_numpy(), R["Quantile.0.975(R)"].to_numpy()
    return np.where(qlo > 1, "inc", np.where(qhi < 1, "dec", "unc"))


def classification_agreement(res):
    """S1 Appendix Figure S8: agreement between reported and reconstructed
    classifications, aligned on t_end. Covid datasets exclude the first
    30 days (t_end < 30); flu uses all aligned windows."""
    name = res["name"]
    min_end = 30 if "covid" in name else 0
    out = {}
    for tag, a, b in [("weekly", res["r_rep_sl"], res["r_agg_sl"]),
                      ("daily", res["r_rep_d"], res["r_agg_d"])]:
        common = np.intersect1d(a["t_end"], b["t_end"])
        common = common[common >= min_end]
        x = a.set_index("t_end").loc[common]
        y = b.set_index("t_end").loc[common]
        ca, cb = classify(x), classify(y)
        agree = int(np.sum(ca == cb))
        out[tag] = (agree, len(common))
    return out


def main():
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    rows = []
    for name, cfg in DATASETS.items():
        print(f"=== {name} ===")
        res = analyze(name, cfg)
        # widths for panels D/F
        for dat in (res["weekly"], res["daily"]):
            dat["agg"]["width"] = (dat["agg"]["Quantile.0.975(R)"]
                                   - dat["agg"]["Quantile.0.025(R)"])
            dat["rep"]["width"] = (dat["rep"]["Quantile.0.975(R)"]
                                   - dat["rep"]["Quantile.0.025(R)"])
        r2, fig = draw_figure(res)
        cls = classification_agreement(res)
        print(f"  figure -> {fig}")
        print(f"  R^2: weekly mean {r2['c']:.2f}, weekly CrI width "
              f"{r2['d']:.3f}, daily mean {r2['e']:.2f}, "
              f"daily CrI width {r2['f']:.2f}")
        for tag, (ag, n) in cls.items():
            print(f"  classification ({tag}): {ag}/{n} = {ag / n * 100:.1f}%")
        # EM convergence (paper: "reconstructions converged within 10 iter.")
        agg = estimate_R_agg(aggregate_inc(res["incid"], dt=7), dt=7, dt_out=7,
                             recon_opt="match", mean_si=cfg["mean_si"],
                             std_si=cfg["std_si"])
        conv = np.max(np.abs(agg["iterations"][:, -1] - agg["iterations"][:, -2]))
        print(f"  EM: max|I_iter10 - I_iter9| = {conv:.3g}")
        rows.append(dict(dataset=name,
                         R2_weekly=f"{r2['c']:.2f}",
                         R2_width_weekly=f"{r2['d']:.{cfg['r2_dec']['d']}f}",
                         R2_daily=f"{r2['e']:.2f}",
                         R2_width_daily=f"{r2['f']:.2f}",
                         class_weekly=f"{cls['weekly'][0]}/{cls['weekly'][1]}"
                                      f" ({cls['weekly'][0] / cls['weekly'][1] * 100:.1f}%)",
                         class_daily=f"{cls['daily'][0]}/{cls['daily'][1]}"
                                     f" ({cls['daily'][0] / cls['daily'][1] * 100:.1f}%)"))

    print("\n== Summary vs paper ==")
    paper = {
        "flu": ("0.91", "0.93", "0.13", "0.14", "81.8%", "44.4%"),
        "covid_cases": ("0.98", "0.99", "0.67", "0.97", "94.9%", "74.4%"),
        "covid_deaths": ("0.98", "1.00", "0.80", "0.97", "93.3%", "85.8%"),
    }
    for r in rows:
        p = paper[r["dataset"]]
        print(f"{r['dataset']:14s} R2: {r['R2_weekly']}/{r['R2_daily']} "
              f"(paper {p[0]}/{p[2]}), width: {r['R2_width_weekly']}/"
              f"{r['R2_width_daily']} (paper {p[1]}/{p[3]}), "
              f"class: {r['class_weekly']}/{r['class_daily']} "
              f"(paper {p[4]}/{p[5]})")


if __name__ == "__main__":
    main()
