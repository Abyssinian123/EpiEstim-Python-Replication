"""Small robustness check: constant underreporting at several levels.

Extends Member C's S1 result (constant p = 0.8) by sweeping the *constant*
reporting rate p over {0.8, 0.6, 0.4} (i.e. 20% / 40% / 60% underreporting) and
measuring whether the daily / weekly-EM Rt estimates stay unbiased and
well-calibrated. Uses the same truth matrix, serial interval and estimation
settings as the formal 100-trajectory run (run_reporting_nash2023_formal.py),
so the numbers are directly comparable to S0-S3.

The point of the sweep: the renewal-process theory (docs/reporting_robustness_theory.md)
predicts that *constant* thinning does not bias R_t at any level - only
*time-varying* p_t does. This script is the numerical confirmation of that claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent          # experiments/analysis/
REPO = BASE.parent.parent                       # repository root
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epiestim import discr_si, estimate_R, estimate_R_agg, simulate_incidence
from epiestim.reporting import aggregate_incidence, apply_reporting

from stat_metrics import TRUE_RT, add_error_columns, summarize_frame

N_DAYS = 70
DT = 7
N_SIM = 100
EPIDEMIC_SEED = 6          # same truth matrix as the formal run
REPORTING_SEED = 4300      # one fixed seed for all levels -> only p varies
P_LEVELS = (0.8, 0.6, 0.4)
MEAN_SI, STD_SI = 3.6, 1.6

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DAILY = "#2a78d6"
EM = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "axes.grid": True,
    "axes.axisbelow": True,
})


def _simulation_si():
    pmf = discr_si(np.arange(31), MEAN_SI, STD_SI)
    return (pmf / pmf.sum())[1:]


def _estimate_one(observed, weekly, simulation_id, label):
    """EM reconstruction + aligned daily estimate for one trajectory."""
    em = estimate_R_agg(weekly, dt=DT, dt_out=DT, iter=10, tol=1e-6,
                        recon_opt="match", mean_si=MEAN_SI, std_si=STD_SI,
                        si_variant="distcrete")
    em_r = em["R"].copy()
    daily = estimate_R(observed, mean_si=MEAN_SI, std_si=STD_SI,
                       t_start=em_r["t_start"].to_numpy(),
                       t_end=em_r["t_end"].to_numpy())["R"]
    rows = []
    for method, result in (("daily_reported", daily), ("weekly_em", em_r)):
        frame = result.copy()
        frame["simulation_id"] = simulation_id
        frame["reporting_scenario"] = label
        frame["estimation_method"] = method
        frame["true_rt"] = TRUE_RT
        rows.append(frame)
    return rows


def run(output_dir=None):
    out = Path(output_dir) if output_dir else (
        REPO / "results" / "reporting_nash2023" / "constant_p")
    out.mkdir(parents=True, exist_ok=True)

    truth = simulate_incidence(TRUE_RT, _simulation_si(), n_days=N_DAYS,
                               n_sim=N_SIM, seed=EPIDEMIC_SEED)

    frames = []
    for p in P_LEVELS:
        label = f"p{p:g}"
        rates = np.full(N_DAYS, p)
        observed = apply_reporting(truth, rates, seed=REPORTING_SEED)
        weekly = aggregate_incidence(observed, dt=DT)
        for sim in range(N_SIM):
            frames.extend(_estimate_one(observed[sim], weekly[sim], sim + 1, label))
        print(f"  p={p:g}: reported total = {int(observed.sum())} "
              f"(true {int(truth.sum())}, ratio {observed.sum()/truth.sum():.3f})")

    estimates = pd.concat(frames, ignore_index=True)
    estimates.to_csv(out / "estimates_all.csv", index=False)

    df = add_error_columns(estimates)
    summary = summarize_frame(df).sort_values(["reporting_scenario", "estimation_method"])
    summary.to_csv(out / "summary.csv", index=False)

    _make_figure(df, out / "constant_p_robustness.png")
    return out, estimates, summary


def _make_figure(df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # left: mean bias over time — the three constant levels should all hover ~0
    for p, color in zip(P_LEVELS, ("#2a78d6", "#eb6834", "#6a9e5a")):
        label = f"p{p:g}"
        sub = df[(df.reporting_scenario == label) &
                 (df.estimation_method == "weekly_em")]
        grp = sub.groupby("t_end")["bias"].mean()
        ax1.plot(grp.index, grp.values, color=color, linewidth=1.6,
                 label=f"p = {p:g}（{round((1 - p) * 100)}% 漏报）")
    ax1.axhline(0, color=MUTED, linewidth=1.0, zorder=0)
    ax1.set_title("恒定漏报程度下的平均偏差（周度 EM）", loc="left")
    ax1.set_xlabel("估计窗口结束日 t_end")
    ax1.set_ylabel("平均偏差  Mean(R) − 1.25")
    ax1.legend(fontsize=8, loc="lower left")

    # right: coverage per level
    cov = (df.groupby(["reporting_scenario", "estimation_method"])["covered"]
             .mean().mul(100).reset_index())
    x = np.arange(len(P_LEVELS))
    width = 0.38
    for i, (method, label, color) in enumerate(
            (("daily_reported", "日度报告", DAILY), ("weekly_em", "周度 EM", EM))):
        vals = [cov[(cov.reporting_scenario == f"p{p:g}") &
                    (cov.estimation_method == method)].covered.iloc[0]
                for p in P_LEVELS]
        bars = ax2.bar(x + (i - 0.5) * width, vals, width, color=color, label=label)
        ax2.bar_label(bars, fmt="%.1f%%", fontsize=7.5, color=INK_2, padding=2)
    ax2.axhline(95, color=MUTED, linewidth=1.0, linestyle=":", zorder=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"p = {p:g}\n（{round((1 - p) * 100)}% 漏报）"
                         for p in P_LEVELS])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("95% 可信区间覆盖率 (%)")
    ax2.set_title("恒定漏报程度下的覆盖率", loc="left")
    ax2.legend(fontsize=8, loc="lower left", ncol=2)

    fig.suptitle("小型稳健性分析：不同恒定漏报程度对 Rt 估计的影响", fontsize=11,
                 x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    out, estimates, summary = run()
    print("\n=== 恒定漏报程度稳健性汇总 ===")
    print(summary.round(4).to_string(index=False))
    print("\noutputs ->", out.resolve())
