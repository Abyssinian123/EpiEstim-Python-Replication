"""Run the Member B reporting extension using the temporary A engine adapter."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epiestim import discr_si, estimate_R, estimate_R_agg, simulate_incidence
from epiestim.reporting import aggregate_incidence, apply_reporting, make_reporting_rates

N_DAYS = 70
DT = 7
TRUE_RT = 1.25  # Current pilot setting; pending full-team confirmation.
EPIDEMIC_SEED = 6
REPORTING_SEEDS = {"S0": 4100, "S1": 4101, "S2": 4102, "S3": 4103}
SCENARIOS = ("S0", "S1", "S2", "S3")
ENGINE_SOURCE_SHA256 = "4DB99C1179E2B6BCAC4C287D8E89E5F928CD85206C07F3C3C99175826EF242C8"
R_COLUMNS = [
    "t_start", "t_end", "Mean(R)", "Std(R)",
    "Quantile.0.025(R)", "Quantile.0.05(R)", "Quantile.0.25(R)",
    "Median(R)", "Quantile.0.75(R)", "Quantile.0.95(R)",
    "Quantile.0.975(R)",
]


def _pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed-as-package"


def _simulation_si():
    # Match A's sim_study.sim_si(): normalise discr_si(0:30), then remove lag 0.
    pmf = discr_si(np.arange(31), 3.6, 1.6)
    return (pmf / pmf.sum())[1:]


def _check_r_table(table, label):
    missing = [col for col in R_COLUMNS if col not in table.columns]
    if missing:
        raise AssertionError(f"{label}: missing expected R columns: {missing}")
    if not np.all(table["t_start"].to_numpy() >= 1):
        raise AssertionError(f"{label}: t_start is not 1-based")
    if not np.all(table["t_end"].to_numpy() <= N_DAYS):
        raise AssertionError(f"{label}: t_end exceeds simulated period")
    if not np.all(table["t_end"].to_numpy() >= table["t_start"].to_numpy()):
        raise AssertionError(f"{label}: invalid window ordering")
    if not np.all(table["t_end"].to_numpy() > 3.6):
        raise AssertionError(f"{label}: an ineligible t_end <= mean_si was emitted")


def _estimate_one(observed, weekly, simulation_id, reporting_scenario):
    em = estimate_R_agg(weekly, dt=DT, dt_out=DT, iter=10, tol=1e-6,
                        recon_opt="match", mean_si=3.6, std_si=1.6,
                        si_variant="distcrete")
    em_r = em["R"].copy()
    _check_r_table(em_r, "estimate_R_agg")

    # Force daily posterior estimates onto exactly the EM output windows.
    daily = estimate_R(observed, mean_si=3.6, std_si=1.6,
                       t_start=em_r["t_start"].to_numpy(),
                       t_end=em_r["t_end"].to_numpy())["R"]
    _check_r_table(daily, "aligned estimate_R")
    if not np.array_equal(daily[["t_start", "t_end"]].to_numpy(),
                          em_r[["t_start", "t_end"]].to_numpy()):
        raise AssertionError("daily and EM estimates do not have identical windows")

    reconstructed_weekly = aggregate_incidence(em["I"], dt=DT)
    if not np.allclose(reconstructed_weekly, weekly, rtol=0, atol=1e-6):
        raise AssertionError("EM reconstructed incidence does not preserve weekly totals")
    outputs = []
    for method, result in (("daily_reported", daily), ("weekly_em", em_r)):
        frame = result.copy()
        frame["simulation_id"] = simulation_id
        frame["rt_scenario"] = "constant_1.25"
        frame["reporting_scenario"] = reporting_scenario
        frame["estimation_method"] = method
        frame["true_rt"] = TRUE_RT
        outputs.append(frame)
    return em, daily, em_r, outputs


def validate_small_scale(n_sim=3):
    if n_sim != 3:
        raise ValueError("the required first validation is exactly 3 trajectories")
    if N_DAYS % DT:
        raise AssertionError("simulation length must be divisible by weekly dt")
    started = time.perf_counter()
    truth = simulate_incidence(TRUE_RT, _simulation_si(), n_days=N_DAYS,
                               n_sim=n_sim, seed=EPIDEMIC_SEED)
    if truth.shape != (n_sim, N_DAYS):
        raise AssertionError(f"unexpected simulator shape: {truth.shape}")
    if np.any(truth < 0) or np.any(truth != np.floor(truth)):
        raise AssertionError("A simulator did not return nonnegative integer counts")

    records, scenario_truth = [], {}
    for scenario in SCENARIOS:
        rates = make_reporting_rates(scenario, N_DAYS)
        observed = apply_reporting(truth, rates, seed=REPORTING_SEEDS[scenario])
        if not np.all(observed <= truth):
            raise AssertionError(f"{scenario}: reported count exceeds true count")
        if scenario == "S0" and not np.array_equal(observed, truth):
            raise AssertionError("S0 must equal true incidence exactly")
        weekly = aggregate_incidence(observed, dt=DT)
        if weekly.shape != (n_sim, 10) or not np.array_equal(weekly.sum(1), observed.sum(1)):
            raise AssertionError(f"{scenario}: weekly aggregation failed conservation")
        scenario_truth[scenario] = truth.copy()
        for sim_idx in range(n_sim):
            try:
                em, daily, em_r, rows = _estimate_one(
                    observed[sim_idx], weekly[sim_idx], sim_idx + 1, scenario)
                records.append({
                    "simulation_id": sim_idx + 1,
                    "reporting_scenario": scenario,
                    "true_case_total": int(truth[sim_idx].sum()),
                    "reported_case_total": int(observed[sim_idx].sum()),
                    "weekly_case_total": int(weekly[sim_idx].sum()),
                    "em_reconstruction_max_weekly_error": float(np.max(np.abs(
                        aggregate_incidence(em["I"], DT) - weekly[sim_idx]))),
                    "status": "ok",
                })
            except Exception as exc:
                # Preserve the failed trajectory in the audit rather than silently dropping it.
                records.append({"simulation_id": sim_idx + 1,
                                "reporting_scenario": scenario,
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "error": str(exc)})
    if not all(np.array_equal(scenario_truth["S0"][i], scenario_truth[s][i])
               for s in SCENARIOS for i in range(n_sim)):
        raise AssertionError("scenario truth trajectories differ")
    audit = pd.DataFrame(records)
    failures = audit.loc[audit.get("status", pd.Series(dtype=str)) == "failed"]
    return {"passed": failures.empty, "audit": audit,
            "elapsed_seconds": time.perf_counter() - started,
            "n_sim": n_sim, "truth": truth}


def run_pilot(output_dir=None, n_sim=10):
    if n_sim < 1 or n_sim > 10:
        raise ValueError("pilot n_sim must be between 1 and 10")
    output_dir = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parents[2] / "results" / "reporting_nash2023" / "pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    truth = simulate_incidence(TRUE_RT, _simulation_si(), n_days=N_DAYS,
                               n_sim=n_sim, seed=EPIDEMIC_SEED)
    data_frames = []
    run_status = []
    for scenario in SCENARIOS:
        rates = make_reporting_rates(scenario, N_DAYS)
        observed = apply_reporting(truth, rates, seed=REPORTING_SEEDS[scenario])
        weekly = aggregate_incidence(observed, DT)
        np.savez_compressed(output_dir / f"{scenario}.npz",
                            true_incidence=truth,
                            observed_incidence=observed,
                            weekly_incidence=weekly,
                            reporting_rates=rates,
                            true_rt=np.full(N_DAYS, TRUE_RT))
        for sim_idx in range(n_sim):
            try:
                _, _, _, rows = _estimate_one(observed[sim_idx], weekly[sim_idx],
                                               sim_idx + 1, scenario)
                data_frames.extend(rows)
                run_status.append({"simulation_id": sim_idx + 1,
                                   "reporting_scenario": scenario,
                                   "status": "ok"})
            except Exception as exc:
                run_status.append({"simulation_id": sim_idx + 1,
                                   "reporting_scenario": scenario,
                                   "status": "failed",
                                   "error_type": type(exc).__name__,
                                   "error": str(exc)})
    estimates = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    # Unified Member C table includes all 11 raw R columns plus metadata.
    estimates.to_csv(output_dir / "estimates_all.csv", index=False)
    for method, filename in (("daily_reported", "estimates_daily.csv"),
                             ("weekly_em", "estimates_weekly_em.csv")):
        estimates.loc[estimates["estimation_method"] == method].to_csv(
            output_dir / filename, index=False)
    pd.DataFrame(run_status).to_csv(output_dir / "run_status.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = dict(zip(SCENARIOS, ("black", "steelblue", "darkorange", "seagreen")))
    for scenario in SCENARIOS:
        for method, label, ls in (("daily_reported", "Daily reported", "-"),
                                  ("weekly_em", "Weekly EM", "--")):
            subset = estimates[(estimates.reporting_scenario == scenario) &
                               (estimates.estimation_method == method)]
            summary = subset.groupby("t_end").agg(mean=("Mean(R)", "mean"),
                                                    lo=("Quantile.0.025(R)", "mean"),
                                                    hi=("Quantile.0.975(R)", "mean"))
            if summary.empty:
                continue
            x = summary.index.to_numpy()
            ax.plot(x, summary["mean"], color=colors[scenario], linestyle=ls,
                    label=f"{scenario} {label}")
            ax.fill_between(x, summary["lo"], summary["hi"],
                            color=colors[scenario], alpha=.08)
    ax.axhline(TRUE_RT, color="red", linestyle=":", label="True Rt = 1.25")
    ax.set(xlabel="Window end day (1-based)", ylabel="Estimated Rt",
           title="Pilot: reporting scenarios and Rt estimates")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "rt_estimates.png", dpi=160)
    plt.close(fig)

    metadata = {
        "status": "pilot; not a final study",
        "rt_scenario": "constant_1.25",
        "true_rt": TRUE_RT,
        "true_rt_setting_note": "Current pre-experiment setting; pending full-team confirmation.",
        "n_sim": n_sim, "n_days": N_DAYS, "dt": DT,
        "epidemic_seed": EPIDEMIC_SEED,
        "reporting_seeds": REPORTING_SEEDS,
        "reporting_seed_policy": "Independent fixed seed per scenario; same truth matrix across scenarios.",
        "mean_si": 3.6, "std_si": 1.6,
        "estimation": {"method": "parametric_si", "iter": 10, "tol": 1e-6,
                        "recon_opt": "match", "dt_out": 7,
                        "si_variant": "distcrete"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                      "pandas": pd.__version__, "scipy": _pkg_version("scipy"),
                      "matplotlib": matplotlib.__version__},
        "engine_source": "Team formal src/epiestim/epiestim.py from A commit 079c67a; imported via src/epiestim/__init__.py",
        "engine_source_sha256": ENGINE_SOURCE_SHA256,
        "true_rt_field": "Constant value supplied to simulate_incidence; no per-day Rt return is provided by A's function.",
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": ["S0.npz", "S1.npz", "S2.npz", "S3.npz",
                    "estimates_all.csv", "estimates_daily.csv",
                    "estimates_weekly_em.csv", "run_status.csv", "rt_estimates.png"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir, pd.DataFrame(run_status), metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--n-sim", type=int, default=10)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[2] / "results" / "reporting_nash2023" / "pilot")
    root.mkdir(parents=True, exist_ok=True)
    check = validate_small_scale()
    check["audit"].to_csv(root / "three_trajectory_validation.csv", index=False)
    print(f"3-trajectory validation passed={check['passed']} elapsed={check['elapsed_seconds']:.2f}s")
    if not check["passed"]:
        raise SystemExit("3-trajectory validation failed; pilot expansion stopped")
    if not args.validate_only:
        output, status, metadata = run_pilot(root, args.n_sim)
        print(f"pilot output: {output}")
        print(f"pilot elapsed: {metadata['elapsed_seconds']:.2f}s; failed rows: {(status.status == 'failed').sum()}")
