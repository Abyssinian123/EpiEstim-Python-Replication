"""Formal 100-trajectory reporting experiment using A's integrated engine."""

from __future__ import annotations

import argparse
import hashlib
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

from epiestim import estimate_R, estimate_R_agg, simulate_incidence
from epiestim.reporting import aggregate_incidence, apply_reporting, make_reporting_rates
from run_reporting_nash2023_pilot import (
    DT, EPIDEMIC_SEED, N_DAYS, REPORTING_SEEDS, R_COLUMNS, SCENARIOS,
    TRUE_RT, _estimate_one, _simulation_si,
)

ENGINE_SHA256 = "4DB99C1179E2B6BCAC4C287D8E89E5F928CD85206C07F3C3C99175826EF242C8"


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _sha256_array(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _validate_windows(frame):
    if not np.all(frame.t_start >= 1) or not np.all(frame.t_end <= N_DAYS):
        raise AssertionError("output window lies outside the 1-based 1..70 timeline")
    if not np.all(frame.t_end >= frame.t_start) or not np.all(frame.t_end > 3.6):
        raise AssertionError("invalid window or window violates A's t_end > mean_si rule")


def _run_boundary_diagnostics(out):
    cases = {
        "all_zero": np.zeros(N_DAYS, dtype=np.int64),
        "extinct_after_seed": np.r_[np.repeat(1, 7), np.zeros(N_DAYS - 7, dtype=np.int64)],
        "low_counts": np.r_[np.repeat(2, 7), np.ones(N_DAYS - 7, dtype=np.int64)],
    }
    diagnostics = []
    for name, incidence in cases.items():
        record = {"case": name, "true_case_total": int(incidence.sum()),
                  "daily_status": "not_run", "em_status": "not_run",
                  "error_type": "", "error": ""}
        try:
            daily = estimate_R(incidence, mean_si=3.6, std_si=1.6)["R"]
            record["daily_status"] = "ok" if np.isfinite(daily["Mean(R)"]).all() else "anomaly"
            record["daily_rows"] = len(daily)
        except Exception as exc:
            record["daily_status"] = "failed"
            record["error_type"], record["error"] = type(exc).__name__, str(exc)
        try:
            weekly = aggregate_incidence(incidence, dt=DT)
            result = estimate_R_agg(weekly, dt=DT, dt_out=DT, iter=10,
                                    tol=1e-6, recon_opt="match", mean_si=3.6,
                                    std_si=1.6, si_variant="distcrete")
            record["em_status"] = "ok" if np.isfinite(result["R"]["Mean(R)"]).all() else "anomaly"
            record["em_rows"] = len(result["R"])
            record["em_reconstructed_cases"] = float(np.sum(result["I"]))
            record["em_weekly_max_error"] = float(np.max(np.abs(
                aggregate_incidence(result["I"], DT) - weekly)))
        except Exception as exc:
            record["em_status"] = "failed"
            record["error_type"], record["error"] = type(exc).__name__, str(exc)
        diagnostics.append(record)
    pd.DataFrame(diagnostics).to_csv(out / "boundary_cases.csv", index=False)
    return diagnostics


def run_formal(output_dir=None, n_sim=100):
    if n_sim != 100:
        raise ValueError("formal run is fixed at exactly 100 trajectories")
    if N_DAYS % DT:
        raise AssertionError("70-day series must divide evenly into 7-day weeks")
    out = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parents[2] / "results" / "reporting_nash2023" / "formal")
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result directory: {out}")
    out.mkdir(parents=True)
    boundary_diagnostics = _run_boundary_diagnostics(out)

    all_started = time.perf_counter()
    sim_started = time.perf_counter()
    truth = simulate_incidence(TRUE_RT, _simulation_si(), n_days=N_DAYS,
                               n_sim=n_sim, seed=EPIDEMIC_SEED)
    simulation_batch_seconds = time.perf_counter() - sim_started
    if truth.shape != (n_sim, N_DAYS):
        raise AssertionError(f"unexpected true incidence shape {truth.shape}")
    if np.any(truth < 0) or np.any(truth != np.floor(truth)):
        raise AssertionError("simulator output must be nonnegative integer case counts")

    # Re-run the stochastic inputs from their fixed seeds and verify exact reproducibility.
    truth_again = simulate_incidence(TRUE_RT, _simulation_si(), n_days=N_DAYS,
                                     n_sim=n_sim, seed=EPIDEMIC_SEED)
    if not np.array_equal(truth, truth_again):
        raise AssertionError("epidemic simulation failed fixed-seed reproducibility check")

    estimate_frames, statuses = [], []
    scenario_cache = {}
    per_scenario_diagnostics = {}
    for scenario in SCENARIOS:
        scenario_started = time.perf_counter()
        rates = make_reporting_rates(scenario, N_DAYS)
        observed = apply_reporting(truth, rates, seed=REPORTING_SEEDS[scenario])
        observed_again = apply_reporting(truth_again, rates, seed=REPORTING_SEEDS[scenario])
        if not np.array_equal(observed, observed_again):
            raise AssertionError(f"{scenario}: reporting draws are not reproducible")
        if np.any(observed > truth):
            raise AssertionError(f"{scenario}: reported cases exceed true cases")
        if scenario == "S0" and not np.array_equal(observed, truth):
            raise AssertionError("S0 observations must exactly equal true incidence")
        weekly = aggregate_incidence(observed, dt=DT)
        if weekly.shape != (n_sim, N_DAYS // DT):
            raise AssertionError(f"{scenario}: unexpected weekly shape {weekly.shape}")
        if not np.array_equal(weekly.sum(axis=1), observed.sum(axis=1)):
            raise AssertionError(f"{scenario}: aggregation did not conserve counts")

        reconstructed = np.full((n_sim, N_DAYS), np.nan, dtype=float)
        for sim_idx in range(n_sim):
            row_started = time.perf_counter()
            status = {
                "simulation_id": sim_idx + 1,
                "rt_scenario": "constant_1.25",
                "reporting_scenario": scenario,
                "true_case_total": int(truth[sim_idx].sum()),
                "reported_case_total": int(observed[sim_idx].sum()),
                "weekly_case_total": int(weekly[sim_idx].sum()),
                "simulation_batch_seconds": simulation_batch_seconds,
                "trajectory_pipeline_seconds": np.nan,
                "estimate_seconds": np.nan,
                "status": "failed",
                "anomaly_count": 0,
                "error_type": "",
                "error": "",
            }
            try:
                em, daily, em_r, outputs = _estimate_one(
                    observed[sim_idx], weekly[sim_idx], sim_idx + 1, scenario)
                _validate_windows(daily)
                _validate_windows(em_r)
                if not np.array_equal(daily[["t_start", "t_end"]].to_numpy(),
                                      em_r[["t_start", "t_end"]].to_numpy()):
                    raise AssertionError("daily and EM windows are not strictly aligned")
                reconstructed[sim_idx] = em["I"]
                reconstructed_weekly = aggregate_incidence(em["I"], dt=DT)
                weekly_error = float(np.max(np.abs(reconstructed_weekly - weekly[sim_idx])))
                numeric = pd.concat([daily[R_COLUMNS], em_r[R_COLUMNS]], ignore_index=True)
                anomaly_count = int((~np.isfinite(numeric.select_dtypes(include=[np.number]))).sum().sum())
                for frame in outputs:
                    frame["paired_truth_id"] = sim_idx + 1
                    estimate_frames.append(frame)
                status.update({
                    "status": "ok" if anomaly_count == 0 else "anomaly",
                    "anomaly_count": anomaly_count,
                    "estimate_seconds": time.perf_counter() - row_started,
                    "em_max_weekly_total_error": weekly_error,
                    "daily_window_count": len(daily),
                    "em_window_count": len(em_r),
                })
                if anomaly_count:
                    status["error"] = "Non-finite values found in numeric R output columns"
            except Exception as exc:
                status.update({"error_type": type(exc).__name__, "error": str(exc)})
            status["trajectory_pipeline_seconds"] = time.perf_counter() - row_started
            statuses.append(status)

        scenario_cache[scenario] = {
            "true_incidence": truth,
            "observed_incidence": observed,
            "weekly_incidence": weekly,
            "reporting_rates": rates,
            "true_rt": np.full(N_DAYS, TRUE_RT, dtype=float),
            "em_reconstructed_incidence": reconstructed,
        }
        np.savez_compressed(out / f"{scenario}.npz", **scenario_cache[scenario])
        per_scenario_diagnostics[scenario] = {
            "scenario_elapsed_seconds": time.perf_counter() - scenario_started,
            "true_incidence_sha256": _sha256_array(truth),
            "observed_incidence_sha256": _sha256_array(observed),
            "all_simulations_share_same_truth": bool(np.array_equal(truth, truth_again)),
        }
        pd.DataFrame(statuses).to_csv(out / "run_status.csv", index=False)

    estimates = pd.concat(estimate_frames, ignore_index=True) if estimate_frames else pd.DataFrame()
    estimates.to_csv(out / "estimates_all.csv", index=False)
    estimates.loc[estimates.estimation_method == "daily_reported"].to_csv(
        out / "estimates_daily.csv", index=False)
    estimates.loc[estimates.estimation_method == "weekly_em"].to_csv(
        out / "estimates_weekly_em.csv", index=False)

    # Pair-preserving keys are explicit, and aligned windows are checked across methods.
    key = ["simulation_id", "rt_scenario", "reporting_scenario", "t_start", "t_end"]
    daily_keys = estimates.loc[estimates.estimation_method == "daily_reported", key]
    em_keys = estimates.loc[estimates.estimation_method == "weekly_em", key]
    if not daily_keys.sort_values(key).reset_index(drop=True).equals(
            em_keys.sort_values(key).reset_index(drop=True)):
        raise AssertionError("full-run daily and EM estimate keys do not match")

    # Figure summarizes the full intervals while retaining scenario pairing in raw CSV.
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"S0": "black", "S1": "steelblue", "S2": "darkorange", "S3": "seagreen"}
    for scenario in SCENARIOS:
        for method, label, linestyle in (("daily_reported", "Daily reported", "-"),
                                         ("weekly_em", "Weekly EM", "--")):
            subset = estimates[(estimates.reporting_scenario == scenario) &
                               (estimates.estimation_method == method)]
            grouped = subset.groupby("t_end").agg(
                mean=("Mean(R)", "mean"), low=("Quantile.0.025(R)", "mean"),
                high=("Quantile.0.975(R)", "mean"))
            if not grouped.empty:
                x = grouped.index.to_numpy()
                ax.plot(x, grouped["mean"], color=colors[scenario], linestyle=linestyle,
                        label=f"{scenario} {label}")
                ax.fill_between(x, grouped.low, grouped.high, color=colors[scenario], alpha=.07)
    ax.axhline(TRUE_RT, color="red", linestyle=":", label="True Rt = 1.25")
    ax.set(xlabel="Window end day (1-based)", ylabel="Estimated Rt",
           title="Formal 100-trajectory reporting experiment")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rt_estimates.png", dpi=170)
    plt.close(fig)

    status_frame = pd.DataFrame(statuses)
    metadata = {
        "status": "formal 100-trajectory experiment",
        "rt_scenario": "constant_1.25",
        "true_rt": TRUE_RT,
        "true_rt_definition": "Constant transmission Rt supplied to A's simulate_incidence for all 70 returned days; the 7 initial historical seed days are excluded from those 70 days.",
        "n_sim": n_sim, "n_days": N_DAYS, "dt": DT, "dt_out": DT,
        "epidemic_seed": EPIDEMIC_SEED, "reporting_seeds": REPORTING_SEEDS,
        "reporting_seed_policy": "Independent fixed Generator seed per reporting scenario; all scenarios use the same true-incidence matrix.",
        "mean_si": 3.6, "std_si": 1.6,
        "simulation_si": "A discr_si(0:30), normalized and w0 removed.",
        "estimation": {"method": "parametric_si", "iter": 10, "tol": 1e-6,
                        "recon_opt": "match", "si_variant": "distcrete",
                        "window_alignment": "Daily estimate_R is evaluated on the exact t_start/t_end pairs returned by estimate_R_agg."},
        "formal_engine_commit": "079c67a",
        "formal_engine_sha256": ENGINE_SHA256,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                      "pandas": pd.__version__, "scipy": _version("scipy"),
                      "matplotlib": matplotlib.__version__},
        "simulation_batch_seconds": simulation_batch_seconds,
        "total_elapsed_seconds": time.perf_counter() - all_started,
        "scenario_diagnostics": per_scenario_diagnostics,
        "reproducibility_check": {
            "passed": True,
            "method": "Re-generated the 100 truth trajectories and each scenario's observed matrix from the recorded seeds and compared arrays exactly.",
            "estimation": "Estimation engine is deterministic for these fixed input arrays and options.",
        },
        "run_counts": status_frame.status.value_counts().to_dict(),
        "boundary_case_diagnostics": boundary_diagnostics,
        "outputs": [f"{s}.npz" for s in SCENARIOS] + [
            "estimates_all.csv", "estimates_daily.csv", "estimates_weekly_em.csv",
            "run_status.csv", "boundary_cases.csv", "rt_estimates.png",
            "formal_check_report.md"],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_check_report(out, status_frame, estimates, metadata)
    return out, status_frame, estimates, metadata


def _write_check_report(out, statuses, estimates, metadata):
    errors = statuses.loc[statuses.status != "ok"]
    daily = estimates.loc[estimates.estimation_method == "daily_reported"]
    em = estimates.loc[estimates.estimation_method == "weekly_em"]
    lines = [
        "# 正式报告率模拟检查报告", "",
        f"- 轨迹数：{metadata['n_sim']}；模拟时长：{metadata['n_days']} 天；Rt={TRUE_RT}。",
        f"- 状态：ok={int((statuses.status == 'ok').sum())}，异常/失败={len(errors)}。",
        f"- 日度估计行数：{len(daily)}；周度 EM 估计行数：{len(em)}。",
        "- S0–S3 共用相同真病例；各场景固定随机种子可复现。",
        "- S0 观测病例等于真病例；周汇总总数与日观测总数相等。",
        "- 日度与周度 EM 估计按 simulation_id、场景及 t_start/t_end 配对。",
        "- EM 重建病例已逐轨迹重聚合核对输入周总数；详细误差在 run_status.csv。",
        f"- 总耗时：{metadata['total_elapsed_seconds']:.2f} 秒。",
        "- 数值设定：`estimate_R_agg` 使用 `si_variant=distcrete`、`recon_opt=match`、`dt_out=7`。",
    ]
    if errors.empty:
        lines.append("- 轨迹错误与非有限估计值：无。")
    else:
        lines.extend(["", "## 异常或失败轨迹", ""])
        lines.append(errors[["simulation_id", "reporting_scenario", "status", "error_type", "error"]].to_markdown(index=False))
    (out / "formal_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-sim", type=int, default=100)
    args = parser.parse_args()
    path, status, estimates, meta = run_formal(args.output_dir, args.n_sim)
    print(f"Formal outputs: {path}")
    print(f"Elapsed: {meta['total_elapsed_seconds']:.2f}s; status counts: {meta['run_counts']}")
