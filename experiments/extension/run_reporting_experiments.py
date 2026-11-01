"""Generate a clearly labeled synthetic fixture for reporting-model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from epiestim.reporting import aggregate_incidence, apply_reporting, make_reporting_rates

SCENARIOS = ("S0", "S1", "S2", "S3")
EPIDEMIC_SEED = 6
REPORTING_SEED = 2026


def synthetic_incidence(n_sim=100, n_days=70, seed=EPIDEMIC_SEED):
    """Independent illustrative fixture, not the original paper's simulator."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_sim, n_days), dtype=np.int64)
    out[:, :min(7, n_days)] = 10
    for day in range(7, n_days):
        # Deliberately simple stochastic fixture; no Rt is asserted or exported.
        infectiousness = out[:, max(0, day - 7):day].mean(axis=1)
        out[:, day] = rng.poisson(np.maximum(0.2, 1.15 * infectiousness))
    return out


def run(output_dir=None, n_sim=100, n_days=70):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "results" / "reporting"
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = synthetic_incidence(n_sim, n_days)
    rng = np.random.default_rng(REPORTING_SEED)
    rate_map, observed_map, weekly_map = {}, {}, {}
    for scenario in SCENARIOS:
        rates = make_reporting_rates(scenario, n_days)
        observed = apply_reporting(truth, rates, seed=rng)
        weekly = aggregate_incidence(observed, 7)
        rate_map[scenario], observed_map[scenario], weekly_map[scenario] = rates, observed, weekly
        np.savez_compressed(output_dir / f"{scenario}.npz", true_incidence=truth,
                            observed_incidence=observed, weekly_incidence=weekly,
                            reporting_rates=rates)
    metadata = {
        "description": "Independent synthetic incidence fixture; not a reproduction of the original paper simulator.",
        "n_sim": n_sim, "n_days": n_days, "dt": 7,
        "epidemic_seed": EPIDEMIC_SEED, "reporting_seed": REPORTING_SEED,
        "reporting_seed_stream": "Separate Generator stream; scenarios consume it in S0-S3 order.",
        "true_rt": None,
        "scenario_ids": list(SCENARIOS),
        "assumptions": "True counts use an illustrative stochastic fixture; reporting is Binomial(I_t, p_t).",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to generate the required figures") from exc
    days = np.arange(1, n_days + 1)
    fig, ax = plt.subplots()
    for key, rates in rate_map.items(): ax.plot(days, rates, label=key)
    ax.set(xlabel="Day (1-based)", ylabel="Reporting probability", title="Reporting-rate scenarios", ylim=(0, 1.05)); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "reporting_rates.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(days, truth[0], label="True incidence", linewidth=2)
    for key, obs in observed_map.items(): ax.plot(days, obs[0], label=f"Reported {key}", alpha=.8)
    ax.set(xlabel="Day (1-based)", ylabel="Cases per day", title="One shared true trajectory and reported observations"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "daily_incidence.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots()
    weeks = np.arange(1, weekly_map["S0"].shape[1] + 1)
    for key, weekly in weekly_map.items(): ax.plot(weeks, weekly[0], marker="o", label=key)
    ax.set(xlabel="Aligned week (day 1 starts week 1)", ylabel="Reported cases per 7-day week", title="Weekly aggregated reported incidence"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "weekly_incidence.png", dpi=160); plt.close(fig)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--n-sim", type=int, default=100)
    parser.add_argument("--n-days", type=int, default=70)
    args = parser.parse_args()
    print(run(args.output_dir, args.n_sim, args.n_days))
