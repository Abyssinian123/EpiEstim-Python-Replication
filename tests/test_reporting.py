import numpy as np
import pytest

from epiestim.reporting import aggregate_incidence, apply_reporting, make_reporting_rates


def test_reporting_extremes_and_shape():
    x = np.array([[0, 2, 5], [7, 4, 1]])
    assert np.array_equal(apply_reporting(x, 1, seed=1), x)
    assert np.array_equal(apply_reporting(x, 0, seed=1), np.zeros_like(x))
    y = apply_reporting(x, [0.2, 0.5, 0.9], seed=9)
    assert y.shape == x.shape
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all((y >= 0) & (y <= x))


def test_reproducibility_and_validation():
    x = np.full(1000, 20.0)
    assert np.array_equal(apply_reporting(x, .4, seed=42), apply_reporting(x, .4, seed=42))
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_reporting(x, 1.1)
    with pytest.raises(ValueError, match="integer counts"):
        apply_reporting([1, 2.5], .5)
    with pytest.raises(ValueError, match="integer counts"):
        apply_reporting([-1, 2], .5)


def test_weekly_aggregation_conserves_counts():
    x = np.arange(70).reshape(1, 70)
    weekly = aggregate_incidence(x)
    assert weekly.shape == (1, 10)
    assert weekly.sum() == x.sum()
    with pytest.raises(ValueError, match="multiple"):
        aggregate_incidence(np.arange(8))


def test_scenario_rates_and_boundaries():
    scenarios = [make_reporting_rates(f"S{i}") for i in range(4)]
    assert all(x.shape == (70,) for x in scenarios)
    assert np.all(scenarios[0] == 1)
    assert np.all(scenarios[1] == .8)
    assert np.all(scenarios[2][:34] == .8) and np.all(scenarios[2][34:] == .4)
    assert scenarios[3][33] == .8
    assert scenarios[3][34] == .8 and scenarios[3][55] == .4
    assert np.all(scenarios[3][56:] == .4)


def test_binomial_mean_sanity():
    truth = np.full((5000, 70), 20)
    p = np.linspace(.2, .9, 70)
    observed = apply_reporting(truth, p, seed=100)
    assert np.allclose(observed.mean(axis=0), 20 * p, atol=.25)
