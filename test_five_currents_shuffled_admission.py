import numpy as np
import pandas as pd
import pytest

from five_currents_shuffled_admission import (
    apply_admission_gate, block_permutation_indices, cash_before_first_trade,
    pooled_null_sharpes, sharpes_from_permutations, training_slice,
)


def sample():
    rng = np.random.default_rng(9)
    targets = rng.normal(0, .01, (126, 3))
    targets[10:15, 2] = np.nan
    weights = rng.normal(0, .2, (4, 126, 3))
    weights[0, :, 2] = 0
    costs = np.full((4, 126), .0001)
    valid = ~(~np.isfinite(targets)[None, :, :] & (abs(weights) > 1e-10)).any(axis=2)
    return weights, targets, costs, valid


def test_pnl_shuffle_does_not_change_sharpe():
    values = np.random.default_rng(2).normal(.001, .01, 100)
    shuffled = np.random.default_rng(3).permutation(values)
    np.testing.assert_allclose(values.mean() / values.std(), shuffled.mean() / shuffled.std())


def test_permutation_is_joint_and_preserves_missingness():
    _, targets, _, _ = sample()
    targets[:, 1] = targets[:, 0] * 2
    p = block_permutation_indices(targets, np.random.default_rng(2))
    np.testing.assert_array_equal(np.sort(p), np.arange(len(targets)))
    np.testing.assert_array_equal(np.isfinite(targets[p]), np.isfinite(targets))
    np.testing.assert_allclose(targets[p, 1], targets[p, 0] * 2)
    assert not np.array_equal(p, np.arange(len(targets)))


def test_randomized_sharpes_match_scalar_calculation():
    weights, targets, costs, valid = sample()
    permutations = np.stack([np.arange(len(targets)),
                             block_permutation_indices(targets, np.random.default_rng(8))])
    actual = sharpes_from_permutations(weights, targets, costs, valid, permutations)
    for draw, p in enumerate(permutations):
        for concept in range(len(weights)):
            returns = (weights[concept] * np.nan_to_num(targets[p], nan=0)).sum(axis=1) - costs[concept]
            returns = returns[valid[concept]]
            expected = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
            np.testing.assert_allclose(actual[draw, concept], expected, atol=1e-12)


def test_null_is_deterministic_and_batch_independent():
    args = sample()
    first = pooled_null_sharpes(*args, draws=23, seed=7, batch_size=5)
    second = pooled_null_sharpes(*args, draws=23, seed=7, batch_size=11)
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_costs_are_not_shuffled_or_dropped():
    weights, targets, costs, valid = sample()
    p = np.arange(len(targets))[None, :]
    with_cost = sharpes_from_permutations(weights, targets, costs, valid, p)
    without_cost = sharpes_from_permutations(weights, targets, costs * 0, valid, p)
    assert np.all(with_cost < without_cost)


def test_gate_is_additional_and_percentiles_are_nested():
    selection = pd.DataFrame({"refit_ordinal": [1, 1, 1, 1],
                              "selected": [True, True, False, True],
                              "train_base_sharpe": [.2, 1.1, 4, 1.5]})
    thresholds = pd.DataFrame({"refit_ordinal": [1], "p80": [.5], "p90": [1.1], "p95": [1.4]})
    selected = [apply_admission_gate(selection, thresholds, p).selected.to_numpy() for p in (80, 90, 95)]
    assert not selected[0][2]
    assert not selected[1][1]
    assert np.all(selected[1] <= selected[0]) and np.all(selected[2] <= selected[1])


def test_gate_missing_threshold_fails_closed():
    selection = pd.DataFrame({"refit_ordinal": [2], "selected": [True], "train_base_sharpe": [1.]})
    with pytest.raises(ValueError):
        apply_admission_gate(selection, pd.DataFrame({"refit_ordinal": [1], "p80": [.5]}), 80)


def test_training_history_excludes_future_inputs():
    index = pd.bdate_range("2020-01-01", periods=150)
    row = pd.Series({"train_start": index[0], "train_end": index[125],
                     "forecast_start": index[127], "training_sessions": 126})
    positions = training_slice(index, row)
    weights, targets, costs, valid = sample()
    extended = np.vstack([targets, np.zeros((24, 3))])
    before = pooled_null_sharpes(weights, extended[positions], costs, valid, draws=10, seed=8)
    extended[126:] = 10000
    after = pooled_null_sharpes(weights, extended[positions], costs, valid, draws=10, seed=8)
    np.testing.assert_array_equal(before, after)


def test_invalid_training_embargo_is_rejected():
    index = pd.bdate_range("2020-01-01", periods=150)
    row = pd.Series({"train_start": index[0], "train_end": index[126],
                     "forecast_start": index[127], "training_sessions": 127})
    with pytest.raises(AssertionError):
        training_slice(index, row)


def test_cash_warmup_is_real_cash_not_missing_active_pnl():
    index = pd.bdate_range("2020-01-01", periods=5)
    path = pd.DataFrame({"gross_exposure": [0, 0, .2, .3, 0],
                         "gross_return": [np.nan, np.nan, .01, np.nan, 0],
                         "cost": [np.nan, np.nan, .001, .001, .001],
                         "net_return": [np.nan, np.nan, .009, np.nan, -.001]}, index=index)
    fixed = cash_before_first_trade(path, index[1])
    assert np.isnan(fixed.net_return.iloc[0])
    assert fixed.net_return.iloc[1] == 0
    assert np.isnan(fixed.net_return.iloc[3])
    assert fixed.net_return.iloc[4] == -.001
