import numpy as np
import pandas as pd
import pytest

from five_currents_consistency_objectives import (
    OBJECTIVES, activity_eligibility, cap_upside, completed_period_returns,
    fitness_scores, objective_admission, objective_value, performance_consistency, rolling_sharpes, win_rate,
)
from five_currents_lightweight_combination import training_sharpe


def test_upside_cap_retains_every_loss_and_nan():
    values = np.r_[np.linspace(-.02, .01, 100), .5, np.nan]
    capped, threshold = cap_upside(values)
    np.testing.assert_array_equal(capped[values < 0], values[values < 0])
    assert capped[-2] == threshold
    assert np.isnan(capped[-1])
    assert np.nanmax(capped) < .02


def test_capped_score_does_not_increase_when_an_existing_extreme_gain_grows():
    rng = np.random.default_rng(41)
    values = rng.normal(0, .001, 504)
    values[200] = .1
    reference = training_sharpe(cap_upside(values)[0])
    values[200] = 100
    assert training_sharpe(cap_upside(values)[0]) == pytest.approx(reference)


def test_rolling_sharpe_waits_for_full_calendar_window_and_requires_marks():
    values = np.linspace(-.01, .02, 100)
    scores = rolling_sharpes(values)
    assert np.isnan(scores[:62]).all()
    assert scores[62] == pytest.approx(training_sharpe(values[:63]))
    values[:14] = np.nan
    assert np.isnan(rolling_sharpes(values)[62])


def test_period_returns_compound_and_exclude_incomplete_week():
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    values = np.full(len(dates), .01)
    result = completed_period_returns(values, dates, dates, dates[-1], "W-FRI")
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(1.01 ** 5 - 1)


def test_missing_mark_invalidates_week_not_imputed_as_zero():
    dates = pd.bdate_range("2024-01-01", "2024-01-19")
    values = np.full(len(dates), .001)
    values[7] = np.nan
    result = completed_period_returns(values, dates, dates, dates[-1], "W-FRI")
    assert list(result.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-19")]


def test_partial_first_week_is_excluded_using_reference_calendar():
    reference = pd.bdate_range("2024-01-01", "2024-01-19")
    dates = reference[2:]
    result = completed_period_returns(np.full(len(dates), .001), dates, reference, dates[-1], "W-FRI")
    assert pd.Timestamp("2024-01-05") not in result.index
    assert len(result) == 2


def test_partial_and_missing_months_excluded():
    reference = pd.bdate_range("2024-01-01", "2024-04-15")
    dates = reference[2:]
    values = np.full(len(dates), .001)
    values[dates.get_loc("2024-03-05")] = np.nan
    result = completed_period_returns(values, dates, reference, dates[-1], "M")
    assert list(result.index) == [pd.Timestamp("2024-02-29")]


def test_future_outcomes_rejected_and_future_reference_dates_do_not_change_periods():
    dates = pd.bdate_range("2024-01-01", "2024-02-29")
    values = np.full(len(dates), .001)
    with pytest.raises(ValueError):
        completed_period_returns(values, dates, dates, "2024-02-28", "M")
    long_reference = pd.bdate_range("2024-01-01", "2030-12-31")
    a = completed_period_returns(values, dates, dates, dates[-1], "M")
    b = completed_period_returns(values, dates, long_reference, dates[-1], "M")
    pd.testing.assert_series_equal(a, b)


def test_flat_cash_periods_do_not_count_as_wins():
    assert win_rate([0, 0, .01, -.01]) == .5
    assert np.isnan(win_rate([0, 0, .01]))


def test_recalibration_uses_no_raw_sharpe_half_gate_for_new_objectives():
    dates = pd.bdate_range("2022-01-03", periods=504)
    values = np.full((1, len(dates)), .001)
    # Many profitable weeks but a losing first half, offset by one second-half gain.
    values[0, 60] = -.30
    values[0, 400] = .40
    exposure = np.ones_like(values)
    original = objective_admission(values, values - .00001, exposure, dates, dates, dates[-1], "sharpe")
    weekly = objective_admission(values, values - .00001, exposure, dates, dates, dates[-1], "weekly")
    assert original.base_first_half.iloc[0] < 0
    assert not original.selected.iloc[0]
    assert weekly.base_first_half.iloc[0] > 0
    assert weekly.selected.iloc[0]


def test_high_win_rate_with_negative_expectancy_is_not_admitted():
    dates = pd.bdate_range("2022-01-03", periods=504)
    values = np.full((1, len(dates)), .0001)
    values[0, 240] = -.5
    result = objective_admission(values, values, np.ones_like(values), dates, dates, dates[-1], "weekly")
    assert result.base_full.iloc[0] > 0
    assert not result.positive_stressed_expectancy.iloc[0]
    assert not result.selected.iloc[0]


def test_halves_do_not_use_period_outcomes_from_later_half():
    dates = pd.bdate_range("2022-01-03", periods=504)
    rng = np.random.default_rng(44)
    values = rng.normal(.0002, .001, len(dates))
    for objective in OBJECTIVES:
        initial = fitness_scores(values, values - .00001, dates, dates, dates[-1], objective)
        changed = values.copy()
        changed[252:] = 100
        later = fitness_scores(changed, changed - .00001, dates, dates, dates[-1], objective)
        assert initial["base_first_half"] == pytest.approx(later["base_first_half"], nan_ok=True)


def test_insufficient_activity_still_fails_independent_of_objective():
    values = np.ones((2, 126)) * .001
    exposure = np.ones_like(values)
    exposure[1, :63] = 0
    np.testing.assert_array_equal(activity_eligibility(values, values, exposure), [True, False])


def test_reported_win_rate_does_not_reward_sitting_in_cash():
    dates = pd.bdate_range("2024-01-01", "2024-01-19")
    values = np.full(len(dates), .001)
    values[5:10] = 0
    metrics = performance_consistency(values, dates, dates, dates[-1])
    assert metrics["weekly_win_rate"] == pytest.approx(2 / 3)
    assert metrics["weekly_active_win_rate"] == 1
    assert metrics["weekly_observations"] == 3
    assert metrics["weekly_nonflat_observations"] == 2


def test_rolling_median_rejects_path_rescued_by_one_exceptional_gain():
    rng = np.random.default_rng(81)
    values = rng.normal(-.0002, .001, 504)
    values[250] = .5
    assert training_sharpe(values) > 0
    assert np.nanmedian(rolling_sharpes(values)) < 0
