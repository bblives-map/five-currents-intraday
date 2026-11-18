from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from five_currents_lightweight_combination import (
    SCORE_KEYS, TrainingBook, allocate_refits, apply_product_scaling, blend_concepts,
    correlation_discount, greedy_backward, normalize_votes, residual_diagnostic,
)
from us_commodity_disciplined_core import ProductExecution
from us_commodity_pooled_harmonization_core import restore_gross_after_product_tilt
from us_commodity_research_core import PRODUCT_ORDER, _cap_and_normalize_rows


def test_equal_votes_match_nanmean_and_empty_book_is_cash():
    rng = np.random.default_rng(7)
    concepts = rng.normal(size=(3, 60, 5))
    concepts[0, 5, 2] = np.nan
    np.testing.assert_allclose(blend_concepts(concepts, [1, 1, 1]), np.nanmean(concepts, axis=0))
    np.testing.assert_array_equal(blend_concepts(concepts, [0, 0, 0]), np.zeros((60, 5)))


def test_votes_reject_negative_or_nonfinite_values():
    for invalid in ([-1, 2], [np.nan, 1], [np.inf, 0]):
        with pytest.raises(ValueError):
            normalize_votes(invalid)


def test_correlation_discount_penalizes_duplicate_pair_not_unique_signal():
    rng = np.random.default_rng(8)
    one, independent = rng.normal(size=(2, 500))
    votes, redundancy, audit = correlation_discount(np.stack([one, 3 * one, independent]), [0, 1, 2])
    assert votes[2] > votes[0]
    assert votes[0] == pytest.approx(votes[1])
    assert redundancy[0] >= 1
    assert len(audit) == 3
    assert votes.sum() == pytest.approx(1)


def test_negative_correlation_is_not_penalized_and_ineligible_votes_are_zero():
    one = np.arange(100, dtype=float)
    votes, redundancy, _ = correlation_discount(np.stack([one, -one, one]), [0, 1])
    np.testing.assert_allclose(votes, [.5, .5, 0])
    np.testing.assert_allclose(redundancy, 0)


def test_spearman_discount_invariant_to_monotonic_transformation():
    rng = np.random.default_rng(13)
    values = rng.uniform(.01, .99, size=(4, 400))
    original = correlation_discount(values, [0, 1, 2, 3])[0]
    transformed = correlation_discount(np.exp(values), [0, 1, 2, 3])[0]
    np.testing.assert_allclose(original, transformed)


def test_unknown_correlation_is_logged_not_an_elimination_gate():
    values = np.stack([np.arange(50, dtype=float), np.full(50, np.nan)])
    votes, _, audit = correlation_discount(values, [0, 1])
    np.testing.assert_allclose(votes, [.5, .5])
    assert audit[0]["unknown_correlation"]
    assert audit[0]["observations"] == 0


def fake_evaluation(votes):
    members = tuple(np.flatnonzero(votes))
    score = {(0, 1, 2): 1, (0, 1): 2, (0, 2): .5, (1, 2): .5,
             (0,): 1, (1,): 1, (2,): 0}[members]
    return SimpleNamespace(scores=dict.fromkeys(SCORE_KEYS, score))


def test_greedy_removes_only_consistently_detrimental_concept_then_stops():
    votes, audit = greedy_backward([0, 1, 2], 3, fake_evaluation)
    np.testing.assert_allclose(votes, [.5, .5, 0])
    accepted = [row for row in audit if row["accepted"]]
    assert len(accepted) == 1
    assert accepted[0]["removed_index"] == 2
    assert len(audit) == 5


def test_greedy_rejects_improvement_that_fails_one_half_or_cost_check():
    def evaluate(votes):
        result = fake_evaluation(votes)
        if tuple(np.flatnonzero(votes)) == (0, 1):
            result.scores["stressed_second_half"] = .99
        return result
    votes, audit = greedy_backward([0, 1, 2], 3, evaluate)
    np.testing.assert_allclose(votes, [1 / 3] * 3)
    assert not any(row["accepted"] for row in audit)


def test_greedy_keeps_single_concept_and_existing_cash():
    votes, audit = greedy_backward([1], 3, lambda x: pytest.fail("Should not evaluate removals"))
    np.testing.assert_allclose(votes, [0, 1, 0])
    assert not audit
    votes, audit = greedy_backward([], 3, lambda x: pytest.fail("Should not evaluate removals"))
    assert not votes.any()
    assert not audit


def test_residual_retains_alpha_and_removes_beta_exposure():
    rng = np.random.default_rng(33)
    x = rng.normal(.0002, .01, 500)
    noise = rng.normal(size=500)
    noise -= noise.mean()
    noise -= np.mean(noise * (x - x.mean())) / np.var(x) * (x - x.mean())
    y = .0005 + .7 * x + .001 * noise
    result = residual_diagnostic(y, x)
    assert result["beta"] == pytest.approx(.7)
    assert result["annualized_alpha"] == pytest.approx(.0005 * 252)
    assert result["residual_sharpe"] > 0
    centered_residual = y - (y.mean() - .7 * x.mean()) - .7 * x
    assert abs(centered_residual.mean()) < 1e-12


def test_residual_undefined_for_no_rest_portfolio():
    result = residual_diagnostic(np.arange(100), np.zeros(100))
    assert np.isnan(result["residual_sharpe"])


def fixture_book():
    rng = np.random.default_rng(14)
    index = pd.bdate_range("2020-01-02", periods=190)
    concepts = rng.normal(0, .05, size=(3, len(index), 5))
    target = rng.normal(.0002, .01, size=(len(index), 5))
    rate = np.full_like(target, .0001)
    identity = np.full(target.shape, "A", dtype=object)
    identity[100:] = "B"
    return concepts, index, ProductExecution(target, rate, 2 * rate, identity)


def test_training_decisions_ignore_all_future_inputs_and_alias_mutations():
    concepts, index, execution = fixture_book()
    positions = np.arange(10, 150)
    evaluator = TrainingBook(concepts, index, execution, positions)
    original = evaluator.evaluate([1, 1, 1])
    concepts[:, 150:] = 1234
    execution.target_return[150:] = -999
    execution.base_cost_rate[150:] = 10
    execution.stressed_cost_rate[150:] = 20
    execution.identity[150:] = "FUTURE"
    later = TrainingBook(concepts, index, execution, positions).evaluate([1, 1, 1])
    np.testing.assert_allclose(original.base, later.base)
    np.testing.assert_allclose(original.coefficients, later.coefficients)
    assert original.scores == later.scores
    concepts[:, :150] = -777
    np.testing.assert_allclose(evaluator.concepts, fixture_book()[0][:, :150])


def test_training_candidates_share_missing_mark_mask():
    concepts, index, execution = fixture_book()
    execution.target_return[40, 0] = np.nan
    concepts[1:, 40, 0] = 0
    evaluator = TrainingBook(concepts, index, execution, np.arange(10, 150))
    assert np.isnan(evaluator.evaluate([1, 1, 1]).base[30])
    assert np.isnan(evaluator.evaluate([0, 1, 1]).base[30])


def test_scaling_respects_product_and_gross_caps():
    rng = np.random.default_rng(20)
    raw = rng.normal(size=(300, 5))
    weights = apply_product_scaling(raw, np.array([.9, .025, .025, .025, .025]))
    assert np.abs(weights).max() <= .4 + 1e-12
    assert np.abs(weights).sum(axis=1).max() <= 1 + 1e-12


def test_batched_scaling_matches_original_row_bisection():
    rng = np.random.default_rng(123)
    raw = rng.normal(0, .2, size=(500, 5))
    raw[0] = 0
    raw[1, :4] = 0
    coefficients = np.array([.7, .1, .08, .07, .05])
    unscaled = _cap_and_normalize_rows(raw, maximum_gross=1, product_cap=.4)
    tilted = _cap_and_normalize_rows(raw * coefficients, maximum_gross=1, product_cap=.4)
    expected = restore_gross_after_product_tilt(unscaled, tilted, maximum_gross=1, product_cap=.4)
    np.testing.assert_allclose(apply_product_scaling(raw, coefficients), expected, rtol=1e-13, atol=1e-14)


def test_allocator_holds_votes_and_coefficients_between_refits():
    concepts, index, _ = fixture_book()
    names = ("a", "b", "c")
    audits, coefficients = [], []
    for ordinal, start, end, votes in ((1, 130, 150, [.5, .5, 0]), (2, 151, 170, [0, 0, 1])):
        for name, vote in zip(names, votes, strict=True):
            audits.append({"refit_ordinal": ordinal, "forecast_start": index[start], "forecast_end": index[end],
                           "economic_family": name, "concept_vote": vote})
        for product in PRODUCT_ORDER:
            coefficients.append({"refit_ordinal": ordinal, "product": product, "inverse_volatility_coefficient": .2})
    output = allocate_refits(concepts, names, index, pd.DataFrame(audits), pd.DataFrame(coefficients))
    np.testing.assert_allclose(output[130:151], apply_product_scaling(np.mean(concepts[:2, 130:151], axis=0), np.full(5, .2)))
    np.testing.assert_allclose(output[151:171], apply_product_scaling(concepts[2, 151:171], np.full(5, .2)))
    assert not output[:130].any()
    assert not output[171:].any()
