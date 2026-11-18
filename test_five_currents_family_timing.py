import numpy as np
import pytest

from five_currents_family_timing import (
    fitness, select_recipes, family_allocation, fit_schedule, simulate,
)
from five_currents_execution_timing import causal_choice


def candidates(days=260, families=4, recipes=3):
    rng = np.random.default_rng(102)
    base = rng.normal(.0003, .001, (days, families, recipes, 1))
    return np.repeat(base, 3, axis=3)


def test_individual_family_admission_matches_original_rule():
    x = candidates()
    chosen, _, _ = select_recipes(x)
    for family in range(x.shape[1]):
        assert chosen[family] == causal_choice(x[:, family], lookback=None)[0]


def test_causal_schedule_is_prefix_invariant_and_embargoed():
    x = candidates()
    c, alpha, audit, first = fit_schedule(x, refit=5)
    changed = x.copy()
    changed[200:] *= -100
    d, beta, _, _ = fit_schedule(changed, refit=5)
    np.testing.assert_array_equal(c[:200], d[:200])
    for method in alpha:
        np.testing.assert_allclose(alpha[method][:200], beta[method][:200])
        assert (alpha[method] >= 0).all()
        assert (alpha[method].sum(axis=1) <= 1+1e-12).all()
    assert first == 127 and (c[:first] == -1).all()
    assert (audit.train_end_exclusive < audit.start).all()


def test_frozen_split_never_uses_test_outcomes():
    x = candidates(days=600)
    c, alpha, _, cut = fit_schedule(x, frozen_fraction=.3)
    changed = x.copy()
    changed[cut-1:] = 100
    d, beta, _, _ = fit_schedule(changed, frozen_fraction=.3)
    assert (c[cut:] >= 0).any()
    np.testing.assert_array_equal(c, d)
    for method in alpha:
        np.testing.assert_allclose(alpha[method], beta[method])


def test_training_coverage_and_both_half_gates():
    x = np.ones((150, 2, 3))*.001
    x[:, 0] += np.sin(np.arange(150))[:, None]*.0002
    x[:75, 1] = -.001
    x[75:, 1] = .01
    _, eligible = fitness(x)
    assert eligible.tolist() == [True, False]
    x[:80, 0] = np.nan
    assert not fitness(x)[1].any()


def test_greedy_adds_a_diversifier_but_not_a_duplicate():
    rng = np.random.default_rng(88)
    x = rng.normal(.0004, .002, (260, 1))
    y = .0004-(x-.0004)+rng.normal(0, .0001, x.shape)
    paths = np.stack([np.repeat(x, 3, axis=1), np.repeat(y, 3, axis=1),
                      np.repeat(x, 3, axis=1)], axis=1)
    scores, eligible = fitness(paths)
    allocation = family_allocation(paths, np.where(eligible, scores, np.nan), "greedy_three")
    assert allocation[1] > 0
    assert (allocation > 0).sum() == 2


def setup_simulation():
    weights = np.zeros((2, 4, 3, 5))
    weights[:, :, :, 0] = .2
    choices = np.zeros((3, 5, 2), dtype=int)
    alpha = np.zeros_like(choices, dtype=float)
    alpha[:, 0] = .5
    cube = {"returns": np.full((1, 4, 3, 5), .01), "entry": np.full((1, 4, 3, 5), 100.)}
    return weights, cube, choices, alpha


def test_opposing_orders_net_to_cash_even_when_prices_missing():
    w, cube, c, a = setup_simulation()
    w[1] *= -1
    cube["returns"][:] = np.nan
    cube["entry"][:] = np.nan
    out = simulate(w, cube, c, a)
    np.testing.assert_allclose(out["net"], 0)
    np.testing.assert_allclose(out["cost"], 0)


def test_same_recipe_netting_fee_and_additive_attribution():
    w, cube, c, a = setup_simulation()
    w[1] *= -.5
    out = simulate(w, cube, c, a)
    assert out["gross"][0] == pytest.approx(.05*.01)
    assert out["cost"][0] == pytest.approx(.05*(2.16/10000+2*.01/100))
    assert out["independent_cost"][0] == pytest.approx(3*out["cost"][0])
    np.testing.assert_allclose(out["by_family"].sum(axis=1), out["net"])
    np.testing.assert_allclose(out["by_asset"]["net"].sum(axis=1), out["net"])


def test_different_windows_do_not_net_each_others_costs():
    w, cube, c, a = setup_simulation()
    cube = {k: np.repeat(v, 2, axis=0) for k, v in cube.items()}
    w[1] *= -1
    c[:, :, 1] = 1
    out = simulate(w, cube, c, a)
    assert (out["cost"] > 0).all()
    np.testing.assert_allclose(out["gross"], 0)


def test_active_missing_returns_stay_missing_and_cash_stays_zero():
    w, cube, c, a = setup_simulation()
    cube["returns"][0, 0, 1, 0] = np.nan
    out = simulate(w, cube, c, a)
    assert np.isnan(out["net"][1])
    a[:] = 0
    np.testing.assert_allclose(simulate(w, cube, c, a)["net"], 0)


def test_gross_cap_does_not_use_price_availability():
    w, cube, c, a = setup_simulation()
    w[:] = .5
    a[:] = .5
    out = simulate(w, cube, c, a)
    assert (out["exposure"] <= 1+1e-12).all()
    cube["returns"][:, :, :, 0] = np.nan
    other = simulate(w, cube, c, a)
    np.testing.assert_allclose(other["cap"], out["cap"])
    assert np.isnan(other["net"]).all()


def test_maximum_three_families_and_nonnegative_equal_weights():
    x = candidates(families=10)
    _, score, paths = select_recipes(x)
    alpha = family_allocation(paths, score, "greedy_three")
    assert 1 <= (alpha > 0).sum() <= 3
    np.testing.assert_allclose(alpha[alpha > 0], 1/(alpha > 0).sum())
