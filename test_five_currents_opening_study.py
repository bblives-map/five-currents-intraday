from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from five_currents_front_intraday import ROOTS, evaluate
from five_currents_opening_study import (
    allocated_sleeves, lag_contract_fields, opening_moves, opening_predictors,
    session_open,
)


def test_opening_clock_is_singapore_morning_and_handles_dst():
    assert session_open("2026-10-06") == pd.Timestamp("2026-10-05 22:00Z")
    assert session_open("2026-12-08") == pd.Timestamp("2026-12-07 23:00Z")
    assert session_open("2026-10-05").tz_convert("Asia/Singapore").hour == 6


def test_monday_uses_friday_close_not_a_weekend_forecast():
    index = pd.bdate_range("2026-10-01", periods=4)
    weights = pd.DataFrame(np.arange(4)[:, None] * np.ones((1, 5)) / 10,
                           index=index, columns=ROOTS)
    result = opening_predictors(weights, pd.DatetimeIndex(["2026-10-05"]))
    assert result.signal_date.eq(pd.Timestamp("2026-10-02")).all()
    assert result.net_magnitude.eq(.1).all()
    assert result.change_magnitude.eq(.1).all()


def test_close_refresh_change_is_against_previous_final_target():
    index = pd.bdate_range("2026-10-01", periods=4)
    preview = pd.DataFrame(.2, index=index, columns=ROOTS)
    finalized = pd.DataFrame(.1, index=index, columns=ROOTS)
    result = opening_predictors(preview, pd.DatetimeIndex(["2026-10-05"]),
                                previous_weights=finalized)
    assert result.change_magnitude.eq(.1).all()


def test_input_lag_follows_contract_across_rank_roll_and_does_not_forward_fill():
    index = pd.bdate_range("2026-10-01", periods=3)
    columns = pd.MultiIndex.from_product([["CL1 Comdty", "CL2 Comdty"], ["PX_VOLUME", "OPEN_INT", "PX_LAST"]])
    wide = pd.DataFrame([[10, 100, 70, 20, 200, 71], [30, 300, 72, 40, 400, 73],
                         [50, 500, 74, 60, 600, 75]], index=index, columns=columns, dtype=float)
    mapping = pd.DataFrame({"date": np.repeat(index, 2), "ticker": ["CL1 Comdty", "CL2 Comdty"]*3,
                            "contract": ["A", "B", "B", "C", "C", "D"]})
    metadata = pd.DataFrame({"ticker": ["CL1 Comdty", "CL2 Comdty"], "root": ["CL", "CL"], "rank": [1, 2]})
    with patch("five_currents_opening_study.ROOTS", ("CL",)):
        lagged = lag_contract_fields(wide, mapping, metadata)
    assert lagged.iloc[1][("CL1 Comdty", "PX_VOLUME")] == 20
    assert lagged.iloc[1][("CL1 Comdty", "OPEN_INT")] == 200
    assert np.isnan(lagged.iloc[1][("CL2 Comdty", "PX_VOLUME")])
    assert lagged.iloc[2][("CL1 Comdty", "PX_VOLUME")] == 40
    pd.testing.assert_series_equal(lagged[("CL1 Comdty", "PX_LAST")], wide[("CL1 Comdty", "PX_LAST")])
    mutated = wide.copy()
    mutated.loc[index[1]:, (slice(None), ["PX_VOLUME", "OPEN_INT"])] *= 1000
    with patch("five_currents_opening_study.ROOTS", ("CL",)):
        changed = lag_contract_fields(mutated, mapping, metadata)
    pd.testing.assert_series_equal(lagged.iloc[1], changed.iloc[1])


def test_opening_exit_can_cross_utc_midnight_but_stays_in_same_singapore_day():
    day = pd.Timestamp("2026-10-06")
    stamp = session_open(day)
    bars = pd.DataFrame([{"symbol": "CL", "ts": stamp+pd.Timedelta(hours=h),
                          "open": 70+h, "volume": 1, "bar_count": 1,
                          "local_symbol": "CLX6", "expiry": "20261120"}
                         for h in (0, 1, 2, 3, 4, 6, 12)])
    result = opening_moves(bars, [day])
    twelve = result.loc[result.root.eq("CL") & result.hours.eq(12)].iloc[0]
    assert twelve.status == "matched"
    assert twelve.exit_utc == pd.Timestamp("2026-10-06 10:00Z")
    assert twelve.entry_utc.tz_convert("Asia/Singapore").date() == twelve.exit_utc.tz_convert("Asia/Singapore").date()


def test_source_is_chosen_at_entry_not_using_availability_of_exit():
    day = pd.Timestamp("2026-10-06")
    stamp = session_open(day)
    bars = pd.DataFrame([{"symbol": symbol, "ts": stamp+pd.Timedelta(hours=h),
                          "open": 70, "volume": 1, "bar_count": 1,
                          "local_symbol": "CLX6", "expiry": "20261120"}
                         for symbol, h in (("CL", 0), ("MCL", 0), ("MCL", 1))])
    result = opening_moves(bars, [day])
    one = result.loc[result.root.eq("CL") & result.hours.eq(1)].iloc[0]
    assert one.status == "missing exit"


def test_reversal_negates_gross_return_not_cost_and_fee_has_no_extra_multiplier():
    day = pd.Timestamp("2026-10-06")
    predictors = pd.DataFrame([{"session": day, "signal_date": day-pd.Timedelta(days=1),
                                "root": "CL", "net_magnitude": .2, "net_sign": .2,
                                "change_magnitude": .1, "change_sign": .2}])
    moves = pd.DataFrame([{"session": day, "root": "CL", "hours": 1, "status": "matched",
                           "entry_price": 70., "price_return": .01}])
    result = evaluate(predictors, moves).set_index("variant")
    assert result.at["net_magnitude", "commission_return"] == pytest.approx(.2 / (70*100)*2.16)
    assert result.at["reverse_net_magnitude", "gross_return"] == pytest.approx(-.002)
    assert result.at["reverse_net_magnitude", "commission_return"] > 0


def test_unnetted_sleeves_reconcile_and_preserve_exactly_cancelling_cost_exposure():
    index = pd.bdate_range("2026-10-01", periods=3)
    concepts = np.zeros((2, 3, 5))
    concepts[0, :, 0] = .2
    concepts[1, :, 0] = -.2
    concepts[0, :, 1] = .2
    names = ("first", "second")
    selection = pd.DataFrame({"refit_ordinal": [1, 1], "forecast_start": [index[0]]*2,
                              "forecast_end": [index[-1]]*2, "economic_family": names, "selected": [True]*2})
    coefficients = pd.DataFrame({"refit_ordinal": [1]*5, "product": ROOTS,
                                 "inverse_volatility_coefficient": [1.]*5})
    final, sleeves, cancelled = allocated_sleeves(concepts, names, index, selection, coefficients)
    np.testing.assert_allclose(sleeves.sum(axis=0), final, atol=1e-10)
    assert cancelled == 3
    assert np.abs(sleeves[:, :, 0]).sum() > 0
    assert np.all(final[:, 0] == 0)


@pytest.mark.parametrize("values", [[], [0.], [.01], [0., 0.], [.02, -.01, 0, np.nan, .03, -.04]])
def test_fast_metrics_match_audited_reference(values):
    from five_currents_front_intraday import stats as reference
    from five_currents_opening_study import stats as fast
    first, second = reference(values), fast(values)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=1e-12, equal_nan=True)


def test_gross_pnl_reconciliation_aligns_labels_not_groupby_sort_order():
    from five_currents_opening_study import assert_same_gross_pnl
    index = pd.MultiIndex.from_tuples([("2026-10-06", "NG"), ("2026-10-06", "GC")])
    first = pd.Series([.01, -.03], index=index)
    assert_same_gross_pnl(first, first.sort_index())
    with pytest.raises(AssertionError):
        assert_same_gross_pnl(first, first+1.)


def test_vectorized_report_preserves_metrics_missingness_and_zero_positions():
    from five_currents_opening_study import collect_metrics as reference
    from five_currents_opening_metrics import collect_metrics as fast
    days = pd.bdate_range("2026-10-01", periods=5)
    rows = []
    for position, day in enumerate(days):
        for root in ROOTS:
            ready = not (root == "SI" and position == 1)
            gross = .001*(position-2) if ready else np.nan
            rows.append({"session": day, "root": root, "hours": 1, "variant": "net_magnitude",
                         "ready": ready, "weight": .2 if position else 0., "gross_return": gross,
                         "commission_return": .0001 if ready else np.nan,
                         "slippage_return": .0002 if ready else np.nan,
                         "net_return": gross-.0001, "slippage_net_return": gross-.0003})
    paths = pd.DataFrame(rows)
    first, second, curves_first, curves_second = [], [], [], []
    reference(paths, "control", "portfolio_net", first, curves_first)
    fast(paths, "control", "portfolio_net", second, curves_second)
    pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second), check_exact=False, atol=1e-12)
    for left, right in zip(curves_first, curves_second):
        pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12)


def test_policy_comparison_uses_identical_dates_not_zero_filled_missing_results():
    from compare_five_currents_opening_policies import PRIMARY_POLICIES, common_date_metrics
    rows = []
    for policy in PRIMARY_POLICIES:
        for position, day in enumerate(pd.bdate_range("2026-10-01", periods=3)):
            value = np.nan if policy == "price_only" and position == 1 else .001*(position+1)
            rows.append({"session": day, "policy": policy, "signal": "portfolio_net",
                         "price_route": "front", "book": "four", "variant": "net_magnitude",
                         "hours": 1, "gross_return": value, "net_return": value,
                         "slippage_net_return": value})
    result = common_date_metrics(pd.DataFrame(rows))
    assert result.days.eq(2).all()
    assert result.sharpe.nunique() == 1


def test_price_route_comparison_pairs_dates_before_computing_sharpe():
    from audit_five_currents_opening_price_routes import paired_route_metrics
    rows = []
    for route in ("front", "second_series_proxy"):
        for position, day in enumerate(pd.bdate_range("2026-10-01", periods=3)):
            value = .001*(position+1)*(1 if route == "front" else -1)
            if route == "front" and position == 1:
                value = np.nan
            rows.append({"session": day, "policy": "lagged_volume_oi", "signal": "portfolio_net",
                         "price_route": route, "book": "four", "variant": "net_magnitude", "hours": 6,
                         "gross_return": value, "net_return": value, "slippage_net_return": value})
    result = paired_route_metrics(pd.DataFrame(rows))
    assert result.days.eq(2).all()
    assert result.paired_route_net_return_correlation.eq(-1.).all()
