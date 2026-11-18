from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from five_currents_intraday_replay import (
    ROOTS, HORIZONS, entry_time, identity_key, signal_legs, match_bars,
    usable_bar, daily_paths, summarize, reverse_paths, standard_contract_costs, recost_matches,
    micro_contract_costs, MICROS, MULTIPLIERS,
)


@pytest.mark.parametrize("day,hour",[("2026-07-15",6),("2026-01-15",7),
                                      ("2026-03-06",7),("2026-03-09",6),
                                      ("2026-10-30",6),("2026-11-02",7)])
def test_chicago_entry_observes_us_not_european_dst(day,hour):
    assert entry_time(day) == pd.Timestamp(day+f"T{hour:02d}:00:00Z")


def fixture_signals():
    index=pd.bdate_range("2026-10-01",periods=4)
    weights=pd.DataFrame(0.,index=index,columns=ROOTS)
    weights["CL"]=[.10,.25,-.15,.95]
    identities=pd.DataFrame({root:[root+"X6"]*len(index) for root in ROOTS},index=index)
    return weights,identities


def endpoint_bars(leg, *, symbol="MCL", table="bars_1m", sign=1):
    return pd.DataFrame([{"table":table,"symbol":symbol,"contract_key":leg["contract_key"],
                          "ts":leg["entry_utc"]+pd.Timedelta(minutes=h),"open":100.+sign*h/60,
                          "volume":100.,"bar_count":10}
                         for h in (0,*HORIZONS)])


def test_prior_close_only_and_target_change_are_different():
    w,identities=fixture_signals()
    legs=signal_legs(w,identities,start="2026-10-05",end="2026-10-05")
    assert legs.loc[legs["mode"].eq("target"),"weight"].iloc[0] == .25
    assert legs.loc[legs["mode"].eq("change"),"weight"].iloc[0] == pytest.approx(.15)
    assert legs.signal_date.eq(pd.Timestamp("2026-10-02")).all()


def test_contract_switch_has_old_exit_and_new_entry_not_false_same_month_delta():
    w,identities=fixture_signals()
    identities.loc[pd.Timestamp("2026-10-02"),"CL"]="CLZ6"
    legs=signal_legs(w,identities,start="2026-10-05",end="2026-10-05")
    changed=legs.loc[legs["mode"].eq("change")].set_index("contract_key").weight.to_dict()
    assert changed == {"CL:2026-12":.25,"CL:2026-11":-.10}


@pytest.mark.parametrize("root,contract,key",[("NG","MNGX6","NG:2026-11"),
                                               ("GC","MGCZ26","GC:2026-12"),
                                               ("HG","MHGZ6","HG:2026-12")])
def test_micro_aliases_match_exact_delivery_month(root,contract,key):
    assert identity_key(root,contract,"2026-10-05") == key


def test_unknown_identity_is_not_guessed():
    assert identity_key("CL","CLunknown","2026-10-05") is None


def test_two_unknown_months_do_not_silently_net_or_overwrite():
    weights, identities = fixture_signals()
    identities["CL"] = "unknown"
    legs = signal_legs(weights, identities, start="2026-10-05", end="2026-10-05")
    change = legs.loc[legs["mode"].eq("change")]
    assert sorted(change.weight) == [-.10, .25]
    assert change.contract_key.isna().all()


@pytest.mark.parametrize("volume,count,valid",[(0.,10,False),(100.,0,False),(100.,10,True)])
def test_forward_filled_or_untraded_bars_are_rejected(volume,count,valid):
    assert usable_bar(SimpleNamespace(open=100.,volume=volume,bar_count=count)) is valid


def test_signed_return_and_roundtrip_costs_from_minute_open():
    w,ids=fixture_signals()
    legs=signal_legs(w,ids,start="2026-10-05",end="2026-10-05")
    bars=endpoint_bars(legs.iloc[0])
    result=match_bars(legs,bars)
    one=result.loc[result["mode"].eq("target") & result.horizon.eq(60)].iloc[0]
    assert one.gross_return == pytest.approx(.25*.01)
    assert one.cost_return == pytest.approx(.25*(2*.01+5.50/1000)/100)
    assert one.execution_symbol == "CL"
    assert one.execution_multiplier == 1000.
    assert one.net_return == pytest.approx(one.gross_return-one.cost_return)
    assert one.stressed_return == pytest.approx(one.gross_return-2*one.cost_return)


def test_reversal_negates_exposure_but_never_execution_costs():
    original = pd.DataFrame({"gross_return": [.01, -.02, np.nan, 0.],
                             "cost_return": [.001, .002, np.nan, 0.]})
    original["net_return"] = original.gross_return - original.cost_return
    original["stressed_return"] = original.gross_return - 2 * original.cost_return
    reversed_path = reverse_paths(original)
    np.testing.assert_allclose(reversed_path.net_return, [-.011, .018, np.nan, 0.], equal_nan=True)
    np.testing.assert_allclose(reversed_path.stressed_return, [-.012, .016, np.nan, 0.], equal_nan=True)
    np.testing.assert_allclose(reversed_path.net_return + original.net_return,
                               -2 * original.cost_return, equal_nan=True)
    pd.testing.assert_frame_equal(reverse_paths(reversed_path), original)


def test_no_wrong_expiry_or_future_exit_driven_source_selection():
    w,ids=fixture_signals()
    legs=signal_legs(w,ids,start="2026-10-05",end="2026-10-05")
    micro=endpoint_bars(legs.iloc[0])
    standard=endpoint_bars(legs.iloc[0],symbol="CL").iloc[:1]
    matched=match_bars(legs,pd.concat([standard,micro],ignore_index=True))
    assert matched.status.ne("matched").all()
    assert matched.symbol.eq("CL").all()
    wrong=micro.assign(contract_key="CL:2026-12")
    assert match_bars(legs,wrong).status.ne("matched").all()


def test_all_horizons_matched_sample_and_missing_outcomes_not_imputed():
    w,ids=fixture_signals()
    legs=signal_legs(w,ids,start="2026-10-05",end="2026-10-06")
    all_bars=pd.concat([endpoint_bars(legs.loc[legs.session.eq(day)].iloc[0]) for day in legs.session.unique()],ignore_index=True)
    missing=entry_time("2026-10-06")+pd.Timedelta(minutes=360)
    all_bars=all_bars.loc[all_bars.ts.ne(missing)]
    matches=match_bars(legs,all_bars)
    paths=daily_paths(matches,pd.to_datetime(["2026-10-05","2026-10-06"]))
    missing_returns=paths.loc[paths.root.eq("CL") & paths.session.eq(pd.Timestamp("2026-10-06")) & paths.horizon.eq(360)]
    assert missing_returns.net_return.isna().all()
    metrics,curves=summarize(paths, sample="all-horizons")
    assert metrics.loc[metrics.book.eq("portfolio"),"days"].eq(1).all()
    assert metrics.loc[metrics.book.eq("CL"),"days"].eq(1).all()
    assert curves.loc[curves.book.eq("portfolio"),"session"].eq(pd.Timestamp("2026-10-05")).all()
    independent, _ = summarize(paths)
    cl = independent.loc[independent.book.eq("CL") & independent["mode"].eq("target")].set_index("horizon")
    assert cl.at[60, "days"] == 2
    assert cl.at[360, "days"] == 1


@pytest.mark.parametrize("root,price,slip_dollars", [("CL", 70., 20.), ("NG", 3., 20.),
                                                   ("GC", 3000., 20.), ("HG", 5., 25.),
                                                   ("SI", 35., 50.)])
def test_costs_reconcile_to_one_standard_lot_in_dollars(root, price, slip_dollars):
    from five_currents_intraday_replay import MULTIPLIERS
    capital = 1_000_000.
    weight = price * MULTIPLIERS[root] / capital
    cost = standard_contract_costs(root, price, weight)
    assert cost["commission_return"] * capital == pytest.approx(5.50)
    assert cost["slippage_return"] * capital == pytest.approx(slip_dollars)
    assert cost["cost_return"] * capital == pytest.approx(5.50 + slip_dollars)
    assert standard_contract_costs(root, price, -weight)["cost_return"] == pytest.approx(cost["cost_return"])


def test_recost_saved_micro_price_proxy_preserves_gross_and_source():
    weights, identities = fixture_signals()
    legs = signal_legs(weights, identities, start="2026-10-05", end="2026-10-05")
    matched = match_bars(legs, endpoint_bars(legs.iloc[0]))
    # Simulate the obsolete fee coupling, then verify deterministic correction.
    obsolete = matched.copy()
    obsolete["cost_return"] = obsolete.weight.abs() * (2*.01 + 2*1.60/100) / obsolete.entry_price
    corrected = recost_matches(obsolete)
    np.testing.assert_allclose(corrected.cost_return, matched.cost_return)
    pd.testing.assert_series_equal(corrected.gross_return, obsolete.gross_return)
    assert corrected.symbol.eq("MCL").all()
    assert corrected.execution_symbol.eq("CL").all()


@pytest.mark.parametrize("root,price", [("CL", 70.), ("NG", 3.), ("GC", 3000.),
                                      ("HG", 5.), ("SI", 35.)])
@pytest.mark.parametrize("quantity", [1., 3., -1.])
def test_supplied_micro_fee_is_216_per_round_trip_lot_without_conversion(root, price, quantity):
    capital = 1_000_000.
    multiplier = MULTIPLIERS[MICROS[root]]
    weight = quantity * price * multiplier / capital
    costs = micro_contract_costs(root, price, weight)
    assert costs["commission_usd_per_round_trip_lot"] == 2.16
    assert costs["commission_return"] * capital == pytest.approx(abs(quantity) * 2.16)
    assert costs["quantity_per_unit_capital"] * capital == pytest.approx(abs(quantity))
    exit_price = price + .1
    gross_dollars = weight * (exit_price / price - 1.) * capital
    assert gross_dollars == pytest.approx(quantity * multiplier * .1)


def test_recost_to_micro_preserves_source_and_charges_micro_lots():
    weights, identities = fixture_signals()
    legs = signal_legs(weights, identities, start="2026-10-05", end="2026-10-05")
    matched = match_bars(legs, endpoint_bars(legs.iloc[0], symbol="CL"))
    micro = recost_matches(matched, execution_basis="micro")
    assert micro.symbol.eq("CL").all()
    assert micro.execution_symbol.eq("MCL").all()
    assert micro.commission_usd_per_round_trip_lot.eq(2.16).all()
    np.testing.assert_allclose(micro.commission_return, micro.quantity_per_unit_capital * 2.16)
    np.testing.assert_allclose(micro.gross_return, matched.gross_return)
