import numpy as np
import pandas as pd

from audit_five_currents_cached_replay import candidate_leave_one_out, leg_values


def test_reversed_change_uses_micro_round_trip_cost_once():
    joined = pd.DataFrame([{
        "session": pd.Timestamp("2026-10-12"), "clock": "us_open", "hours": 4,
        "root": "CL", "identity_basis": "verified_month", "trade_pair_confirmed": True,
        "status": "matched", "entry_price": 75., "price_return": .01,
        "change_magnitude": .04,
    }])
    result = leg_values(joined, "reverse_change_magnitude").iloc[0]
    assert result.ready
    assert result.weight == -.04
    np.testing.assert_allclose(result.gross_return, -.04 * .01, atol=1e-16)
    np.testing.assert_allclose(result.commission_return, .04 * 2.16 / (75 * 100), atol=1e-16)
    np.testing.assert_allclose(result.slippage_return, .04 * 2 * .01 / 75, atol=1e-16)
    np.testing.assert_allclose(result.slippage_net_return,
                               result.gross_return - result.commission_return
                               - result.slippage_return, atol=1e-16)


def test_inactive_asset_needs_no_price_and_pays_no_cost():
    joined = pd.DataFrame([{
        "session": pd.Timestamp("2026-10-12"), "clock": "us_open", "hours": 4,
        "root": "GC", "identity_basis": "verified_month", "trade_pair_confirmed": None,
        "status": "missing entry", "entry_price": np.nan, "price_return": np.nan,
        "change_magnitude": 0.,
    }])
    result = leg_values(joined, "reverse_change_magnitude").iloc[0]
    assert result.ready
    assert result.slippage_net_return == 0.


def test_leave_one_out_uses_same_dates_without_rescaling():
    days = pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-12"])
    values = {"CL": (.01, .02, -.01), "NG": (-.02, .01, .01),
              "GC": (.01, -.01, .02), "HG": (.02, .02, .01)}
    rows = [{"session": day, "root": root, "hours": 4, "slippage_net_return": value}
            for root, series in values.items() for day, value in zip(days, series)]
    result = candidate_leave_one_out(pd.DataFrame(rows)).set_index("omitted_asset")
    assert result.loc["none", "days"] == 3
    full = sum(np.array(v) for v in values.values())
    without_hg = full - np.array(values["HG"])
    np.testing.assert_allclose(result.loc["none", "total_return"], np.prod(1 + full) - 1)
    np.testing.assert_allclose(result.loc["HG", "total_return"], np.prod(1 + without_hg) - 1)
