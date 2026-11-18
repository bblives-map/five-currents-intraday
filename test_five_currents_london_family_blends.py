import numpy as np
import pandas as pd

from run_five_currents_london_family_blends import mixed_exit_legs


def fixture(core, dryup, missing_twelve=False):
    session = pd.Timestamp("2026-10-12")
    components = pd.DataFrame([{"session": session, "signal_date": pd.Timestamp("2026-10-09"),
                                "root": "CL", "mixed_core": core, "mixed_dryup": dryup}])
    rows = []
    for hours, price in ((6, 101.), (12, 102.)):
        rows.append({"session": session, "root": "CL", "hours": hours, "entry_price": 100.,
                     "exit_price": np.nan if hours == 12 and missing_twelve else price,
                     "status": "missing same-contract exit" if hours == 12 and missing_twelve else "matched",
                     "entry_ts": pd.Timestamp("2026-10-12 07:00:00+00:00"), "local_symbol": "MCLX6",
                     "trade_pair_confirmed": not (hours == 12 and missing_twelve), "identity_basis": "verified_month"})
    return components, pd.DataFrame(rows)


def test_partial_exit_pnl_and_commission_follow_netted_position_path():
    components, moves = fixture(.09, -.03)
    row = mixed_exit_legs(components, moves).iloc[0]
    assert row.ready
    np.testing.assert_allclose(row.gross_return, .06 * .01 + .09 * .01, atol=1e-15)
    np.testing.assert_allclose(row.commission_return, (.06 + .03 + .09) * 1.08 / (100 * 100), atol=1e-15)
    np.testing.assert_allclose(row.slippage_return, (.06 + .03 + .09) * .01 / 100, atol=1e-15)


def test_opposing_entry_sleeves_pay_only_for_later_orders():
    components, moves = fixture(.04, -.04)
    row = mixed_exit_legs(components, moves).iloc[0]
    assert row.weight == 0.
    np.testing.assert_allclose(row.gross_return, .04 * .01, atol=1e-15)
    np.testing.assert_allclose(row.commission_return, (.04 + .04) * 1.08 / (100 * 100), atol=1e-15)


def test_dryup_only_trade_does_not_require_unused_twelve_hour_exit():
    components, moves = fixture(0., .03, missing_twelve=True)
    row = mixed_exit_legs(components, moves).iloc[0]
    assert row.ready
    np.testing.assert_allclose(row.gross_return, .03 * .01, atol=1e-15)
    np.testing.assert_allclose(row.commission_return, .03 * 2.16 / (100 * 100), atol=1e-15)
