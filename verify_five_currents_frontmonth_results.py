"""Check derived front-month fill ledgers without accessing the database."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import CLOCKS, DELAYS, HOLD_MINUTES, SOURCE


FRONT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_frontmonth_20261025"
PRODUCTS = ("CL", "NG", "GC", "HG")


def main():
    sessions = pd.DatetimeIndex(pd.read_parquet(SOURCE / "sessions.parquet").session)
    routes = pd.read_parquet(FRONT / "route_selection_liquidity.parquet")
    if len(routes) != len(sessions) * len(PRODUCTS):
        raise AssertionError("Missing or duplicate route decisions")
    if routes.duplicated(["root", "session"]).any():
        raise AssertionError("Repeated route decision")
    prior = routes.latest_information_session.notna()
    if not (routes.loc[prior, "latest_information_session"] <
            routes.loc[prior, "session"]).all():
        raise AssertionError("Route decision looks ahead")
    with np.load(FRONT / "front_clock_prices_liquidity.npz") as saved:
        ret = saved["returns"]
        entry = saved["entry"]
    expected_shape = (len(CLOCKS), len(HOLD_MINUTES), len(DELAYS),
                      len(sessions), len(ROOTS))
    if ret.shape != expected_shape or entry.shape != expected_shape:
        raise AssertionError("Price cube shape differs from frozen sessions")
    if not np.array_equal(np.isfinite(ret), np.isfinite(entry)):
        raise AssertionError("Return/entry missingness differs")
    summary = []
    for root in PRODUCTS:
        ai = ROOTS.index(root)
        matched = 0
        counts = {}
        for path in sorted((FRONT / "fill_ledgers_liquidity" / root).glob("*.parquet")):
            frame = pd.read_parquet(path)
            month = pd.Period(f"{path.stem[:4]}-{path.stem[4:]}", freq="M")
            n_sessions = (sessions.to_period("M") == month).sum()
            if len(frame) != n_sessions * len(CLOCKS) * len(HOLD_MINUTES) * len(DELAYS):
                raise AssertionError(f"Missing ledger rows: {path}")
            actual = frame.loc[frame.status.eq("matched")]
            matched += len(actual)
            for status, count in frame.status.value_counts().items():
                counts[status] = counts.get(status, 0) + int(count)
            if actual.empty:
                continue
            if actual.entry_contract.isna().any() or actual.entry_price.le(0).any():
                raise AssertionError(f"Invalid entry identity/price: {path}")
            if actual.exit_price.le(0).any() or actual.exit_time.le(actual.entry_time).any():
                raise AssertionError(f"Invalid exit: {path}")
            entry_gap = (actual.entry_time - actual.planned_entry).dt.total_seconds() / 60
            exit_gap = (actual.exit_time - actual.planned_exit).dt.total_seconds() / 60
            if not entry_gap.between(1, 11).all() or not exit_gap.between(1, 11).all():
                raise AssertionError(f"Fill outside forward ten-minute bar window: {path}")
            np.testing.assert_allclose(actual.price_return.to_numpy(),
                                       actual.exit_price.to_numpy() /
                                       actual.entry_price.to_numpy() - 1)
        if matched != int(np.isfinite(ret[..., ai]).sum()):
            raise AssertionError(f"Cube/ledger match count differs for {root}")
        summary.append({"root": root, "matched_cells": matched,
                        "total_cells": int(ret[..., ai].size), "statuses": counts})
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(),
              "sessions": len(sessions), "products": PRODUCTS,
              "route_decisions_past_only": True,
              "minute_fills_forward_ten_minute_window": True,
              "cube_matches_ledgers": True,
              "results": summary}
    (FRONT / "verification.json").write_text(json.dumps(result, indent=2),
                                               encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
