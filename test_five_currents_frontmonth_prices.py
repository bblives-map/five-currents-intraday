"""Small causality and contract-identity checks for front-month minute fills."""
from __future__ import annotations

import unittest

import pandas as pd

from build_five_currents_frontmonth_prices import FrontRouteBook, fill_pair, prior_volume_route


SESSION = pd.Timestamp("2026-08-03")
CALENDAR = pd.bdate_range("2026-07-01", "2026-09-30")


def book(rows):
    frame = pd.DataFrame(rows, columns=(
        "table", "ts", "local_symbol", "close", "volume", "expiry"))
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, format="mixed")
    return FrontRouteBook(frame)


class FrontMonthFillsTest(unittest.TestCase):
    def test_zero_volume_skipped_and_same_contract_exit_crosses_route(self):
        source = book([
            ("bars_1m", "2026-08-03 01:00+00", "MGCQ6", 4000, 0, 20260827),
            ("bars_1m", "2026-08-03 01:02+00", "MGCQ6", 4001, 2, 20260827),
            ("bars_1m_2m", "2026-08-03 02:03+00", "MGCQ6", 4010, 1, 20260827),
        ])
        row = fill_pair(source, SESSION, "GC", "asia_open", 60, 0, CALENDAR)
        self.assertEqual(row["status"], "matched")
        self.assertEqual(row["entry_contract"], "MGCQ6")
        self.assertEqual(row["entry_time"], pd.Timestamp("2026-08-03 01:03+00"))
        self.assertEqual(row["exit_table"], "bars_1m_2m")
        self.assertFalse(row["roll_bridge"])

    def test_pre_expiry_moves_only_to_second_nearby(self):
        source = book([
            ("bars_1m", "2026-08-03 01:00+00", "MGCQ6", 4000, 1, 20260804),
            ("bars_1m_2m", "2026-08-03 01:00+00", "MGCV6", 4002, 1, 20261027),
            ("bars_1m_2m", "2026-08-03 02:01+00", "MGCV6", 4004, 1, 20261027),
        ])
        row = fill_pair(source, SESSION, "GC", "asia_open", 60, 0, CALENDAR)
        self.assertEqual(row["status"], "matched")
        self.assertTrue(row["roll_bridge"])
        self.assertEqual(row["entry_contract"], "MGCV6")

    def test_different_contract_is_not_an_exit(self):
        source = book([
            ("bars_1m", "2026-08-03 01:00+00", "MGCQ6", 4000, 1, 20260827),
            ("bars_1m", "2026-08-03 02:01+00", "MGCV6", 4010, 1, 20261027),
            ("bars_1m_2m", "2026-08-03 02:01+00", "MGCZ6", 4020, 1, 20261229),
        ])
        row = fill_pair(source, SESSION, "GC", "asia_open", 60, 0, CALENDAR)
        self.assertEqual(row["status"], "missing same-contract exit")

    def test_prior_volume_selects_tradable_second_without_peeking(self):
        earlier = [
            ("bars_1m", "2026-07-31 01:00+00", "MGCQ6", 4000, 1, 20260827),
            ("bars_1m_2m", "2026-07-31 01:00+00", "MGCV6", 4002, 50, 20261027),
            ("bars_1m", "2026-08-03 01:00+00", "MGCQ6", 4001, 1, 20260827),
            ("bars_1m_2m", "2026-08-03 01:00+00", "MGCV6", 4003, 500, 20261027),
            ("bars_1m_2m", "2026-08-03 02:01+00", "MGCV6", 4004, 1, 20261027),
        ]
        prior = pd.DatetimeIndex(["2026-07-31"])
        source = book(earlier)
        preferred, volumes = prior_volume_route(source, SESSION, prior)
        self.assertEqual(preferred, "bars_1m_2m")
        self.assertEqual(volumes["bars_1m_2m"], 50)
        fill = fill_pair(source, SESSION, "GC", "asia_open", 60, 0, CALENDAR,
                         preferred_table=preferred)
        self.assertEqual(fill["entry_contract"], "MGCV6")
        later_changed = earlier.copy()
        later_changed[3] = ("bars_1m_2m", "2026-08-03 01:00+00", "MGCV6", 4003,
                            0, 20261027)
        self.assertEqual(prior_volume_route(book(later_changed), SESSION, prior)[0], preferred)


if __name__ == "__main__":
    unittest.main()
