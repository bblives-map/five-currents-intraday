"""Small deterministic checks for the grouped, offline intraday replay."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from five_currents_execution_timing import asset_pnl, MICRO_MULTIPLIERS
from five_currents_group_clock_core import (
    CLOCKS, DELAYS, HOLD_MINUTES, asia_time, candidate_pnl, fit_scores,
    select_classes, weights_for_class,
)
from run_five_currents_group_clock import schedule


class GroupClockTests(unittest.TestCase):
    def test_asia_clock_and_first_tradeable_session(self):
        self.assertEqual(str(asia_time(pd.Timestamp("2026-10-23"), "09:00")),
                         "2026-10-23 01:00:00+00:00")
        dates = pd.date_range("2026-10-01", periods=6)
        sessions = pd.DatetimeIndex(dates[3:])
        raw = np.zeros((1, len(dates), 5))
        raw[0, :, 0] = np.arange(len(dates))
        allocated = 2*raw
        old = weights_for_class(raw, allocated, dates, sessions, family=0,
                                signal_class="gross__magnitude")
        changed = weights_for_class(raw, allocated, dates, sessions, family=0,
                                    signal_class="gross_change__magnitude")
        acceleration = weights_for_class(raw, allocated, dates, sessions, family=0,
                                         signal_class="net_change_of_change__magnitude")
        self.assertEqual(old[0, 0], 1.)
        self.assertEqual(changed[0, 0], 1.)
        self.assertEqual(acceleration[0, 0], 0.)
        inverted = weights_for_class(raw, allocated, dates, sessions, family=0,
                                     signal_class="gross__reverse_sign")
        self.assertEqual(inverted[0, 0], -.2)
        raw[0, 3:, 0] = 1000
        np.testing.assert_array_equal(old[0], weights_for_class(raw, allocated, dates,
            sessions, family=0, signal_class="gross__magnitude")[0])

    def test_active_unknown_is_not_cash_and_micro_commission_is_not_scaled(self):
        shape = (len(CLOCKS), len(HOLD_MINUTES), len(DELAYS), 1, 5)
        move = np.full(shape, np.nan)
        entry = np.full(shape, np.nan)
        weight = np.zeros((1, 5))
        weight[0, 0] = 1.
        cube = {"returns": move, "entry": entry}
        self.assertTrue(np.isnan(candidate_pnl(weight, cube, (0,))["net"][0, 0, 0]))
        move[0, 0, 0, 0, 0] = 0.
        entry[0, 0, 0, 0, 0] = 100.
        gross, commission, slippage = asset_pnl(weight, move[0, 0, 0], entry[0, 0, 0])
        self.assertEqual(gross[0, 0], 0.)
        self.assertAlmostEqual(commission[0, 0], 2.16/(100*MICRO_MULTIPLIERS["CL"]))
        self.assertAlmostEqual(slippage[0, 0], .02/100)
        self.assertTrue(np.isfinite(candidate_pnl(weight, cube, (0,))["net"][0, 0, 0]))

    def test_fit_uses_only_observations_before_forecast(self):
        days = 150
        sessions = pd.date_range("2025-01-01", periods=days)
        # One family, one class, one recipe, three delay neighborhoods.
        training = np.full((days, 1, 1, 1, 3), .001)
        training[::3] = .002
        choices, audit = schedule(training, (0,), sessions, first=126,
                                  lookback=126, refit=5, class_count=1)
        self.assertTrue((audit.train_end_exclusive < audit.start).all())
        self.assertTrue((audit.last_training_session < audit.forecast_start).all())
        original = choices.copy()
        training[126:] = -.5
        rerun, _ = schedule(training, (0,), sessions, first=126,
                            lookback=126, refit=5, class_count=1)
        np.testing.assert_array_equal(original[126:131], rerun[126:131])

    def test_admission_rejects_negative_and_sparse_recipes(self):
        days = 150
        x = np.zeros((days, 3, 3))
        wave = np.arange(days) % 3 - 1
        x[:, 0, :] = .002 + wave[:, None]*.0002
        x[:, 1, :] = -.002 + wave[:, None]*.0002
        x[:, 2, :] = x[:, 0, :]
        x[:100, 2, :] = np.nan
        _, eligible = fit_scores(x)
        np.testing.assert_array_equal(eligible, [True, False, False])
        ranked, _, support = select_classes(x,
                                            families=1, classes=1, recipes=3)
        np.testing.assert_array_equal(ranked, [0])
        self.assertEqual(support[0], 1)


if __name__ == "__main__":
    unittest.main()
