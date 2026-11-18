"""Causal selection and independent-cost checks for the two-family study."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from five_currents_group_clock_core import CLASSES, CLOCKS, HOLD_MINUTES
from run_five_currents_two_family_independent import (
    CELLS, choose_class, combine, one_family_path, scored,
)


class IndependentFamilyTests(unittest.TestCase):
    def test_coverage_gate_uses_active_trades(self):
        x = np.empty((150, 2, 3))
        wave = (np.arange(150) % 3 - 1) * .0001
        x[:, 0] = .002 + wave[:, None]
        x[:, 1] = .002 + wave[:, None]
        x[:50, 1] = np.nan
        score, eligible, matched, coverage = scored(x, np.ones((150, 2), dtype=bool))
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertEqual(matched[0], 150)
        self.assertLess(coverage[1], .75)

    def test_plateau_needs_multiple_clocks_and_frozen_fit_is_past_only(self):
        n = 150
        sessions = pd.date_range("2025-01-01", periods=n)
        neighbors = np.full((n, 2, len(CLASSES), CELLS, 3), np.nan)
        weights = np.zeros((2, len(CLASSES), n))
        gross = np.full((2, len(CLASSES), CELLS, 4, n), np.nan)
        cost = gross.copy()
        weights[0, 0] = .2
        wave = (np.arange(n) % 3 - 1) * .0001
        cells = (0, 1, len(HOLD_MINUTES))
        for cell in cells:
            neighbors[:, 0, 0, cell] = (.002 + wave)[:, None]
            gross[0, 0, cell] = (.002 + wave)[None, :]
            cost[0, 0, cell] = .0001
        library = {"neighbors": neighbors, "weights": weights,
                   "gross": gross, "cost": cost}
        ci, score, support = choose_class(library, 0, 149, rule="plateau")
        self.assertEqual(ci, 0)
        self.assertEqual(support, 3)
        path, ledger = one_family_path(library, 0, ci, 126, None, None, sessions)
        self.assertEqual(len(ledger), 1)
        self.assertLess(ledger[0]["last_training_session"], ledger[0]["forecast_start"])
        self.assertTrue((path["cell"][:126] == -1).all())
        first = path["cell"][126]
        neighbors[126:, 0, 0, first] = -.2
        repeated, _ = one_family_path(library, 0, ci, 126, None, None, sessions)
        self.assertEqual(first, repeated["cell"][126])

    def test_two_independent_orders_pay_both_costs(self):
        a = {"gross": np.array([.02]), "cost": np.array([.003]),
             "planned": np.array([True])}
        b = {"gross": np.array([.01]), "cost": np.array([.005]),
             "planned": np.array([True])}
        result = combine([a, b])
        self.assertAlmostEqual(result["net"][0], (.02-.003+.01-.005)/2)
        self.assertAlmostEqual(result["cost"][0], .004)


if __name__ == "__main__":
    unittest.main()
