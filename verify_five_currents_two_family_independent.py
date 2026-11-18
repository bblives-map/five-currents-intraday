"""Audit independent family and basket arithmetic against exported paths."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_two_family_independent import (
    OUTPUT, ROOT, FAMILIES, ROOTS, stats,
)
from run_five_currents_group_clock import bundle


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        assert digest(Path(path)) == expected, path
    paths = pd.read_parquet(output / "paths.parquet")
    ledger = pd.read_parquet(output / "refit_ledger.parquet")
    classes = pd.read_csv(output / "class_selection.csv")
    sessions = bundle(ROOT / "five_currents_execution_timing_20261023")[2]
    assert (ledger.train_end_exclusive < ledger.start).all()
    assert (ledger.last_training_session < ledger.forecast_start).all()
    first = {cut: int(len(sessions)*cut) for cut in manifest["cuts"]}
    compared, maximum = 0, 0.
    keys = ("timeline", "cut", "fit")
    for combo, group in paths.groupby(list(keys), sort=False):
        timeline, cut, fit = combo
        assert len(group) == len(sessions)*(len(ROOTS)*3+len(manifest["basket_members"]))
        by_root = {}
        for root in ROOTS:
            family = [group.loc[group.root.eq(root) & group.unit.eq("family")
                      & group.family.eq(name)].sort_values("session") for name in FAMILIES]
            asset_bundle = group.loc[group.root.eq(root) & group.unit.eq("asset_bundle")].sort_values("session")
            assert all(len(item) == len(sessions) for item in (*family, asset_bundle))
            for field in ("gross", "cost", "net"):
                expected = np.mean([item[field].to_numpy() for item in family], axis=0)
                observed = asset_bundle[field].to_numpy()
                np.testing.assert_allclose(observed, expected, atol=1e-12,
                                           rtol=0, equal_nan=True)
                diff = np.abs(observed-expected)
                if np.isfinite(diff).any():
                    maximum = max(maximum, float(np.nanmax(diff)))
            np.testing.assert_array_equal(asset_bundle.planned.to_numpy(),
                np.logical_or(family[0].planned.to_numpy(), family[1].planned.to_numpy()))
            by_root[root] = asset_bundle
            compared += 1
        for label, members in manifest["basket_members"].items():
            basket = group.loc[group.root.eq(label) & group.unit.eq("basket")].sort_values("session")
            assert len(basket) == len(sessions)
            for field in ("gross", "cost", "net"):
                expected = np.mean([by_root[root][field].to_numpy() for root in members], axis=0)
                np.testing.assert_allclose(basket[field].to_numpy(), expected,
                                           atol=1e-12, rtol=0, equal_nan=True)
            compared += 1
    for row in classes.itertuples():
        chosen = ledger.loc[ledger.root.eq(row.root) & ledger.timeline.eq(row.timeline)
                            & ledger.cut.eq(row.cut) & ledger.family.eq(row.family)]
        if pd.isna(row.class_index) or row.class_index < 0:
            assert chosen.empty
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "source_hashes_match": True, "past_only_refits": True,
              "reconciled_asset_and_basket_paths": compared,
              "max_absolute_family_to_asset_difference": maximum,
              "market_api_access": False, "database_access": False}
    save_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    for directory in (OUTPUT, ROOT / "five_currents_two_family_independent_best_20261024"):
        print(directory.name, verify(directory))
