"""Independent path and provenance checks for the grouped offline study."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_group_clock_core import OUTPUT, GROUPS, CLASSES, weights_for_class
from five_currents_execution_timing import ROOTS
from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_group_clock import (
    TIMELINES, bundle, evaluate_positions, order_positions,
)


def verify(output=OUTPUT):
    manifest = json.loads((output / "manifest.json").read_text())
    source = output.parent / "five_currents_execution_timing_20261023"
    assert str(source.resolve()) == manifest["source"]
    for filename, expected in manifest["source_sha256"].items():
        assert digest(source / filename) == expected, filename
    arrays, names, sessions, dates = bundle(source)
    paths = pd.read_parquet(output / "paths.parquet")
    ledger = pd.read_parquet(output / "refit_ledger.parquet")
    with np.load(output / "clock_prices.npz") as saved:
        cube = {key: saved[key] for key in saved.files}
    with np.load(source / "price_cube.npz") as old:
        for field in ("returns", "entry"):
            expected = old[field][[2, 3, 4, 6], :8][:, :, [2, 3, 4, 5]]
            np.testing.assert_array_equal(cube[field][2:], expected)
    asia = pd.read_parquet(output / "asia_fill_ledger.parquet")
    matched = asia.loc[asia.status.eq("matched")]
    assert len(matched) == int(np.isfinite(cube["returns"][:2]).sum())
    assert (matched.entry_ts >= matched.planned_entry).all()
    assert (matched.entry_ts <= matched.planned_entry+pd.Timedelta(minutes=10)).all()
    assert (matched.exit_ts > matched.entry_ts).all()
    assert (matched.exit_ts >= matched.planned_exit).all()
    assert (matched.exit_ts <= matched.planned_exit+pd.Timedelta(minutes=10)).all()
    max_error, checked = 0., 0
    cache = {}
    with np.load(output / "schedules.npz") as schedules:
        for group, indices in GROUPS.items():
            for timeline in TIMELINES:
                raw_key, allocated_key, lag = TIMELINES[timeline]
                weights = cache.get(timeline)
                if weights is None:
                    weights = np.empty((len(names), len(CLASSES), len(sessions), 5))
                    for fi in range(len(names)):
                        for ci, signal_class in enumerate(CLASSES):
                            weights[fi, ci] = weights_for_class(
                                arrays[raw_key], arrays[allocated_key], dates, sessions,
                                lag=lag, family=fi, signal_class=signal_class)
                    cache[timeline] = weights
                for count in (1, 2):
                    key = "__".join((group, timeline, "cut30", "252_monthly", str(count), str(count)))
                    if key not in schedules.files:
                        continue
                    choices = schedules[key]
                    rows = ledger.loc[ledger.id.eq(key)].sort_values(["start", "slot"])
                    for row in rows.itertuples():
                        assert row.train_end_exclusive < row.start
                        assert row.last_training_session < row.forecast_start
                        block = choices[row.start:row.end, row.slot]
                        assert np.all(block[:, 0] == row.family_index)
                        assert np.all(block[:, 1] == row.class_index) if row.family_index >= 0 else True
                        assert np.all(block[:, 2] == row.cell)
                    pos = order_positions(weights, choices, indices)
                    replay = evaluate_positions(pos, cube, indices)
                    stored = paths.loc[paths.id.eq(key)].sort_values("session")
                    assert len(stored) == len(sessions)
                    for field in ("net", "gross", "cost", "exposure"):
                        np.testing.assert_allclose(stored[field].to_numpy(), replay[field],
                                                   atol=1e-12, rtol=0, equal_nan=True)
                        difference = np.abs(stored[field].to_numpy()-replay[field])
                        if np.isfinite(difference).any():
                            max_error = max(max_error, float(np.nanmax(difference)))
                    for ai in indices:
                        np.testing.assert_allclose(stored[ROOTS[ai]].to_numpy(),
                            replay["by_asset"][:, ai], atol=1e-12, rtol=0, equal_nan=True)
                    observed = np.isfinite(replay["net"])
                    components = replay["by_asset"][:, list(indices)]
                    np.testing.assert_allclose(components[observed].sum(axis=1),
                                               replay["net"][observed], atol=1e-12, rtol=0)
                    checked += 1
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "primary_paths_rebuilt": checked, "maximum_absolute_difference": max_error,
              "all_source_hashes_match": True, "all_refits_past_only": True,
              "asset_returns_add_to_group": True, "reused_clocks_match_pinned_cube": True,
              "asia_matched_fills": len(matched), "asia_fills_forward_only": True,
              "live_data_access": False}
    save_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(verify())
