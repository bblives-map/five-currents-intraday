"""Reproduce saved schedules and align every attribution statistic to its portfolio dates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import ROOTS
from five_currents_family_timing import mapped_weights, simulate
from prepare_five_currents_execution_timing import save_json
from run_five_currents_execution_timing_study import signals, metric_row
from run_five_currents_family_timing import OUTPUT, load_grid


def reconcile(output):
    import json
    manifest = json.loads((output / "manifest.json").read_text())
    source = signals(Path(manifest["source"]))
    paths = pd.read_parquet(output / "paths.parquet")
    rows, checks = [], 0
    with np.load(output / "schedules.npz") as saved:
        for grid in ("regional", "fixed_hold"):
            cube, _ = load_grid(Path(manifest["source"]), grid)
            for timeline in ("fresh_lagged", "older_settled"):
                weights = mapped_weights(source, timeline)
                batch = paths.loc[paths.grid.eq(grid) & paths.timeline.eq(timeline)]
                for sid, frame in batch.groupby("id"):
                    frame = frame.sort_values("session")
                    choices, alpha = saved[sid+"__choices"], saved[sid+"__allocations"]
                    result = simulate(weights, cube, choices, alpha)
                    np.testing.assert_allclose(result["net"], frame.net, atol=1e-12, equal_nan=True)
                    mask = ~frame.calibrating.to_numpy() & np.isfinite(result["net"])
                    returns = result["net"][mask]
                    wealth = np.r_[1., np.cumprod(1+returns)[:-1]]
                    info = {key: frame.iloc[0][key] for key in ("id", "grid", "timeline", "book", "scope", "fit", "method")}
                    totals = {}
                    for kind, matrix, labels in (("asset", result["by_asset"]["net"], ROOTS),
                                                 ("family", result["by_family"], manifest["families"])):
                        selected = matrix[mask]
                        np.testing.assert_allclose(selected.sum(axis=1), returns, atol=1e-12)
                        contributions = wealth @ selected
                        np.testing.assert_allclose(contributions.sum(), np.prod(1+returns)-1, atol=1e-12)
                        totals[kind] = float(contributions.sum())
                        for i, name in enumerate(labels):
                            extra = {"allocation_sessions": int(np.any(alpha[mask, :, i] > 0, axis=1).sum())} if kind == "family" else {}
                            rows.append(metric_row(selected[:, i], **info, kind=kind, component=name,
                                observed_pnl_contribution=float(contributions[i]), **extra))
                    checks += 1
                print("[common-date attribution]", grid, timeline, flush=True)
    pd.DataFrame(rows).to_csv(output / "attribution_common_dates.csv", index=False)
    save_json(output / "common_attribution_audit.json", {"reproduced_portfolios": checks,
        "rule": "All components use exactly their portfolio's complete post-calibration dates. Observed wealth-weighted dollar/NAV contributions sum to the observed portfolio total return.",
        "no_retraining": True, "new_market_api_requests": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    with threadpool_limits(limits=1):
        reconcile(parser.parse_args().output.resolve())
