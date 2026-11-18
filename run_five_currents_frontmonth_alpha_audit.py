"""Re-select the archived daily alpha treatments on corrected front-month fills.

This is descriptive hindsight selection. The daily vote arrays are unchanged;
only the minute entry/exit market and fill price cube are replaced.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import (
    CLOCKS, HOLD_MINUTES, SOURCE, candidate_pnl, row_stats, weights_for_class,
)
from run_five_currents_group_clock import bundle


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
FRONT = ROOT / "five_currents_frontmonth_20261025"
OLD_AUDIT = ROOT / "five_currents_gc4_pure_alpha_20261024"
OUT = FRONT / "alpha_audit"
PRODUCTS = ("CL", "NG", "GC", "HG")
VARIANTS = tuple(f"{base}__{style}" for base in ("gross", "gross_change")
                 for style in ("magnitude", "sign", "reverse_magnitude", "reverse_sign"))
MIN_COVERAGE = {"CL": .85, "NG": .85, "GC": .75, "HG": .75}
GROUPS = {"energy": ("CL", "NG"), "gold": ("GC",), "copper": ("HG",),
          "overall_four": PRODUCTS}


def digest(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def sharpe(values):
    return row_stats(values)["sharpe"]


def without_best_five(values, active):
    copy = np.asarray(values, dtype=float).copy()
    winners = np.flatnonzero(active & np.isfinite(copy) & (copy > 0))
    if len(winners):
        copy[winners[np.argsort(copy[winners])[-5:]]] = np.nan
    return sharpe(copy)


def candidate_metrics(gross, cost, planned, sessions, info):
    net = gross - cost
    matched = planned & np.isfinite(net)
    return {**info, "net_sr": sharpe(net),
            "stress_sr": sharpe(gross - 2 * cost),
            "after_top5_sr": without_best_five(net, matched),
            "sr_2024": sharpe(net[sessions.year == 2024]),
            "sr_2025": sharpe(net[sessions.year == 2025]),
            "sr_2026": sharpe(net[sessions.year == 2026]),
            "planned_days": int(planned.sum()),
            "matched_active_days": int(matched.sum()),
            "priced_trade_coverage": float(matched.sum() / planned.sum())
            if planned.any() else np.nan,
            "observed_days": int(np.isfinite(net).sum())}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Completed front-month audit exists; do not overwrite")
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(OLD_AUDIT / "recomputed_votes.npz") as saved:
        raw = saved["gc4_lagged"]
        settled = saved["gc4_settled"]
        np.testing.assert_array_equal(saved["dates"],
                                      dates.to_numpy(dtype="datetime64[ns]"))
    price_path = FRONT / "front_clock_prices_liquidity.npz"
    with np.load(price_path) as saved:
        prices = {name: saved[name] for name in saved.files}
    if prices["returns"].shape[3] != len(sessions):
        raise AssertionError("Minute cube is not aligned to daily signal sessions")
    if not np.isfinite(raw).all():
        raise AssertionError("Daily signal arrays have missing values")
    np.savez_compressed(OUT / "recomputed_votes.npz", gc4_lagged=raw,
                        gc4_settled=settled, dates=dates.to_numpy(dtype="datetime64[ns]"))
    rows = []
    with threadpool_limits(limits=1):
        for root in PRODUCTS:
            ai = ROOTS.index(root)
            for fi, family in enumerate(names):
                for signal_class in VARIANTS:
                    weights = weights_for_class(raw, raw, dates, sessions, lag=0,
                                                family=fi, signal_class=signal_class)
                    planned = np.abs(weights[:, ai]) > 1e-12
                    pnl = candidate_pnl(weights, prices, (ai,))
                    for cell in range(len(CLOCKS) * len(HOLD_MINUTES)):
                        clock = CLOCKS[cell // len(HOLD_MINUTES)]
                        hold = int(HOLD_MINUTES[cell % len(HOLD_MINUTES)])
                        gross, cost = pnl["gross"][cell, 0], pnl["cost"][cell, 0]
                        entry = {"state": "gc4", "root": root, "family": family,
                                 "signal_class": signal_class, "clock": clock,
                                 "hold_minutes": hold}
                        result = candidate_metrics(gross, cost, planned, sessions, entry)
                        result["delay15_sr"] = sharpe(
                            pnl["gross"][cell, 2] - pnl["cost"][cell, 2])
                        rows.append(result)
                print(f"[front grid] {root} {family}", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_parquet(OUT / "individual_grid.parquet", index=False)
    eligible = grid.loc[(grid.priced_trade_coverage >= grid.root.map(MIN_COVERAGE))
                        & grid.matched_active_days.ge(100)]
    winners = (eligible.sort_values("net_sr", ascending=False)
               .groupby(["state", "root", "family"], sort=False).head(1).copy())
    winners.to_parquet(OUT / "individual_hindsight_winners.parquet", index=False)
    print(f"[front winners] {len(winners)} root/family rules", flush=True)
    paths = {}
    with threadpool_limits(limits=1):
        for row in winners.itertuples(index=False):
            ai, fi = ROOTS.index(row.root), names.index(row.family)
            weights = weights_for_class(raw, raw, dates, sessions, lag=0,
                                        family=fi, signal_class=row.signal_class)
            planned = np.abs(weights[:, ai]) > 1e-12
            pnl = candidate_pnl(weights, prices, (ai,))
            cell = CLOCKS.index(row.clock) * len(HOLD_MINUTES) + HOLD_MINUTES.index(row.hold_minutes)
            paths[(row.root, row.family)] = (
                pnl["gross"][cell, 0], pnl["cost"][cell, 0], planned)
    grouped = []
    for group, roots in GROUPS.items():
        for family in names:
            picked = winners.loc[winners.family.eq(family) & winners.root.isin(roots)]
            if len(picked) != len(roots):
                continue
            legs = [paths[(row.root, family)] for row in picked.itertuples()]
            gross = np.mean([leg[0] for leg in legs], axis=0)
            cost = np.mean([leg[1] for leg in legs], axis=0)
            planned = np.any([leg[2] for leg in legs], axis=0)
            grouped.append(candidate_metrics(gross, cost, planned, sessions,
                           {"state": "gc4", "group": group, "family": family,
                            "asset_choices": "; ".join(
                                f"{row.root}:{row.signal_class}@{row.clock}/{row.hold_minutes}m"
                                for row in picked.itertuples())}))
    pd.DataFrame(grouped).to_parquet(OUT / "hindsight_group_metrics.parquet", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": len(sessions), "products": PRODUCTS,
        "daily_signal_sha256": digest(OLD_AUDIT / "recomputed_votes.npz"),
        "front_price_sha256": digest(price_path),
        "selection": "Same-history winner for each asset/family across eight vote treatments, six clocks, eight holds; descriptive hindsight, not OOS",
        "price_rule": "Full one-minute front/second-month micro bar-close fills after positive volume; roll chosen on completed five-session volume, with expiry safety; same listed contract on exit",
        "no_api_calls": True, "no_database_access": True,
    }, indent=2), encoding="utf-8")
    return winners


if __name__ == "__main__":
    run()
