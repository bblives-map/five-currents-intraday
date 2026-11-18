"""No-fit intraday execution footprint of two frozen raw family alpha votes."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import (
    SOURCE, CLOCKS, HOLD_MINUTES, DELAYS, weights_for_class,
    candidate_pnl, row_stats,
)
from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_group_clock import bundle, TIMELINES


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUTPUT = ROOT / "five_currents_two_family_pure_20261024"
PRICE_SOURCE = ROOT / "five_currents_group_clock_20261024" / "clock_prices.npz"
FAMILIES = ("aggregate_volume_carry", "carry_basis_reversal")
STYLES = ("magnitude", "sign", "reverse_magnitude", "reverse_sign")
BASES = ("gross", "gross_change")
VARIANTS = tuple(f"{base}__{style}" for base in BASES for style in STYLES)
TIMELINE_NAMES = tuple(TIMELINES)
CELLS = len(CLOCKS)*len(HOLD_MINUTES)
GROUPS = {"CL_NG": (0, 1), "GC_SI": (2, 3), "GC_HG": (2, 4),
          "CL_HG": (0, 4), "CL_GC": (0, 2), "CL_GC_HG": (0, 2, 4),
          "CL_NG_GC_HG": (0, 1, 2, 4)}


def periods(sessions):
    return {"full": np.ones(len(sessions), dtype=bool),
            "2024": sessions.year == 2024,
            "2025": sessions.year == 2025,
            "2026": sessions.year == 2026,
            "post_30_pct": np.arange(len(sessions)) >= int(len(sessions)*.3)}


def metrics(gross, cost, planned, mask, info):
    g, c, active = gross[mask], cost[mask], planned[mask]
    net = g-c
    result = {**info, **row_stats(net),
              "gross_sharpe": row_stats(g)["sharpe"],
              "stressed_sharpe": row_stats(g-2*c)["sharpe"],
              "planned_days": int(active.sum()),
              "matched_active_days": int((active & np.isfinite(net)).sum()),
              "priced_trade_coverage": float(np.isfinite(net[active]).mean()) if active.any() else np.nan}
    return result


def run(output=OUTPUT):
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Study already exists; use a new output directory")
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(PRICE_SOURCE) as saved:
        prices = {key: saved[key] for key in saved.files}
    frozen = {str(path): digest(path) for path in
              (SOURCE / "weight_arrays.npz", SOURCE / "sessions.parquet", PRICE_SOURCE)}
    shape = (len(TIMELINE_NAMES), len(FAMILIES), len(ROOTS), CELLS, len(DELAYS), len(sessions))
    direct_gross = np.full(shape, np.nan)
    direct_cost = np.full(shape, np.nan)
    direct_planned = np.zeros((len(TIMELINE_NAMES), len(FAMILIES), len(ROOTS), len(sessions)), dtype=bool)
    rows = []
    masks = periods(sessions)
    for ti, timeline in enumerate(TIMELINE_NAMES):
        raw_key, allocated_key, lag = TIMELINES[timeline]
        raw, allocated = arrays[raw_key], arrays[allocated_key]
        for fi, family in enumerate(FAMILIES):
            family_index = names.index(family)
            for ai, root in enumerate(ROOTS):
                for signal_class in VARIANTS:
                    weights = weights_for_class(raw, allocated, dates, sessions,
                        lag=lag, family=family_index, signal_class=signal_class)
                    planned = np.abs(weights[:, ai]) > 1e-12
                    pnl = candidate_pnl(weights, prices, (ai,))
                    if signal_class == "gross__magnitude":
                        direct_gross[ti, fi, ai] = pnl["gross"]
                        direct_cost[ti, fi, ai] = pnl["cost"]
                        direct_planned[ti, fi, ai] = planned
                    for cell in range(CELLS):
                        clock, hold = CLOCKS[cell//len(HOLD_MINUTES)], HOLD_MINUTES[cell%len(HOLD_MINUTES)]
                        for period, mask in masks.items():
                            rows.append(metrics(pnl["gross"][cell, 0], pnl["cost"][cell, 0], planned, mask,
                                {"root": root, "family": family, "timeline": timeline,
                                 "signal_class": signal_class, "clock": clock,
                                 "hold_minutes": hold, "period": period}))
                print(f"[frozen family] {root} / {timeline} / {family}", flush=True)
    np.savez_compressed(output / "direct_paths.npz", gross=direct_gross,
                        cost=direct_cost, planned=direct_planned,
                        sessions=sessions.to_numpy(dtype="datetime64[ns]"))
    pd.DataFrame(rows).to_parquet(output / "individual_rule_metrics.parquet", index=False)
    group_rows = []
    all_groups = {**{root: (ai,) for ai, root in enumerate(ROOTS)}, **GROUPS}
    group_shape = (len(TIMELINE_NAMES), len(all_groups), CELLS, len(DELAYS), len(sessions))
    group_gross = np.full(group_shape, np.nan)
    group_cost = np.full(group_shape, np.nan)
    group_planned = np.zeros((len(TIMELINE_NAMES), len(all_groups), len(sessions)), dtype=bool)
    group_names = tuple(all_groups)
    for ti, timeline in enumerate(TIMELINE_NAMES):
        for gi, (group, indices) in enumerate(all_groups.items()):
            gross = direct_gross[ti][:, list(indices)].mean(axis=(0, 1))
            cost = direct_cost[ti][:, list(indices)].mean(axis=(0, 1))
            planned = direct_planned[ti][:, list(indices)].any(axis=(0, 1))
            # Preserve each family's round trip and equal fixed product budgets.
            assert gross.shape == (CELLS, len(DELAYS), len(sessions))
            group_gross[ti, gi] = gross
            group_cost[ti, gi] = cost
            group_planned[ti, gi] = planned
            for cell in range(CELLS):
                clock, hold = CLOCKS[cell//len(HOLD_MINUTES)], HOLD_MINUTES[cell%len(HOLD_MINUTES)]
                for period, mask in masks.items():
                    group_rows.append(metrics(gross[cell, 0], cost[cell, 0], planned, mask,
                        {"group": group, "timeline": timeline, "clock": clock,
                         "hold_minutes": hold, "period": period}))
    pd.DataFrame(group_rows).to_parquet(output / "fixed_group_metrics.parquet", index=False)
    np.savez_compressed(output / "fixed_group_paths.npz", gross=group_gross,
                        cost=group_cost, planned=group_planned, groups=group_names,
                        sessions=sessions.to_numpy(dtype="datetime64[ns]"))
    assert all(digest(Path(path)) == value for path, value in frozen.items())
    save_json(output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": frozen, "sessions": len(sessions),
        "start": str(sessions[0].date()), "end": str(sessions[-1].date()),
        "families": FAMILIES, "variants": VARIANTS, "timelines": TIMELINES,
        "clocks": CLOCKS, "hold_minutes": HOLD_MINUTES, "fill_delays": DELAYS,
        "groups": {name: [ROOTS[i] for i in indices] for name, indices in all_groups.items()},
        "rule": "Frozen raw family vote, first tradable next session; no new family admission, class selection, timing selection, portfolio refit, or trained weights in this study.",
        "interpretation": "Full grids are descriptive. No ex-post maximum is an out-of-sample recommendation. Fixed group paths use the same stated clock/hold for both independently traded families and all group members, with equal budgets and both round trips costed.",
        "costs": "Micro futures USD 2.16 per round trip plus the existing source slippage allowances; active unknown fills stay missing.",
        "upstream_limit": "The original family definitions and parameters were informed by earlier research; historical publication vintages are not available. SIL minute-bar coverage is sparse.",
        "database_access": False, "market_api_access": False, "production_changes": False})
    print(f"[complete] {len(rows):,} individual rule-period rows; {len(group_rows):,} fixed group-period rows", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
