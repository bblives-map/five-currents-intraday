"""Offline, descriptive per-commodity alpha grid with a shorter GC signal curve."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_daily.curve_activation import mixed_concepts
from five_currents_daily.curve_policy import CURVE_RANK_LIMITS
from five_currents_daily.curve_signals import CURVE_CONCEPTS
from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import (
    CLOCKS, HOLD_MINUTES, DELAYS, SOURCE, candidate_pnl, row_stats,
    weights_for_class,
)
from five_currents_opening_study import lag_contract_fields, rebuild_concepts
from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_group_clock import bundle


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUT = ROOT / "five_currents_gc4_pure_alpha_20261024"
SOURCE_REPLAY = SOURCE / "inputs" / "daily_replay.pkl"
PRICE_SOURCE = ROOT / "five_currents_group_clock_20261024" / "clock_prices.npz"
GC4 = {**CURVE_RANK_LIMITS, "GC": 4}
VARIANTS = tuple(f"{base}__{style}" for base in ("gross", "gross_change")
                 for style in ("magnitude", "sign", "reverse_magnitude", "reverse_sign"))
MIN_COVERAGE = {"CL": .85, "NG": .85, "GC": .75, "SI": .25, "HG": .75}
GROUPS = {"energy": ("CL", "NG"), "precious": ("GC", "SI"),
          "copper": ("HG",), "overall_four": ("CL", "NG", "GC", "HG"),
          "overall_five": tuple(ROOTS)}


def sharpe(y):
    return row_stats(y)["sharpe"]


def without_best_five(y, active):
    copy = np.asarray(y, float).copy()
    winners = np.flatnonzero(active & np.isfinite(copy) & (copy > 0))
    if len(winners):
        copy[winners[np.argsort(copy[winners])[-5:]]] = np.nan
    return sharpe(copy)


def candidate_metrics(gross, cost, planned, sessions, info):
    net = gross - cost
    active = planned & np.isfinite(net)
    years = {f"sr_{year}": sharpe(net[sessions.year == year]) for year in (2024, 2025, 2026)}
    return {**info, **years, "net_sr": sharpe(net),
            "after_top5_sr": without_best_five(net, active),
            "stress_sr": sharpe(gross - 2*cost),
            "planned_days": int(planned.sum()), "matched_active_days": int(active.sum()),
            "priced_trade_coverage": float(active.sum()/planned.sum()) if planned.any() else np.nan,
            "observed_days": int(np.isfinite(net).sum())}


def results_for_group(paths, winners, group, sessions, state):
    roots = GROUPS[group]
    rows = []
    for family in winners.family.unique():
        picked = winners.loc[winners.family.eq(family) & winners.root.isin(roots)]
        if len(picked) != len(roots):
            continue
        legs = [paths[(state, row.root, family)] for row in picked.itertuples()]
        gross = np.mean([leg[0] for leg in legs], axis=0)
        cost = np.mean([leg[1] for leg in legs], axis=0)
        planned = np.any([leg[2] for leg in legs], axis=0)
        rows.append(candidate_metrics(gross, cost, planned, sessions,
                    {"state": state, "group": group, "family": family,
                     "asset_choices": "; ".join(
                         f"{row.root}:{row.signal_class}@{row.clock}/{row.hold_minutes}m"
                         for row in picked.itertuples())}))
    return rows


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Use a new output directory; prior audit is immutable")
    original = pickle.loads(SOURCE_REPLAY.read_bytes())
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(PRICE_SOURCE) as saved:
        prices = {key: saved[key] for key in saved.files}
    baseline = mixed_concepts(original)
    np.testing.assert_allclose(baseline, arrays["raw_settled"], rtol=0, atol=0, equal_nan=True)
    print("[rebuild] dashboard settled votes reconcile", flush=True)
    lagged_wide = lag_contract_fields(original.combined.wide, original.combined.mapping,
                                     original.panel.metadata)
    baseline_lagged = rebuild_concepts(original, original, lagged_wide)
    np.testing.assert_allclose(baseline_lagged, arrays["raw_lagged"], rtol=0, atol=0, equal_nan=True)
    print("[rebuild] dashboard lagged votes reconcile", flush=True)
    gc4_settled = mixed_concepts(original, rank_limits=GC4)
    gc4_lagged = rebuild_concepts(original, original, lagged_wide, rank_limits=GC4)
    technical = [i for i, name in enumerate(names) if name not in CURVE_CONCEPTS]
    np.testing.assert_allclose(gc4_lagged[technical], baseline_lagged[technical],
                               rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(gc4_settled[technical], baseline[technical],
                               rtol=0, atol=0, equal_nan=True)
    # Shared product caps otherwise leak the GC-only sensitivity into other assets.
    for i, root in enumerate(ROOTS):
        if root != "GC":
            gc4_lagged[:, :, i] = baseline_lagged[:, :, i]
            gc4_settled[:, :, i] = baseline[:, :, i]
    print("[rebuild] GC-only rank-four votes ready", flush=True)
    np.savez_compressed(OUT / "recomputed_votes.npz", gc4_lagged=gc4_lagged,
                        gc4_settled=gc4_settled, dates=dates.to_numpy(dtype="datetime64[ns]"))
    rows = []
    books = {"gc4": gc4_lagged, "dashboard": baseline_lagged}
    for state, raw in books.items():
        for ai, root in enumerate(ROOTS):
            if state == "dashboard" and root != "GC":
                continue
            for fi, family in enumerate(names):
                for signal_class in VARIANTS:
                    weights = weights_for_class(raw, raw, dates, sessions, lag=0,
                                                family=fi, signal_class=signal_class)
                    planned = np.abs(weights[:, ai]) > 1e-12
                    pnl = candidate_pnl(weights, prices, (ai,))
                    for cell in range(len(CLOCKS)*len(HOLD_MINUTES)):
                        clock = CLOCKS[cell//len(HOLD_MINUTES)]
                        hold = HOLD_MINUTES[cell%len(HOLD_MINUTES)]
                        result = candidate_metrics(pnl["gross"][cell, 0], pnl["cost"][cell, 0],
                            planned, sessions, {"state": state, "root": root, "family": family,
                            "signal_class": signal_class, "clock": clock, "hold_minutes": hold})
                        result["delay15_sr"] = sharpe(pnl["gross"][cell, 2]-pnl["cost"][cell, 2])
                        rows.append(result)
                print(f"[grid] {state} {root} {family}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "individual_grid.parquet", index=False)
    eligible = frame.loc[(frame.priced_trade_coverage >= frame.root.map(MIN_COVERAGE))
                         & (frame.matched_active_days >= np.where(frame.root.eq("SI"), 80, 100))]
    # Deliberately hindsight: rank each family's independently tradable commodity rule by net SR.
    winners = (eligible.sort_values("net_sr", ascending=False)
               .groupby(["state", "root", "family"], sort=False).head(1).copy())
    winners.to_parquet(OUT / "individual_hindsight_winners.parquet", index=False)
    paths = {}
    root_to_index = {root: i for i, root in enumerate(ROOTS)}
    for row in winners.itertuples(index=False):
        state, ai, fi = row.state, root_to_index[row.root], names.index(row.family)
        raw = books[state]
        weights = weights_for_class(raw, raw, dates, sessions, lag=0,
                                    family=fi, signal_class=row.signal_class)
        planned = np.abs(weights[:, ai]) > 1e-12
        pnl = candidate_pnl(weights, prices, (ai,))
        cell = CLOCKS.index(row.clock)*len(HOLD_MINUTES)+HOLD_MINUTES.index(row.hold_minutes)
        paths[(state, row.root, row.family)] = (
            pnl["gross"][cell, 0], pnl["cost"][cell, 0], planned)
    group_rows = []
    for group in GROUPS:
        group_rows.extend(results_for_group(paths, winners.loc[winners.state.eq("gc4")],
                                            group, sessions, "gc4"))
    pd.DataFrame(group_rows).to_parquet(OUT / "hindsight_group_metrics.parquet", index=False)
    baseline_gc = winners.loc[winners.state.eq("dashboard") & winners.root.eq("GC")]
    comparison = winners.loc[winners.state.eq("gc4") & winners.root.eq("GC")].merge(
        baseline_gc, on="family", suffixes=("_gc4", "_dashboard"))
    comparison.to_parquet(OUT / "gc_curve_comparison.parquet", index=False)
    inputs = [SOURCE_REPLAY, SOURCE / "weight_arrays.npz", SOURCE / "sessions.parquet", PRICE_SOURCE]
    save_json(OUT / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": {str(p): digest(p) for p in inputs},
        "baseline_rank_limits": CURVE_RANK_LIMITS, "test_rank_limits": GC4,
        "gc_only_isolation": "Other commodity vote slices are restored exactly to the pinned dashboard baseline after recomputation, avoiding cross-product normalization spillover.",
        "baseline_rebuild_exact": True, "session_start": str(sessions[0].date()),
        "session_end": str(sessions[-1].date()), "sessions": len(sessions),
        "alpha_families": names, "variants": VARIANTS, "clocks": CLOCKS,
        "holds_minutes": HOLD_MINUTES, "entry_delays_minutes": DELAYS,
        "minimum_priced_coverage": MIN_COVERAGE,
        "selection_warning": "All winner and group rows choose rules on the same history used for evaluation. No OOS claim. Original family parameters themselves are inherited from prior research.",
        "costs": "Micro commission USD 2.16 RT plus original one-tick-per-side slippage; stress doubles all modelled costs.",
        "unpriced_active_trades": "Missing, never imputed to zero or replaced by daily settlement.",
        "market_api_access": False, "database_access": False, "live_collector_changes": False})
    print(f"[complete] {len(frame):,} grid cells, {len(winners)} winners, {len(group_rows)} groups", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
