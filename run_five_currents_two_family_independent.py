"""Past-only contract-specific execution test for two existing daily families.

No market or database access. Every family and commodity chooses its own vote
transform, regional clock, and hold; two independent orders retain both costs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import (
    SOURCE, CLOCKS, HOLD_MINUTES, CLASSES, DELAYS, weights_for_class,
    candidate_pnl, fit_scores, row_stats,
)
from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_group_clock import bundle, TIMELINES


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUTPUT = ROOT / "five_currents_two_family_independent_20261024"
PRICE_SOURCE = ROOT / "five_currents_group_clock_20261024" / "clock_prices.npz"
FAMILIES = ("aggregate_volume_carry", "carry_basis_reversal")
FITS = {"frozen": (None, None), "252_weekly": (252, 5), "252_monthly": (252, 21),
        "504_weekly": (504, 5), "504_monthly": (504, 21),
        "expanding_weekly": (None, 5), "expanding_monthly": (None, 21)}
CUTS = (0.3, 0.4, 0.5)
CELLS = len(CLOCKS)*len(HOLD_MINUTES)


def compute_library(arrays, dates, sessions, prices, root_index, family_indices, timeline):
    raw_key, allocated_key, lag = TIMELINES[timeline]
    n = len(sessions)
    weights = np.empty((2, len(CLASSES), n))
    gross = np.empty((2, len(CLASSES), CELLS, len(DELAYS), n))
    cost = np.empty_like(gross)
    neighbors = np.empty((n, 2, len(CLASSES), CELLS, 3))
    for fi, family in enumerate(family_indices):
        for ci, signal_class in enumerate(CLASSES):
            weight = weights_for_class(arrays[raw_key], arrays[allocated_key], dates, sessions,
                                       lag=lag, family=family, signal_class=signal_class)
            weights[fi, ci] = weight[:, root_index]
            pnl = candidate_pnl(weight, prices, (root_index,))
            gross[fi, ci] = pnl["gross"]
            cost[fi, ci] = pnl["cost"]
            neighbors[:, fi, ci] = pnl["stressed"][:, :3].transpose(2, 0, 1)
    return {"weights": weights, "gross": gross, "cost": cost, "neighbors": neighbors}


def scored(training, planned):
    """Add prospective active-fill support to the existing doubled-cost test."""
    score, eligible = fit_scores(training)
    active = np.broadcast_to(planned, training.shape[:2])
    matched = np.isfinite(training[:, :, 0]) & active
    n_active = matched.sum(axis=0)
    n_planned = active.sum(axis=0)
    coverage = np.divide(n_active, n_planned, out=np.zeros(len(n_active), dtype=float),
                         where=n_planned > 0)
    eligible &= (n_active >= 30) & (coverage >= .75)
    return score, eligible, n_active, coverage


def choose_class(library, fi, end, *, rule="plateau"):
    """Select one transformation using its third-best eligible execution recipe."""
    x = library["neighbors"][:end, fi].reshape(end, len(CLASSES)*CELLS, 3)
    active = np.abs(library["weights"][fi, :, :end].T) > 1e-12
    planned = np.repeat(active, CELLS, axis=1)
    score, eligible, matched, coverage = scored(x, planned)
    if rule == "best":
        best = int(np.argmax(np.where(eligible, score, -np.inf))) if eligible.any() else -1
        return (best//CELLS, float(score[best]), 1) if best >= 0 else (-1, np.nan, 0)
    if rule != "plateau":
        raise ValueError(rule)
    scores = score.reshape(len(CLASSES), CELLS)
    allowed = eligible.reshape(len(CLASSES), CELLS)
    ranked = []
    for ci in range(len(CLASSES)):
        cells = np.flatnonzero(allowed[ci])
        if len(cells) < 3 or len(set(cells//len(HOLD_MINUTES))) < 2:
            continue
        third = float(np.sort(scores[ci, cells])[-3])
        ranked.append((third, ci, int(len(cells))))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return (ranked[0][1], ranked[0][0], ranked[0][2]) if ranked else (-1, np.nan, 0)


def choose_cell(library, fi, ci, begin, stop):
    if ci < 0:
        return -1, np.nan, 0
    x = library["neighbors"][begin:stop, fi, ci]
    planned = (np.abs(library["weights"][fi, ci, begin:stop]) > 1e-12)[:, None]
    score, eligible, matched, coverage = scored(x, planned)
    cell = int(np.argmax(np.where(eligible, score, -np.inf))) if eligible.any() else -1
    return cell, float(score[cell]) if cell >= 0 else np.nan, int(eligible.sum())


def one_family_path(library, fi, ci, first, lookback, refit, sessions):
    n = len(sessions)
    cell_path = np.full(n, -1, dtype=np.int16)
    ledger = []
    if ci < 0:
        return {"cell": cell_path, "gross": np.zeros(n), "cost": np.zeros(n),
                "planned": np.zeros(n, dtype=bool)}, ledger
    starts = (first,) if refit is None else range(first, n, refit)
    for start in starts:
        stop = start-1  # End before the forecast, including an embargoed session.
        begin = 0 if lookback is None else max(0, stop-lookback)
        end = n if refit is None else min(n, start+refit)
        cell, score, eligible = choose_cell(library, fi, ci, begin, stop)
        cell_path[start:end] = cell
        ledger.append({"start": start, "end": end, "train_start": begin,
                       "train_end_exclusive": stop, "last_training_session": sessions[stop-1],
                       "forecast_start": sessions[start], "cell": cell,
                       "clock": CLOCKS[cell//len(HOLD_MINUTES)] if cell >= 0 else None,
                       "hold_minutes": HOLD_MINUTES[cell%len(HOLD_MINUTES)] if cell >= 0 else np.nan,
                       "training_fitness": score, "eligible_recipes": eligible})
    output_gross = np.zeros(n)
    output_cost = np.zeros(n)
    planned = np.zeros(n, dtype=bool)
    mask = cell_path >= 0
    if mask.any():
        days = np.flatnonzero(mask)
        output_gross[days] = library["gross"][fi, ci, cell_path[days], 0, days]
        output_cost[days] = library["cost"][fi, ci, cell_path[days], 0, days]
        planned[days] = np.abs(library["weights"][fi, ci, days]) > 1e-12
    return {"cell": cell_path, "gross": output_gross, "cost": output_cost,
            "planned": planned}, ledger


def combine(parts):
    gross = np.mean([part["gross"] for part in parts], axis=0)
    cost = np.mean([part["cost"] for part in parts], axis=0)
    planned = np.logical_or.reduce([part["planned"] for part in parts])
    return {"gross": gross, "cost": cost, "net": gross-cost,
            "planned": planned, "stressed": gross-2*cost}


def stats(part, first, sessions, info):
    mask = np.arange(len(sessions)) >= first
    net = part["net"][mask]
    planned = part["planned"][mask]
    return {**info, **row_stats(net),
            "stressed_sharpe": row_stats(part["stressed"][mask])["sharpe"],
            "planned_days": int(planned.sum()),
            "priced_trade_coverage": float(np.isfinite(net[planned]).mean()) if planned.any() else np.nan}


def run(output=OUTPUT, *, class_rule="plateau"):
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Study exists; use a new output directory")
    arrays, names, sessions, dates = bundle(SOURCE)
    family_indices = [names.index(name) for name in FAMILIES]
    with np.load(PRICE_SOURCE) as saved:
        prices = {key: saved[key] for key in saved.files}
    frozen = {str(path): digest(path) for path in
              (SOURCE / "weight_arrays.npz", SOURCE / "sessions.parquet", PRICE_SOURCE)}
    paths, metrics, ledger_rows, classes = [], [], [], []
    for timeline in TIMELINES:
        for ai, root in enumerate(ROOTS):
            library = compute_library(arrays, dates, sessions, prices, ai, family_indices, timeline)
            for cut in CUTS:
                first = int(len(sessions)*cut)
                admitted = []
                for fi, family in enumerate(FAMILIES):
                    ci, score, support = choose_class(library, fi, first-1, rule=class_rule)
                    admitted.append(ci)
                    classes.append({"root": root, "timeline": timeline, "cut": cut,
                                    "family": family, "class": CLASSES[ci] if ci >= 0 else None,
                                    "class_index": ci, "third_recipe_score": score,
                                    "eligible_recipes_in_class": support,
                                    "first_trading_session": sessions[first]})
                for fit, (lookback, refit) in FITS.items():
                    pieces = []
                    for fi, family in enumerate(FAMILIES):
                        part, audit = one_family_path(library, fi, admitted[fi], first,
                                                      lookback, refit, sessions)
                        info = {"root": root, "timeline": timeline, "cut": cut,
                                "fit": fit, "family": family}
                        for row in audit:
                            ledger_rows.append({**info, "signal_class": CLASSES[admitted[fi]], **row})
                        unit = {**part, "net": part["gross"]-part["cost"],
                                "stressed": part["gross"]-2*part["cost"]}
                        pieces.append(unit)
                        metrics.append(stats(unit, first, sessions, {**info, "unit": "family"}))
                        paths.append(pd.DataFrame({"session": sessions, **info, "unit": "family",
                            "gross": unit["gross"], "cost": unit["cost"], "net": unit["net"],
                            "planned": unit["planned"], "calibrating": np.arange(len(sessions))<first}))
                    both = combine(pieces)
                    info = {"root": root, "timeline": timeline, "cut": cut,
                            "fit": fit, "family": "equal_two", "unit": "asset_bundle"}
                    metrics.append(stats(both, first, sessions, info))
                    paths.append(pd.DataFrame({"session": sessions, **info,
                        "gross": both["gross"], "cost": both["cost"], "net": both["net"],
                        "planned": both["planned"], "calibrating": np.arange(len(sessions))<first}))
            print(f"[causal asset] {root} / {timeline}", flush=True)
            del library
    path_frame = pd.concat(paths, ignore_index=True)
    bundles, bundle_metrics = [], []
    roots = ("CL", "NG", "GC", "HG")
    basket_defs = {"energy_CL_NG": ("CL", "NG"), "metals_GC_HG": ("GC", "HG"),
                   "core_CL_GC_HG": ("CL", "GC", "HG"), "four_ex_SI": roots}
    basket_defs.update({"pair_"+"_".join(pair): pair for pair in combinations(roots, 2)})
    for timeline in TIMELINES:
        for cut in CUTS:
            first = int(len(sessions)*cut)
            for fit in FITS:
                available = {root: path_frame.loc[path_frame.root.eq(root)
                    & path_frame.timeline.eq(timeline) & path_frame.cut.eq(cut)
                    & path_frame.fit.eq(fit) & path_frame.unit.eq("asset_bundle")]
                    .sort_values("session").reset_index(drop=True) for root in ROOTS}
                for label, members in basket_defs.items():
                    parts = [{"gross": available[root].gross.to_numpy(),
                              "cost": available[root].cost.to_numpy(),
                              "planned": available[root].planned.to_numpy()} for root in members]
                    book = combine(parts)
                    info = {"root": label, "timeline": timeline, "cut": cut,
                            "fit": fit, "family": "equal_two", "unit": "basket"}
                    bundle_metrics.append(stats(book, first, sessions, info))
                    bundles.append(pd.DataFrame({"session": sessions, **info,
                        "gross": book["gross"], "cost": book["cost"], "net": book["net"],
                        "planned": book["planned"], "calibrating": np.arange(len(sessions))<first}))
    pd.concat([path_frame, *bundles], ignore_index=True).to_parquet(output / "paths.parquet", index=False)
    pd.DataFrame(metrics+bundle_metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(ledger_rows).to_parquet(output / "refit_ledger.parquet", index=False)
    pd.DataFrame(classes).to_csv(output / "class_selection.csv", index=False)
    assert all(digest(Path(path))==value for path, value in frozen.items())
    save_json(output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": frozen, "source_dates": [str(sessions[0].date()), str(sessions[-1].date())],
        "assets": ROOTS, "families": FAMILIES, "timelines": TIMELINES,
        "clocks": CLOCKS, "hold_minutes": HOLD_MINUTES, "signal_classes": CLASSES,
        "cuts": CUTS, "fits": FITS, "basket_members": basket_defs,
        "class_rule": "Within each contract and family, select the best early class by its third-best eligible recipe across at least two clocks." if class_rule == "plateau" else "Within each contract and family, select the class of the single best early eligible recipe; this is a more permissive multiple-testing control.",
        "recipe_rule": "Past-only median doubled-cost Sharpe at 0/5/15 minute fills, positive in both training halves; active matched count >=30 and coverage >=75%; one completed-session embargo.",
        "cost_rule": "Each family has an independent round trip, costed at micro $2.16 plus source slippage; 50/50 family and equal asset budgets, no gross scale-up.",
        "caveat": "The two families and their original parameters are hindsight-informed. Static and rolling candidates share the early training prefix. SI is kept as an abstention if it lacks sufficient matched active bars. Basket comparisons are descriptive after-the-fact, not separate prospective model selection.",
        "database_access": False, "market_api_access": False, "production_changes": False})
    print(f"[complete] {len(metrics):,} asset/family rows and {len(bundle_metrics):,} basket rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-rule", choices=("plateau", "best"), default="plateau")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    destination = arguments.output or (OUTPUT if arguments.class_rule == "plateau"
        else ROOT / "five_currents_two_family_independent_best_20261024")
    with threadpool_limits(limits=1):
        run(destination.resolve(), class_rule=arguments.class_rule)
