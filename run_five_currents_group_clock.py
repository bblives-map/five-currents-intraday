"""Offline grouped family-by-clock study with past-only class and recipe selection."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import ROOTS, asset_pnl, summarize_matrix
from five_currents_group_clock_core import (
    SOURCE, OUTPUT, GROUPS, CLOCKS, HOLD_MINUTES, DELAYS, CLASSES,
    build_prices, weights_for_class, candidate_pnl, fit_scores, select_classes, row_stats,
)
from prepare_five_currents_execution_timing import digest, save_json


FITS = {"252_weekly": (252, 5), "252_monthly": (252, 21),
        "504_weekly": (504, 5), "504_monthly": (504, 21),
        "expanding_weekly": (None, 5), "expanding_monthly": (None, 21)}
CONTROLS = {"252_monthly_cut40": (252, 21, .4), "252_monthly_cut50": (252, 21, .5)}
TIMELINES = {"fresh_lagged": ("raw_lagged", "allocated_lagged", 0),
             "older_settled": ("raw_settled", "allocated_settled", 1)}


def bundle(source):
    with np.load(source / "weight_arrays.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    names = json.loads((source / "source_manifest.json").read_text())["signal_names"]
    sessions = pd.DatetimeIndex(pd.read_parquet(source / "sessions.parquet").session)
    dates = pd.DatetimeIndex(arrays["dates"].astype("datetime64[ns]"))
    return arrays, names, sessions, dates


def prepare_candidates(arrays, dates, sessions, cube, indices, *, timeline, group, names):
    raw_key, allocated_key, lag = TIMELINES[timeline]
    raw, allocated = arrays[raw_key], arrays[allocated_key]
    days, families, classes, cells = len(sessions), len(names), len(CLASSES), len(CLOCKS)*len(HOLD_MINUTES)
    weights = np.empty((families, classes, days, len(ROOTS)))
    neighbors = np.empty((days, families, classes, cells, 3))
    grid = []
    cut = int(days*.3)
    for fi, family in enumerate(names):
        for ci, signal_class in enumerate(CLASSES):
            weight = weights_for_class(raw, allocated, dates, sessions, lag=lag, family=fi,
                                       signal_class=signal_class)
            weights[fi, ci] = weight
            pnl = candidate_pnl(weight, cube, indices)
            neighbors[:, fi, ci] = pnl["stressed"][:, :3].transpose(2, 0, 1)
            planned = np.abs(weight[:, list(indices)]).sum(axis=1)>1e-12
            for period, mask in (("full", np.ones(days, dtype=bool)),
                                 ("post_30_pct", np.arange(days)>=cut)):
                for cell, (clock, hold) in enumerate(product(CLOCKS, HOLD_MINUTES)):
                    gross = pnl["gross"][cell, 0, mask]
                    cost = pnl["cost"][cell, 0, mask]
                    net = gross-cost
                    active = planned[mask]
                    grid.append({"group": group, "timeline": timeline, "family": family,
                        "signal_class": signal_class, "clock": clock, "hold_minutes": hold,
                        "period": period, **row_stats(net),
                        "gross_sharpe": row_stats(gross)["sharpe"],
                        "stressed_sharpe": row_stats(gross-2*cost)["sharpe"],
                        "planned_days": int(active.sum()),
                        "priced_trade_coverage": float(np.isfinite(net[active]).mean()) if active.any() else np.nan})
        print(f"[full grid] {group} / {timeline} / {family}", flush=True)
    return weights, neighbors, grid


def chosen_classes(neighbors, sessions):
    families, classes, cells = neighbors.shape[1:4]
    result, rows = {}, []
    for fraction in (.3, .4, .5):
        cut = int(len(sessions)*fraction)
        ranked, scores, support = select_classes(neighbors[:cut-1].reshape(cut-1, families*classes*cells, 3),
                                                  families=families, classes=classes, recipes=cells)
        result[fraction] = tuple(int(i) for i in ranked)
        for rank, ci in enumerate(ranked, 1):
            rows.append({"cut_fraction": fraction, "cut_session": sessions[cut], "class_rank": rank,
                         "signal_class": CLASSES[ci], "class_index": int(ci),
                         "second_family_stressed_sharpe": float(scores[ci]),
                         "supporting_families": int(support[ci]),
                         "single_family_fallback": bool(support[ci] < 2)})
    return result, rows


def schedule(neighbors, classes, sessions, *, first, lookback, refit, class_count):
    """At each refit, select one family/clock/hold within each fixed early class."""
    days, families, _, cells, _ = neighbors.shape
    chosen = np.full((days, class_count, 3), -1, dtype=np.int16)
    audit = []
    for start in range(first, days, refit):
        stop = start-1  # one completed-session embargo
        begin = max(0, stop-lookback) if lookback is not None else 0
        end = min(start+refit, days)
        for slot, ci in enumerate(classes[:class_count]):
            x = neighbors[begin:stop, :, ci].reshape(stop-begin, families*cells, 3)
            scores, eligible = fit_scores(x)
            selected = int(np.argmax(np.where(eligible, scores, -np.inf))) if eligible.any() else -1
            if selected >= 0:
                fi, cell = divmod(selected, cells)
                chosen[start:end, slot] = (fi, ci, cell)
            else:
                fi, cell = -1, -1
            audit.append({"start": start, "end": end, "train_start": begin,
                "train_end_exclusive": stop, "last_training_session": sessions[stop-1],
                "forecast_start": sessions[start], "slot": slot, "family_index": fi,
                "class_index": int(ci), "cell": cell, "fitness": float(scores[selected]) if selected >= 0 else np.nan,
                "eligible_recipes": int(eligible.sum())})
    return chosen, pd.DataFrame(audit)


def order_positions(weights, choices, indices):
    days, slots = choices.shape[:2]
    cells = len(CLOCKS)*len(HOLD_MINUTES)
    pos = np.zeros((cells, days, len(ROOTS)))
    for day in range(days):
        for slot in range(slots):
            fi, ci, cell = choices[day, slot]
            if fi < 0:
                continue
            for ai in indices:
                pos[cell, day, ai] += weights[fi, ci, day, ai]/slots
    ceiling = np.abs(pos).sum(axis=(0, 2))
    pos /= np.maximum(ceiling, 1.)[None, :, None]
    return pos


def evaluate_positions(pos, cube, indices, *, offset=0):
    shape = (len(CLOCKS)*len(HOLD_MINUTES), len(pos[0]), len(ROOTS))
    moves = cube["returns"][:, :, offset].reshape(shape)
    entries = cube["entry"][:, :, offset].reshape(shape)
    gross, fee, slip = asset_pnl(pos, moves, entries)
    by_asset = {"gross": gross.sum(axis=0), "cost": (fee+slip).sum(axis=0)}
    by_asset["net"] = by_asset["gross"]-by_asset["cost"]
    selected = np.asarray(indices)
    result = {}
    for name, values in by_asset.items():
        part = values[:, selected]
        result[name] = np.where(np.isfinite(part).all(axis=1), part.sum(axis=1), np.nan)
    result["stressed"] = result["gross"]-2*result["cost"]
    result["by_asset"] = by_asset["net"]
    result["exposure"] = np.abs(pos).sum(axis=(0, 2))
    return result


def replay_rows(result, sessions, first, info):
    rows = []
    masks = {"full_window": np.ones(len(sessions), dtype=bool),
             "post_calibration": np.arange(len(sessions))>=first,
             "common_late_window": np.arange(len(sessions))>=int(len(sessions)*.5)}
    for period, mask in masks.items():
        x = result["net"][mask]
        planned = result["exposure"][mask]>1e-12
        rows.append({**info, "period": period, **row_stats(x),
            "gross_sharpe": row_stats(result["gross"][mask])["sharpe"],
            "stressed_sharpe": row_stats(result["stressed"][mask])["sharpe"],
            "planned_days": int(planned.sum()),
            "priced_trade_coverage": float(np.isfinite(x[planned]).mean()) if planned.any() else np.nan})
    return rows


def run(source, output, *, asia_open="09:00", asia_close="17:00"):
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed study already exists; use a new output directory")
    arrays, names, sessions, dates = bundle(source)
    cube = build_prices(source, output, asia_open=asia_open, asia_close=asia_close)
    frozen = {name: digest(source / name) for name in ("weight_arrays.npz", "price_cube.npz",
        "source_manifest.json", "contract_targets.parquet", "sessions.parquet", "minute_cache_manifest.json")}
    cuts = {"cut30": .3, "cut40": .4, "cut50": .5}
    metrics, grids, class_rows, ledgers, paths, stress, yearly, asset_rows = [], [], [], [], [], [], [], []
    schedules = {}
    for group, indices in GROUPS.items():
        for timeline in TIMELINES:
            weights, neighbors, described = prepare_candidates(arrays, dates, sessions, cube, indices,
                timeline=timeline, group=group, names=names)
            raw_key, allocated_key, base_lag = TIMELINES[timeline]
            lagged_weights = np.empty_like(weights)
            for fi in range(len(names)):
                for ci, signal_class in enumerate(CLASSES):
                    lagged_weights[fi, ci] = weights_for_class(
                        arrays[raw_key], arrays[allocated_key], dates, sessions,
                        lag=base_lag+1, family=fi, signal_class=signal_class)
            grids.extend(described)
            chosen, class_info = chosen_classes(neighbors, sessions)
            class_rows.extend({"group": group, "timeline": timeline, **item} for item in class_info)
            for cut_name, fraction in cuts.items():
                first = int(len(sessions)*fraction)
                fits = FITS if cut_name == "cut30" else {"252_monthly": FITS["252_monthly"]}
                for fit, (lookback, refit) in fits.items():
                    for count in (1, 2):
                        slots = min(count, len(chosen[fraction]))
                        if not slots:
                            continue
                        choices, audit = schedule(neighbors, chosen[fraction], sessions, first=first,
                                                  lookback=lookback, refit=refit, class_count=slots)
                        info = {"group": group, "timeline": timeline, "cut": cut_name,
                                "fit": fit, "class_count": count, "active_classes": slots}
                        key = "__".join(str(x) for x in info.values())
                        info["id"] = key
                        schedules[key] = choices
                        audit = audit.assign(**info)
                        audit["family"] = audit.family_index.map(dict(enumerate(names)))
                        audit["signal_class"] = audit.class_index.map(dict(enumerate(CLASSES)))
                        audit["clock"] = audit.cell.apply(lambda i: CLOCKS[int(i)//len(HOLD_MINUTES)] if i>=0 else None)
                        audit["hold_minutes"] = audit.cell.apply(lambda i: HOLD_MINUTES[int(i)%len(HOLD_MINUTES)] if i>=0 else np.nan)
                        ledgers.append(audit)
                        pos = order_positions(weights, choices, indices)
                        result = evaluate_positions(pos, cube, indices)
                        metrics.extend(replay_rows(result, sessions, first, info))
                        frame = pd.DataFrame({"session": sessions, **info, "net": result["net"],
                            "gross": result["gross"], "cost": result["cost"],
                            "exposure": result["exposure"], "calibrating": np.arange(len(sessions))<first})
                        for ai, root in enumerate(ROOTS):
                            frame[root] = result["by_asset"][:, ai]
                        paths.append(frame)
                        valid = (np.arange(len(sessions))>=first) & np.isfinite(result["net"])
                        ret = result["net"][valid]
                        wealth_before = np.r_[1., np.cumprod(1+ret)[:-1]]
                        for ai in indices:
                            component = result["by_asset"][valid, ai]
                            asset_rows.append({**info, "root": ROOTS[ai], **row_stats(component),
                                "observed_pnl_contribution": float(wealth_before @ component)})
                        if valid.any():
                            np.testing.assert_allclose(sum(r["observed_pnl_contribution"] for r in asset_rows[-len(indices):]),
                                                       np.prod(1+ret)-1, atol=1e-10)
                        for year in sorted(set(sessions.year)):
                            mask = valid & (sessions.year==year)
                            if mask.any():
                                yearly.append({**info, "year": int(year), **row_stats(result["net"][mask])})
                        for delay, weight_lag in ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1)):
                            if weight_lag:
                                stress_pos = order_positions(lagged_weights, choices, indices)
                            else:
                                stress_pos = pos
                            alt = evaluate_positions(stress_pos, cube, indices, offset=delay)
                            mask = np.arange(len(sessions))>=first
                            paired = mask & np.isfinite(result["net"]) & np.isfinite(alt["net"])
                            stress.append({**info, "delay_minutes": DELAYS[delay], "extra_signal_sessions": weight_lag,
                                **row_stats(alt["net"][mask]),
                                "stressed_sharpe": row_stats(alt["stressed"][mask])["sharpe"],
                                "paired_days": int(paired.sum()),
                                "paired_sharpe": row_stats(alt["net"][paired])["sharpe"],
                                "paired_base_sharpe": row_stats(result["net"][paired])["sharpe"]})
                        print(f"[causal replay] {key}", flush=True)
            del weights, lagged_weights, neighbors
    pd.DataFrame(grids).to_parquet(output / "full_family_clock_grid.parquet", index=False)
    pd.DataFrame(class_rows).to_csv(output / "class_selection.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(output / "refit_ledger.parquet", index=False)
    pd.concat(paths, ignore_index=True).to_parquet(output / "paths.parquet", index=False)
    for name, rows in (("metrics.csv", metrics), ("stress.csv", stress), ("asset_attribution.csv", asset_rows),
                       ("yearly.csv", yearly)):
        pd.DataFrame(rows).to_csv(output / name, index=False)
    np.savez_compressed(output / "schedules.npz", **schedules)
    assert all(digest(source / name)==value for name, value in frozen.items())
    save_json(output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": frozen, "groups": GROUPS,
        "sessions": len(sessions), "start": str(sessions[0].date()), "end": str(sessions[-1].date()),
        "asia_open_sgt": asia_open, "asia_close_sgt": asia_close, "clocks": CLOCKS,
        "hold_minutes": HOLD_MINUTES, "fill_delays_minutes": DELAYS, "signal_classes": CLASSES,
        "fit_rules": FITS, "split_controls": cuts,
        "signal_timelines": TIMELINES, "gross_signal_definition": "Raw family vote before daily allocator",
        "net_signal_definition": "The same family's allocated contribution to the daily portfolio, using the pinned historical allocator",
        "signal_age": "T's latest observed family vote is first traded at T+1; older-settled shifts one additional observed session",
        "fit_selection": "Classes chosen on the early prefix only, using the second strongest eligible family. Up to two classes. Within each class, family/clock/hold retuned using past doubled-cost outcomes at 0/5/15 minute fill offsets, with a one-session embargo.",
        "fees": "USD 2.16 per micro round trip plus the source tick/slippage allowances; identical asset/clock/hold orders net first. Other windows retain separate round trips.",
        "risk": "Precomputed unit source weights; sign votes are 1/5 per active asset; planned gross capped at 1 with no automatic scale-up of smaller groups.",
        "limits": "Cached one-minute bars sampled at five-minute endpoints; active unknown exits stay missing, no fill imputation, no quote-depth execution proof. Some historical listed months are inferred. SIL is sparse and ends in September 2026. The existing signal roster was hindsight informed and publication vintages are unavailable.",
        "new_market_api_requests": False, "database_access": False, "production_changes": False,
        "candidate_rules_per_group_timeline": len(names)*len(CLASSES)*len(CLOCKS)*len(HOLD_MINUTES),
        "full_rule_rows": len(grids), "causal_portfolios": len(paths)})
    print(f"[complete] {len(grids):,} descriptive rows; {len(paths)} causal paths", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--asia-open", default="09:00")
    parser.add_argument("--asia-close", default="17:00")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source.resolve(), args.output.resolve(), asia_open=args.asia_open, asia_close=args.asia_close)
