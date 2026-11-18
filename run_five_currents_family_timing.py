"""Repeat cached intraday research with periodic cross-family admission and timing fits.

No database connections, API requests, scheduler edits, or production writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import CLOCKS, HOLDS, OFFSETS, PRIMARY_VARIANTS, ROOTS, summarize_matrix
from five_currents_family_timing import (
    METHODS, FITS, mapped_weights, training_pools, fit_schedule, simulate, concentration,
)
from run_five_currents_regional_blocks import WINDOWS
from run_five_currents_execution_timing_study import signals, metric_row
from prepare_five_currents_execution_timing import DEFAULT_OUT, digest, save_json


OUTPUT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_family_walkforward_20261024"
BOOKS = {"four": (0, 1, 2, 4), "five": (0, 1, 2, 3, 4)}
INPUTS = ("weight_arrays.npz", "source_manifest.json", "sessions.parquet", "price_cube.npz",
          "regional_block_prices.npz", "contract_targets.parquet")


def load_grid(source, grid):
    if grid == "regional":
        with np.load(source / "regional_block_prices.npz") as saved:
            cube = {name: saved[name] for name in saved.files}
        cells = [{"window": window, "clock": first, "exit": last, "hold": 0}
                 for window, (first, last) in WINDOWS.items()]
    else:
        all_cells = list(product(CLOCKS, HOLDS))
        indices = [i for i, (_, h) in enumerate(all_cells) if h > 0]
        with np.load(source / "price_cube.npz") as saved:
            cube = {name: saved[name].reshape(63, 6, -1, 5)[indices][:, [OFFSETS.index(i) for i in (0, 5, 15, 30)]]
                    for name in ("entry", "returns")}
        cells = [{"window": f"{clock}_{hold}m", "clock": clock, "exit": "fixed_hold", "hold": hold}
                 for clock, hold in (all_cells[i] for i in indices)]
    meta = pd.DataFrame([{**row, "variant": variant, "candidate": i*len(cells)+j}
                         for i, variant in enumerate(PRIMARY_VARIANTS) for j, row in enumerate(cells)])
    return cube, meta


def combine_schedule(fitted, book, scope, fit, method, days, families):
    choices = np.full((days, 5, families), -1, dtype=np.int16)
    alpha = np.zeros((days, 5, families))
    for ai in BOOKS[book]:
        key = book if scope == "shared" else ROOTS[ai]
        c, a, _, start = fitted[key, fit]
        choices[:, ai], alpha[:, ai] = c, a[method]
    return choices, alpha, start


def score_row(result, mask, **metadata):
    return metric_row(result["net"][mask], **metadata,
        stressed_sharpe=summarize_matrix(result["stressed"][mask])["sharpe"][0],
        independent_fee_sharpe=summarize_matrix(result["independent_net"][mask])["sharpe"][0],
        gross_sharpe=summarize_matrix(result["gross"][mask])["sharpe"][0],
        mean_gross_exposure=float(result["exposure"][mask].mean()),
        total_cost=float(np.nansum(result["cost"][mask])))


def run(source_path, output):
    if (output / "manifest.json").exists() or (output / "paths.parquet").exists():
        raise FileExistsError("Use a new output directory; prior research will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    before = {name: digest(source_path / name) for name in INPUTS}
    source = signals(source_path)
    days, families = len(source.sessions), len(source.manifest["signal_names"])
    names = source.manifest["signal_names"]
    metrics, path_frames, audit_frames, asset_rows, family_rows = [], [], [], [], []
    stress_rows, concentration_rows, yearly_rows, comparisons, scalar_rows = [], [], [], [], []
    schedules, series_catalog = {}, []
    baseline_checks, attribution_checks, prefix_checks = 0, 0, 0
    configurations = {**FITS, **{f"frozen_{int(f*100)}_{100-int(f*100)}": (None, None) for f in (.3, .4, .5)}}
    pools = {**{root: (i,) for i, root in enumerate(ROOTS)}, **BOOKS}
    rng = np.random.default_rng(8741)
    for grid in ("regional", "fixed_hold"):
        cube, metadata = load_grid(source_path, grid)
        metadata.to_csv(output / f"{grid}_recipes.csv", index=False)
        old = pd.read_parquet(source_path / ("regional_block_causal_paths.parquet" if grid == "regional" else "walk_forward_paths.parquet"))
        for timeline in ("fresh_lagged", "older_settled"):
            weights = mapped_weights(source, timeline)
            print(f"[candidates] {grid} / {timeline}: {families} families x {len(metadata)} recipes", flush=True)
            neighbors = training_pools(weights, cube, pools)
            fitted = {}
            for pool, training in neighbors.items():
                for fit, (lookback, refit) in configurations.items():
                    fraction = int(fit.split("_")[1])/100 if fit.startswith("frozen_") else None
                    value = fit_schedule(training, lookback=lookback, refit=refit or 21, frozen_fraction=fraction)
                    c, allocations, audit, first = value
                    audit["grid"], audit["timeline"], audit["pool"], audit["fit"] = grid, timeline, pool, fit
                    audit["family"] = audit.family_index.map(dict(enumerate(names)))
                    audit["forecast_start"] = source.sessions[audit.start.to_numpy()]
                    audit["last_training_session"] = source.sessions[audit.train_end_exclusive.to_numpy()-1]
                    audit = audit.merge(metadata, on="candidate", how="left")
                    audit_frames.append(audit)
                    fitted[pool, fit] = (c, allocations, audit, first)
                    if fit == "252_monthly":
                        # Future-return mutation cannot affect any earlier decision.
                        cut = 380
                        altered = training.copy()
                        altered[cut:] *= -100
                        cc, aa, _, _ = fit_schedule(altered, lookback=252, refit=21)
                        np.testing.assert_array_equal(c[:cut], cc[:cut])
                        for method in METHODS:
                            np.testing.assert_allclose(allocations[method][:cut], aa[method][:cut])
                        prefix_checks += 1
                print(f"[retrained] {grid} / {timeline} / {pool}: weekly, monthly and frozen controls", flush=True)
            del neighbors
            lagged_weights = {lag: mapped_weights(source, timeline, extra_lag=lag) for lag in (1, 2)}
            for book, scope, fit, method in product(BOOKS, ("shared", "assetwise"), configurations, METHODS):
                c, a, first = combine_schedule(fitted, book, scope, fit, method, days, families)
                sid = f"{grid}__{timeline}__{book}__{scope}__{fit}__{method}"
                info = dict(id=sid, grid=grid, timeline=timeline, book=book, scope=scope, fit=fit, method=method)
                result = simulate(weights, cube, c, a)
                schedules[sid+"__choices"], schedules[sid+"__allocations"] = c, a
                periods = {"full_window": np.ones(days, dtype=bool),
                           "post_calibration": np.arange(days) >= first,
                           "common_late_window": np.arange(days) >= int(days*.5)}
                for period, mask in periods.items():
                    metrics.append(score_row(result, mask, period=period, **info))
                active_families = np.any(a > 0, axis=1).sum(axis=1)
                frame = pd.DataFrame({"session": source.sessions, **info,
                    **{key: result[key] for key in ("net", "gross", "cost", "independent_net", "stressed", "exposure")},
                    "calibrating": np.arange(days) < first, "selected_families": active_families})
                for ai, root in enumerate(ROOTS):
                    frame[root] = result["by_asset"]["net"][:, ai]
                    if ai in BOOKS[book]:
                        asset_rows.append(metric_row(result["by_asset"]["net"][first:, ai], root=root, **info))
                path_frames.append(frame)
                np.testing.assert_allclose(result["by_family"].sum(axis=1), result["net"], atol=1e-12, equal_nan=True)
                attribution_checks += 1
                for f, name in enumerate(names):
                    family_rows.append(metric_row(result["by_family"][first:, f], family=name,
                        allocation_sessions=int(np.any(a[first:, :, f] > 0, axis=1).sum()), **info))
                for year in sorted(set(source.sessions.year)):
                    mask = (source.sessions.year == year) & (np.arange(days) >= first)
                    if mask.any():
                        yearly_rows.append(score_row(result, mask, year=int(year), **info))
                for row in concentration(result["net"][first:]):
                    concentration_rows.append({**info, **row})
                if scope == "shared" and fit == "252_monthly" and method == "fixed_ten":
                    prefix = "held__" if timeline == "older_settled" else ""
                    selected = old.loc[old.book.eq(book) & old.fit.eq("rolling252_monthly")
                                       & old.signal.isin([prefix+name for name in names])]
                    expected = selected.pivot(index="session", columns="signal", values="net").reindex(source.sessions)
                    expected = expected.sum(axis=1, min_count=families).to_numpy()/families
                    np.testing.assert_allclose(result["independent_net"], expected, atol=1e-12, equal_nan=True)
                    baseline_checks += 1
                if not fit.startswith("frozen"):
                    for delay, lag in ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2)):
                        stressed = result if delay == lag == 0 else simulate(
                            lagged_weights.get(lag, weights), cube, c, a, delay=delay)
                        mask = periods["post_calibration"]
                        common = mask & np.isfinite(result["net"]) & np.isfinite(stressed["net"])
                        stress_rows.append(score_row(stressed, mask, **info, delay_minutes=(0, 5, 15, 30)[delay], lag=lag,
                            paired_days=int(common.sum()),
                            paired_sharpe=summarize_matrix(stressed["net"][common])["sharpe"][0],
                            paired_base_sharpe=summarize_matrix(result["net"][common])["sharpe"][0]))
                    for sign in (-1, 1):
                        control = simulate(weights, cube, c, a, direction=sign)
                        comparisons.append(score_row(control, periods["post_calibration"], **info,
                            control="always_long" if sign == 1 else "always_short"))
                if fit == "252_monthly" and method == "greedy_three":
                    # Independent scalar P&L/fees, sampled from actual net orders.
                    positions = result["positions"]
                    locations = np.argwhere(np.abs(positions) > 1e-12)
                    for index in rng.choice(len(locations), min(12, len(locations)), replace=False):
                        di, cell, ai = locations[index]
                        entry, move = cube["entry"][cell, 0, di, ai], cube["returns"][cell, 0, di, ai]
                        if not np.isfinite(entry+move):
                            continue
                        from five_currents_front_intraday import MICRO_MULTIPLIERS
                        from five_currents_execution_timing import SLIPPAGE_PRICE_PER_SIDE
                        multiplier = MICRO_MULTIPLIERS[ROOTS[ai]]
                        qty = positions[di, cell, ai]/(entry*multiplier)
                        dollar_gross = qty*multiplier*(entry*(1+move)-entry)
                        debit = abs(qty)*(2.16+2*SLIPPAGE_PRICE_PER_SIDE[ROOTS[ai]]*multiplier)
                        gross, fee, slip = __import__("five_currents_execution_timing").asset_pnl(
                            positions[di:di+1, cell:cell+1],
                            cube["returns"][cell:cell+1, 0, di:di+1].transpose(1, 0, 2),
                            cube["entry"][cell:cell+1, 0, di:di+1].transpose(1, 0, 2))
                        error = abs((dollar_gross-debit)-(gross-fee-slip)[0, 0, ai])
                        if error > 1e-12:
                            raise AssertionError(error)
                        scalar_rows.append({"id": sid, "session": source.sessions[di], "root": ROOTS[ai],
                                            "cell": int(cell), "absolute_error": error})
                series_catalog.append({**info, "first_test": str(source.sessions[first].date())})
            print(f"[replay complete] {grid} / {timeline}: {len(path_frames)} cumulative portfolio paths", flush=True)
    frames = {"metrics.csv": metrics, "asset_metrics.csv": asset_rows, "family_attribution.csv": family_rows,
              "stress.csv": stress_rows, "concentration.csv": concentration_rows, "yearly.csv": yearly_rows,
              "direction_controls.csv": comparisons, "independent_scalar_audit.csv": scalar_rows}
    for name, records in frames.items():
        pd.DataFrame(records).to_csv(output / name, index=False)
    pd.concat(path_frames, ignore_index=True).to_parquet(output / "paths.parquet", index=False)
    ledger = pd.concat(audit_frames, ignore_index=True)
    assert (ledger.last_training_session < ledger.forecast_start).all()
    ledger.to_parquet(output / "refit_ledger.parquet", index=False)
    np.savez_compressed(output / "schedules.npz", **schedules)
    after = {name: digest(source_path / name) for name in INPUTS}
    assert before == after
    save_json(output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source_path),
        "start": str(source.sessions[0].date()), "end": str(source.sessions[-1].date()), "sessions": days,
        "families": names, "source_sha256": before, "source_unchanged": True,
        "fits": configurations, "methods": METHODS, "catalog": series_catalog,
        "frozen": "Economic alpha parameters, existing daily volatility normalization, curve/roll/ADV policy, and cached prices",
        "retrained": "Family admission, recipe (clock/holding period/direction), and nonnegative equal-weight combination",
        "selection": "Within each past-only window, choose a delay-robust doubled-cost recipe per family; best family, or greedy additions up to three. Same 126/60-session, 55%-coverage, 15-active-day and two-half-positive gates as prior study.",
        "fees": "USD 2.16 per micro round trip, plus original price slippage. Identical asset/recipe orders net before costs. Different windows retain separate round trips; adjacent-boundary savings are not claimed.",
        "gross_cap": "Unit planned net gross across all windows and assets, scale down only. Four-market book does not recycle the missing silver budget.",
        "availability": "Fresh prior-close price with one-observed-session lagged contract flows; alternative older settled forecast. One complete session embargo on fitted intraday outcomes.",
        "research_bias": "The ten-family roster and original economic parameter regions are hindsight-informed. These causal replays do not erase that bias or constitute a new pristine holdout.",
        "limitations": "Missing active returns stay NaN. Pre-July-2026 listed-month identities partly inferred; SIL thin/ends September. No quote depth/spread execution proof. Intraday data start in June 2024, not 2020.",
        "baseline_reconciliation_checks": baseline_checks, "attribution_checks": attribution_checks,
        "prefix_invariance_checks": prefix_checks, "scalar_checks": len(scalar_rows),
        "new_market_api_requests": False, "database_access": False, "production_changes": False,
    })
    print(f"[complete] {len(path_frames)} portfolios; {baseline_checks} prior-baseline reconciliations; {len(scalar_rows)} scalar checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source.resolve(), args.output.resolve())
