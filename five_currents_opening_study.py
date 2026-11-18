"""Cached-data opening experiment. No broker calls, publishing, or database writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from five_currents_front_intraday import (
    HOURS, MICROS, MICRO_MULTIPLIERS, ROOTS, ROUND_TRIP_FEE,
    SLIPPAGE_PER_SIDE, VARIANTS, evaluate, observed, same_intraday_instrument,
    valid_price,
)

PRICE_CONCEPTS = (
    "curve_quality_reversal", "carry_confirmed_momentum", "carry_basis_reversal",
)
FOUR = ("CL", "NG", "GC", "HG")


def stats(values):
    """Numerically equivalent to the audited Series implementation, without copies."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if not count:
        return {"days": 0, **dict.fromkeys(("sharpe", "total_return", "max_drawdown", "annual_vol",
                                           "win_rate", "median_gain", "median_loss"), np.nan)}
    equity = np.cumprod(1.+values)
    deviation = values.std(ddof=1) if count > 1 else np.nan
    positive, negative = values[values > 0], values[values < 0]
    active = len(positive)+len(negative)
    return {"days": count, "sharpe": values.mean()/deviation*np.sqrt(252) if deviation > 0 else np.nan,
            "total_return": equity[-1]-1,
            "max_drawdown": np.min(equity/np.maximum(np.maximum.accumulate(equity), 1.)-1.),
            "annual_vol": deviation*np.sqrt(252),
            "win_rate": len(positive)/active if active else np.nan,
            "median_gain": float(np.median(positive)) if len(positive) else np.nan,
            "median_loss": float(np.median(negative)) if len(negative) else np.nan}


def session_open(session):
    previous_date = pd.Timestamp(session).normalize() - pd.Timedelta(days=1)
    return (previous_date + pd.Timedelta(hours=17)).tz_localize(
        "America/Chicago").tz_convert("UTC")


def lag_contract_fields(wide, mapping_long, metadata):
    """Retrieve the previous panel session's same listed contract, even across rolls."""
    from us_commodity_research_core import _field_matrix, _mapping_matrix

    result = wide.copy()
    for root in ROOTS:
        mapping = _mapping_matrix(mapping_long, metadata, root, wide.index)
        for field in ("PX_VOLUME", "OPEN_INT"):
            values = _field_matrix(wide, root, field).reindex(wide.index)
            mapping = mapping.reindex(columns=values.columns)
            lagged = np.full(values.shape, np.nan)
            for position in range(1, len(values)):
                previous = {}
                for contract, value in zip(mapping.iloc[position-1], values.iloc[position-1]):
                    if pd.notna(contract) and np.isfinite(value):
                        if contract in previous and previous[contract] != value:
                            raise AssertionError("Conflicting previous-session contract values")
                        previous[contract] = value
                lagged[position] = [previous.get(contract, np.nan)
                                    for contract in mapping.iloc[position]]
            for column, rank in enumerate(values.columns):
                key = (f"{root}{rank} Comdty", field)
                if key in result:
                    result[key] = lagged[:, column]
    return result


def rebuild_concepts(original, operational, wide, *, rank_limits=None):
    return rebuild_concepts_variants(original, operational, wide, (rank_limits,))[0]


def rebuild_concepts_variants(original, operational, wide, rank_limits_list):
    from five_currents_daily.curve_activation import mixed_concepts
    from five_currents_daily.engine import DEFAULT_SOURCE_DIR
    from run_us_commodity_fixed_vintage_robustness import (
        candidate_weights_from_row, install_frozen_features,
    )
    from run_us_commodity_front_roll_causal_ridge import build_signal_libraries
    from us_commodity_execution_overlays import build_execution_overlays
    from us_commodity_front_roll_core import install_execution_ranks
    from us_commodity_pooled_harmonization_core import (
        build_parsimonious_concepts, deduplicate_obv_families,
    )
    from us_commodity_rank1_family_walkforward_core import equal_economic_regions

    library = pd.read_csv(DEFAULT_SOURCE_DIR / "neighborhood_library.csv")
    panel = install_execution_ranks(original.panel, wide, original.combined.mapping,
                                   original.signal_ranks)
    libraries, _ = build_signal_libraries(panel, wide, original.combined.mapping,
                                         required_sources={"technical", "carry_expanded"})
    # ADV gates already start at T-1; do not apply an accidental second delay.
    overlays = build_execution_overlays(panel, original.combined.wide, original.combined.mapping)
    lookup = install_frozen_features(panel, libraries, library)
    candidates = np.stack([candidate_weights_from_row(panel, overlays, lookup, row)
                           for row in library.itertuples(index=False)])
    regions = equal_economic_regions(candidates, library)
    twelve = deduplicate_obv_families(regions.weights, regions.names)
    ten = build_parsimonious_concepts(twelve.weights, twelve.names)
    assert ten.names == operational.concept_names
    temporary = replace(original, combined=replace(original.combined, wide=wide),
                        concept_weights=ten.weights)
    with patch("five_currents_daily.curve_activation.build_execution_overlays", return_value=overlays):
        return tuple(mixed_concepts(temporary, rank_limits=limits) for limits in rank_limits_list)


def allocated_sleeves(concepts, names, index, selection, coefficients, allowed=None):
    """Decompose the archived allocator into signed alpha weights, without refitting."""
    from five_currents_daily.curve_signals import frozen_allocation

    selection = selection.copy()
    if allowed is not None:
        selection["selected"] &= selection.economic_family.isin(allowed)
    final = frozen_allocation(concepts, names, index, selection, coefficients)
    sleeves = np.zeros_like(concepts)
    cancellation_cells = 0
    for ordinal, segment in selection.groupby("refit_ordinal", sort=True):
        first = segment.iloc[0]
        positions = np.flatnonzero((index >= first.forecast_start) & (index <= first.forecast_end))
        admitted = [names.index(name) for name in segment.loc[segment.selected, "economic_family"]]
        if not admitted:
            continue
        votes = concepts[admitted][:, positions] / len(admitted)
        raw = votes.sum(axis=0)
        factors = np.divide(final[positions], raw, out=np.zeros_like(raw), where=np.abs(raw)>1e-12)
        # For exactly cancelling votes, use the segment's positive risk tilt. Their
        # signed sum stays zero, but independent sleeves still incur trading costs.
        coeff = coefficients.loc[coefficients.refit_ordinal.eq(ordinal)].set_index("product")
        tilt = coeff.reindex(ROOTS).inverse_volatility_coefficient.to_numpy(float)
        cancelled = (np.abs(raw) <= 1e-12) & (np.abs(votes).sum(axis=0) > 1e-12)
        cancellation_cells += int(cancelled.sum())
        factors = np.where(cancelled, tilt[None, :], factors)
        sleeves[np.ix_(admitted, positions, np.arange(len(ROOTS)))] = votes * factors[None, :, :]
    np.testing.assert_allclose(sleeves.sum(axis=0), final, atol=1e-10)
    return final, sleeves, cancellation_cells


def opening_predictors(weights, sessions, *, previous_weights=None):
    """Use the most recent completed daily observation strictly before the new session."""
    index = weights.index
    previous_weights = weights if previous_weights is None else previous_weights
    rows = []
    for session in sessions:
        position = int(index.searchsorted(session, side="left")-1)
        if position < 1:
            continue
        source = index[position]
        for root in ROOTS:
            value = float(weights.iloc[position][root])
            change = value - float(previous_weights.iloc[position-1][root])
            rows.append({"session": session, "signal_date": source, "root": root,
                         "net_magnitude": value, "change_magnitude": change,
                         "net_sign": np.sign(value)*.2 if abs(value)>1e-12 else 0.,
                         "change_sign": np.sign(change)*.2 if abs(change)>1e-12 else 0.})
    return pd.DataFrame(rows)


def read_opening_anchors(sessions, output, *, table="bars_1m"):
    from five_currents_daily.db import readonly_connection
    from psycopg2 import sql

    if table not in ("bars_1m", "bars_1m_2m"):
        raise ValueError("Only cached minute tables are allowed")
    stamps = sorted({(session_open(day)+pd.Timedelta(hours=hour)).to_pydatetime()
                     for day in sessions for hour in (0, *HOURS)})
    fields = ("ts", "open", "close", "volume", "bar_count", "symbol", "local_symbol", "expiry")
    rows = []
    print(f"[prices] {len(stamps):,} exact timestamps; read-only cached tables, no API calls", flush=True)
    with readonly_connection(statement_timeout_ms=7000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET application_name='opening_signal_study_readonly'")
            for symbol in (*ROOTS, *MICROS.values()):
                query = sql.SQL("SELECT ts,open,close,volume,bar_count,symbol,local_symbol,expiry "
                                "FROM public.{} WHERE symbol=%s AND ts=ANY(%s::timestamptz[])").format(sql.Identifier(table))
                count = 0
                for start in range(0, len(stamps), 256):
                    cursor.execute(query, (symbol, stamps[start:start+256]))
                    batch = cursor.fetchall()
                    rows.extend(dict(zip(fields, row)) for row in batch)
                    count += len(batch)
                    time.sleep(.04)
                print(f"[prices] {symbol}: {count:,} stored endpoints", flush=True)
    frame = pd.DataFrame(rows, columns=fields)
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    if frame.duplicated(["symbol", "ts"]).any():
        raise AssertionError("Duplicate stored front-series endpoints")
    frame.to_parquet(output / "opening_minute_anchors.parquet", index=False)
    return frame


def opening_moves(bars, sessions, *, require_trade=False):
    lookup = {(r.symbol, r.ts): r for r in bars.itertuples()}
    eligible = observed if require_trade else valid_price
    rows = []
    for session in sessions:
        stamp = session_open(session)
        for root in ROOTS:
            chosen = next((lookup.get((symbol, stamp)) for symbol in (root, MICROS[root])
                           if eligible(lookup.get((symbol, stamp)))), None)
            for hours in HOURS:
                end = stamp + pd.Timedelta(hours=hours)
                record = {"session": session, "root": root, "hours": hours,
                          "entry_utc": stamp, "exit_utc": end, "status": "missing entry"}
                if chosen is not None:
                    record.update(symbol=chosen.symbol, local_symbol=chosen.local_symbol,
                                  expiry=chosen.expiry, entry_price=chosen.open)
                    last = lookup.get((chosen.symbol, end))
                    if not eligible(last):
                        record["status"] = "missing exit"
                    elif not same_intraday_instrument(chosen, last):
                        record["status"] = "front series rolled inside holding interval"
                    else:
                        record.update(status="matched", exit_price=last.open,
                                      price_return=last.open/chosen.open-1.,
                                      entry_trade_confirmed=observed(chosen),
                                      exit_trade_confirmed=observed(last))
                rows.append(record)
    result = pd.DataFrame(rows)
    for field in ("entry_price", "exit_price", "price_return"):
        if field not in result:
            result[field] = np.nan
    return result


def daily_totals(paths, roots):
    selected = paths.loc[paths.root.isin(roots)]
    columns = ["gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return"]
    group = selected.groupby("session")
    ready = group.ready.all() & group.root.nunique().eq(len(roots))
    totals = group[columns].sum(min_count=len(roots)).where(ready, np.nan)
    totals["gross_exposure"] = selected.weight.abs().groupby(selected.session).sum()
    totals["ready"] = ready
    return totals


def assert_same_gross_pnl(first, second):
    compared = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    np.testing.assert_allclose(compared["first"], compared["second"], atol=1e-10)


def collect_metrics(paths, policy, signal, metrics, curves, *, cost_only=False):
    for (variant, hours), group in paths.groupby(["variant", "hours"]):
        for book, roots in (("four", FOUR), ("five", ROOTS), *((root, (root,)) for root in ROOTS)):
            totals = daily_totals(group, roots)
            metadata = {"policy": policy, "signal": signal, "variant": variant,
                        "hours": int(hours), "book": book}
            for basis in ("gross_return", "net_return", "slippage_net_return"):
                values = totals[basis]
                active = totals.ready & totals.gross_exposure.gt(1e-12)
                metrics.append({**metadata, "cost_basis": basis,
                                "active_days": int(active.sum()),
                                "start": values.first_valid_index(), "end": values.last_valid_index(),
                                "mean_gross_exposure": float(totals.loc[totals.ready, "gross_exposure"].mean()),
                                **stats(values)})
            if book in ("four", "five") and not cost_only:
                curves.append(totals.reset_index().assign(**metadata))


def run(args):
    from five_currents_daily.dashboard_core import dollar_quote_replay

    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    source_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    with cache.open("rb") as handle:
        original = pickle.load(handle)
    replay = dollar_quote_replay(original, curve_activation=Path("five_currents_dashboard/curve_policy_activation"))
    pd.testing.assert_frame_equal(replay.weights,
        pd.read_parquet("five_currents_dashboard/snapshots/000032_2026-10-09/weights.parquet"),
        check_exact=False, atol=1e-10, rtol=1e-10)
    sessions = pd.bdate_range(args.start, args.end)
    if args.bar_cache:
        bars = pd.read_parquet(args.bar_cache)
        bars.to_parquet(output / "opening_minute_anchors.parquet", index=False)
    else:
        bars = read_opening_anchors(sessions, output)
    first = bars.ts.min().tz_convert("Asia/Singapore").normalize().tz_localize(None)
    last = bars.ts.max().tz_convert("Asia/Singapore").normalize().tz_localize(None)
    sessions = sessions[(sessions >= first) & (sessions <= last)]
    moves = opening_moves(bars, sessions)
    moves.to_parquet(output / "opening_price_moves.parquet", index=False)
    if args.concept_cache:
        print("[signals] Reusing the just-verified original and input-lag feature arrays", flush=True)
        saved = np.load(args.concept_cache)
        rebuilt, lagged = saved["original"], saved["lagged"]
    else:
        print("[signals] Verifying a full frozen-parameter feature rebuild against the dashboard", flush=True)
        rebuilt = rebuild_concepts(original, replay, original.combined.wide)
        print("[signals] Rebuilding the same parameters with same-contract T-1 volume/OI", flush=True)
        lagged_wide = lag_contract_fields(original.combined.wide, original.combined.mapping, original.panel.metadata)
        lagged = rebuild_concepts(original, replay, lagged_wide)
    np.testing.assert_allclose(rebuilt, replay.concept_weights, atol=1e-10, equal_nan=True)
    assert lagged.shape == rebuilt.shape
    price_positions = [replay.concept_names.index(name) for name in PRICE_CONCEPTS]
    np.testing.assert_allclose(lagged[price_positions], rebuilt[price_positions], atol=1e-10, equal_nan=True)
    np.savez_compressed(output / "concept_weights.npz", original=rebuilt, lagged=lagged)
    metadata = pd.DataFrame({"concept": replay.concept_names,
                             "needs_current_volume_or_oi": [name not in PRICE_CONCEPTS for name in replay.concept_names]})
    metadata.to_csv(output / "alpha_dependencies.csv", index=False)
    metrics, curves, predictions, comparison, gross_metrics = [], [], [], [], []
    weights_by_policy = {}
    for policy, concepts, allowed in (
        ("original_final_flows_reference", rebuilt, None),
        ("lagged_volume_oi", lagged, None),
        ("price_only", rebuilt, PRICE_CONCEPTS),
    ):
        final, sleeves, cancelled = allocated_sleeves(concepts, replay.concept_names, replay.weights.index,
            replay.selection_audit, replay.product_coefficients, allowed)
        weights = pd.DataFrame(final, index=replay.weights.index, columns=ROOTS)
        weights_by_policy[policy] = weights
        if policy == "original_final_flows_reference":
            pd.testing.assert_frame_equal(weights, replay.weights, check_exact=False, atol=1e-10, rtol=1e-10)
        weights.to_parquet(output / f"{policy}_targets.parquet")
        predictor = opening_predictors(weights, sessions)
        predictions.append(predictor.assign(policy=policy, signal="portfolio_net"))
        portfolio = evaluate(predictor, moves)
        collect_metrics(portfolio, policy, "portfolio_net", metrics, curves)
        comparison.append(portfolio.assign(policy=policy, signal="portfolio_net"))
        print(f"[tests] {policy}: net/change/sign/magnitude/reversed; {len(sessions)} candidate sessions", flush=True)
        sleeve_paths = []
        for position, name in enumerate(replay.concept_names):
            if allowed is not None and name not in allowed:
                continue
            standalone = pd.DataFrame(concepts[position], index=weights.index, columns=ROOTS)
            p = opening_predictors(standalone, sessions)
            predictions.append(p.assign(policy=policy, signal=name))
            collect_metrics(evaluate(p, moves), policy, name, metrics, curves, cost_only=True)
            sleeve = pd.DataFrame(sleeves[position], index=weights.index, columns=ROOTS)
            sleeve_path = evaluate(opening_predictors(sleeve, sessions), moves)
            sleeve_paths.append(sleeve_path)
        # Independent sleeves have identical linear gross P&L for magnitude tests,
        # but commission/slippage must be charged on pre-netted quantities.
        independent = pd.concat(sleeve_paths)
        keys = ["session", "root", "hours", "variant"]
        values = ["gross_return", "commission_return", "slippage_return", "net_return", "slippage_net_return"]
        grouped = independent.groupby(keys)
        gross_book = grouped[values].sum(min_count=1)
        ready = grouped.ready.all()
        gross_book.loc[~ready] = np.nan
        gross_book["ready"] = ready
        gross_book["weight"] = independent.weight.abs().groupby([independent[key] for key in keys]).sum()
        gross_book = gross_book.reset_index()
        # Sleeve sign books are not the sign of the net book; retain them as diagnostics.
        collect_metrics(gross_book, policy, "portfolio_unnetted_sleeves", gross_metrics, curves)
        net_magnitude = portfolio.loc[portfolio.variant.eq("net_magnitude")].set_index(keys)
        gross_magnitude = gross_book.loc[gross_book.variant.eq("net_magnitude")].set_index(keys)
        assert_same_gross_pnl(net_magnitude.gross_return, gross_magnitude.gross_return)
        print(f"[checks] {policy}: alpha votes sum to net targets; {cancelled} exactly cancelling cells retained", flush=True)
    # A close-only refresh is the lagged-flow forecast minus the previously finalized
    # target, rather than subtracting yesterday's other lagged-flow preview.
    close_preview = opening_predictors(weights_by_policy["lagged_volume_oi"], sessions,
        previous_weights=weights_by_policy["original_final_flows_reference"])
    close_paths = evaluate(close_preview, moves)
    close_paths = close_paths.loc[close_paths.variant.str.contains("change")]
    collect_metrics(close_paths, "close_refresh_vs_previous_final_target", "portfolio_net", metrics, curves)
    predictions.append(close_preview.assign(policy="close_refresh_vs_previous_final_target", signal="portfolio_net"))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "all_signal_metrics.csv", index=False)
    pd.DataFrame(gross_metrics).to_csv(output / "unnetted_sleeve_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / "opening_predictors.parquet", index=False)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_parquet(output / "portfolio_daily_returns.parquet", index=False)
    compared = pd.concat(comparison, ignore_index=True)
    common = []
    for (variant, hours), group in compared.groupby(["variant", "hours"]):
        for book, roots in (("four", FOUR), ("five", ROOTS)):
            totals = {policy: daily_totals(g, roots) for policy, g in group.groupby("policy")}
            valid = pd.concat({k: v.ready for k, v in totals.items()}, axis=1).all(axis=1)
            for policy, values in totals.items():
                for basis in ("gross_return", "net_return", "slippage_net_return"):
                    common.append({"policy": policy, "variant": variant, "hours": hours,
                                   "book": book, "cost_basis": basis, **stats(values.loc[valid, basis])})
    pd.DataFrame(common).to_csv(output / "common_date_policy_metrics.csv", index=False)
    moves.groupby(["root", "hours", "status"]).size().rename("days").to_csv(output / "coverage.csv")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "entry": "17:00 America/Chicago on the calendar evening before the session; 06:00/07:00 SGT",
        "holding_hours": HOURS, "variants": VARIANTS, "reversed_signals": True,
        "price_only_concepts": PRICE_CONCEPTS, "alpha_parameters_changed": False,
        "refit_decisions_frozen": True, "product_risk_coefficients_frozen": True,
        "lag_policy": "Previous observed daily panel session, same listed contract; missing remains missing",
        "adv_gates": "Original T-1 gates retained; not lagged twice",
        "close_refresh": "Lagged-flow current-close forecast minus previous finalized forecast",
        "original_reference_warning": "Final same-session volume/OI is an availability-optimistic reference, not opening-tradable",
        "historical_publication_vintages_available": False, "database_access": "cached exact timestamps, read-only" if not args.bar_cache else "none",
        "market_api_calls": False, "production_changes": False,
        "commission_per_micro_round_trip_usd": ROUND_TRIP_FEE,
        "source_cache_sha256": source_hash,
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if hashlib.sha256(cache.read_bytes()).hexdigest() != source_hash:
        raise AssertionError("The live cache changed during this offline study; results need a pinned rerun")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = metrics.loc[metrics.signal.eq("portfolio_net") & metrics.book.eq("four") & metrics.cost_basis.eq("net_return")]
    for policy, group in result.groupby("policy"):
        print("\n[commission-net Sharpe / four markets] " + policy, flush=True)
        print(group.pivot(index="variant", columns="hours", values="sharpe").round(3).to_string(), flush=True)
    print("[complete] " + str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-10-09")
    parser.add_argument("--bar-cache", type=Path)
    parser.add_argument("--concept-cache", type=Path, help="Reuse already-verified arrays from this study")
    run(parser.parse_args())
