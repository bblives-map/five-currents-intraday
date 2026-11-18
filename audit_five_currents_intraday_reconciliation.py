"""Read-only, cached-model timing and price-path audit. Never publishes a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_front_intraday import (
    HOURS, MICROS, ROOTS, MICRO_MULTIPLIERS, SLIPPAGE_PER_SIDE, ROUND_TRIP_FEE,
    daily_predictors, entry_time, evaluate, front_month_moves,
    stats, valid_price, same_intraday_instrument,
)

FOUR = ("CL", "NG", "GC", "HG")


def chicago_close(day):
    return (pd.Timestamp(day).normalize() + pd.Timedelta(hours=16)).tz_localize(
        "America/Chicago").tz_convert("UTC")


def canonical(root, row):
    from five_currents_daily.reconciliation import canonical_contract
    key = canonical_contract(row.symbol, row.local_symbol, row.ts)
    return root + ":" + key.split(":", 1)[1] if key else None


def held_predictors(weights, calendar, start, end):
    """Positions entered at a prior close, rather than forecasts newly issued there."""
    calendar = pd.DatetimeIndex(calendar).union(weights.index).sort_values().unique()
    positions = calendar.searchsorted(weights.index) + 1
    valid = positions < len(calendar)
    eligible_weights = weights.loc[valid]
    entry_dates = calendar[positions[valid]]
    sessions = pd.bdate_range(start, end)
    rows = []
    for day in sessions:
        current = int(entry_dates.searchsorted(day, side="left")-1)
        previous_day = day-pd.offsets.BDay(1)
        prior = int(entry_dates.searchsorted(previous_day, side="left")-1)
        if current < 0:
            continue
        source = eligible_weights.index[current]
        assert source < entry_dates[current] < day
        for root in ROOTS:
            value = float(eligible_weights.iloc[current][root])
            previous = float(eligible_weights.iloc[prior][root]) if prior >= 0 else 0.
            change = value-previous
            rows.append({"session": day, "signal_date": source, "root": root,
                         "net_magnitude": value, "change_magnitude": change,
                         "net_sign": np.sign(value)*.2 if abs(value)>1e-12 else 0.,
                         "change_sign": np.sign(change)*.2 if abs(change)>1e-12 else 0.})
    return pd.DataFrame(rows)


def prepare(snapshot, cache, activation, output):
    from five_currents_daily.dashboard_core import _chart_history, dollar_quote_replay
    from us_commodity_asset_family_walkforward_core import product_net_returns
    from us_commodity_disciplined_core import fast_product_backtest

    print("[1/4] Reconstructing dashboard concepts, frozen selections and risk coefficients", flush=True)
    with cache.open("rb") as handle:
        replay = pickle.load(handle)
    replay = dollar_quote_replay(replay, curve_activation=activation)
    weights = pd.read_parquet(snapshot / "weights.parquet")
    pd.testing.assert_frame_equal(replay.weights, weights, check_exact=False, rtol=1e-10, atol=1e-12)
    admin = json.loads((snapshot / "admin.json").read_text())
    history = _chart_history(replay)
    assert history["dates"] == admin["history"]["dates"]
    np.testing.assert_allclose(np.array(history["returns"], dtype=float),
                               np.array(admin["history"]["returns"], dtype=float),
                               equal_nan=True, atol=1e-12, rtol=1e-10)
    for actual, expected in zip(history["signals"], admin["history"]["signals"], strict=True):
        assert actual["name"] == expected["name"]
        np.testing.assert_allclose(np.array(actual["returns"], dtype=float),
                                   np.array(expected["returns"], dtype=float),
                                   equal_nan=True, atol=1e-12, rtol=1e-10)
    path = fast_product_backtest(weights.to_numpy(), replay.panel, replay.execution, "base")
    contributions = product_net_returns(weights.to_numpy(), replay.execution, scenario="base")
    np.testing.assert_allclose(np.nansum(contributions, axis=1)[path.net_return.notna()],
                               path.net_return.dropna(), atol=1e-12, rtol=1e-10)
    from five_currents_daily_reconciliation_audit import independent_daily_ledger
    ledger = independent_daily_ledger(weights, replay.execution.target_return,
                                      replay.execution.identity, replay.execution.base_cost_rate)
    independent = ledger.groupby("signal_date").net_return.sum(min_count=5).reindex(weights.index)
    np.testing.assert_allclose(independent, path.net_return, equal_nan=True, atol=1e-10)
    ledger.to_parquet(output / "independent_daily_scalar_ledger.parquet", index=False)
    index = weights.index
    rows = []
    for ordinal in range(len(index)-2):
        for column, root in enumerate(ROOTS):
            rows.append({"forecast_date": index[ordinal], "entry_close_date": index[ordinal+1],
                         "mark_date": index[ordinal+2], "root": root,
                         "held_weight": weights.iloc[ordinal, column],
                         "fresh_weight": weights.iloc[ordinal+1, column],
                         "daily_price_return": replay.execution.target_return[ordinal, column],
                         "daily_model_net": contributions[ordinal, column],
                         "identity": replay.execution.identity[ordinal, column],
                         "base_cost_rate": replay.execution.base_cost_rate[ordinal, column]})
    reference = pd.DataFrame(rows)
    from five_currents_daily.reconciliation import canonical_contract
    reference["contract_key"] = [canonical_contract(r.root, r.identity, r.forecast_date)
                                 for r in reference.itertuples()]
    from five_currents_daily.reconciliation import marks_from_wide
    input_marks = marks_from_wide(replay.combined.wide, replay.combined.mapping)
    prices = {(r.date, r.product, r.contract_key): r.price for r in input_marks.itertuples()}
    source_checks = []
    for item in reference.itertuples():
        first = prices.get((item.entry_close_date, item.root, item.contract_key))
        last = prices.get((item.mark_date, item.root, item.contract_key))
        actual = last/first-1. if first and last else np.nan
        source_checks.append({"forecast_date": item.forecast_date, "root": item.root,
                              "source_price_return": actual, "model_price_return": item.daily_price_return,
                              "absolute_error": abs(actual-item.daily_price_return)})
    source_checks = pd.DataFrame(source_checks)
    frozen = source_checks.loc[source_checks.forecast_date.lt(pd.Timestamp("2026-07-16"))]
    if frozen.absolute_error.dropna().max() > 1e-10:
        raise AssertionError("Independent input-price lookup disagrees with frozen daily forward returns")
    source_checks.to_csv(output / "independent_daily_price_pair_checks.csv", index=False)
    (output / "daily_reproduction_verification.json").write_text(json.dumps({
        "final_weights_match_snapshot": True, "portfolio_returns_match_snapshot": True,
        "standalone_concepts_match_snapshot": len(history["signals"]),
        "independent_scalar_ledger_matches": True,
        "frozen_input_price_pair_count": int(frozen.absolute_error.notna().sum()),
        "frozen_input_price_pair_maximum_error": float(frozen.absolute_error.max()),
    }, indent=2), encoding="utf-8")
    reference.to_parquet(output / "daily_forecast_execution_mark_reference.parquet", index=False)
    weights.to_parquet(output / "verified_dashboard_weights.parquet")
    history_returns = pd.DataFrame({root: value["returns"] for root, value in history["assets"].items()},
                                   index=pd.to_datetime(history["dates"]))
    history_returns["portfolio"] = history["returns"]
    history_returns.to_parquet(output / "verified_dashboard_marked_returns.parquet")
    print("[1/4] Dashboard weights and marked P&L reproduced to numerical precision", flush=True)
    return weights, reference, history_returns


def read_anchors(sessions, output):
    from five_currents_daily.db import readonly_connection
    from psycopg2 import sql
    wanted = set()
    for day in sessions:
        wanted.update(entry_time(day)+pd.Timedelta(hours=h) for h in (0, *HOURS, 24))
        wanted.add(chicago_close(day)-pd.Timedelta(minutes=1))
    stamps = sorted(x.to_pydatetime() for x in wanted)
    rows = []
    fields = ("ts", "open", "close", "volume", "bar_count", "symbol", "local_symbol", "expiry")
    print(f"[2/4] Indexed, read-only anchor reads: {len(stamps):,} timestamps", flush=True)
    with readonly_connection(statement_timeout_ms=7000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET application_name='daily_intraday_audit_readonly'")
            for table in ("bars_1m", "bars_1m_2m"):
                for symbol in (*ROOTS, *MICROS.values()):
                    query = sql.SQL("SELECT ts,open,close,volume,bar_count,symbol,local_symbol,expiry "
                                    "FROM public.{} WHERE symbol=%s AND ts=ANY(%s::timestamptz[])").format(sql.Identifier(table))
                    count = 0
                    for start in range(0, len(stamps), 384):
                        cursor.execute(query, (symbol, stamps[start:start+384]))
                        batch = cursor.fetchall()
                        rows.extend({**dict(zip(fields, row)), "source_table": table} for row in batch)
                        count += len(batch)
                        time.sleep(.04)
                    print(f"[2/4] {table}/{symbol}: {count:,} cached bars", flush=True)
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    assert not frame.duplicated(["source_table", "symbol", "ts"]).any()
    frame.to_parquet(output / "audited_minute_anchors.parquet", index=False)
    return frame


def portfolio_stats(frame, value, label, required=FOUR):
    selected = frame.loc[frame.root.isin(required)]
    grouped = selected.groupby("session")[value]
    pnl = grouped.sum(min_count=len(required))
    result = {"case": label, **stats(pnl)}
    return result, pnl


def analyze(weights, reference, daily, bars, original_source, output):
    print("[3/4] Replaying fresh forecasts and daily-held positions on identical cached marks", flush=True)
    old_predictors = pd.read_parquet(original_source / "daily_predictors.parquet")
    sessions = pd.DatetimeIndex(old_predictors.session.unique()).sort_values()
    front = bars.loc[bars.source_table.eq("bars_1m")]
    moves = front_month_moves(front, sessions)
    fresh = daily_predictors(weights, sessions.min(), sessions.max())
    held = held_predictors(weights, daily.index, sessions.min(), sessions.max())
    ref = reference.loc[reference.mark_date.isin(sessions)].copy()
    ref = ref.rename(columns={"mark_date": "session"})
    for row in ref.itertuples():
        mask = held.session.eq(row.session) & held.root.eq(row.root)
        if row.session <= pd.Timestamp("2026-07-20"):
            np.testing.assert_allclose(held.loc[mask, "net_magnitude"], row.held_weight, atol=1e-12)
    metrics = []
    for label, predictors in (("fresh_forecast", fresh), ("daily_held_position", held)):
        paths = evaluate(predictors, moves)
        paths.to_parquet(output / f"{label}_intraday_asset_returns.parquet", index=False)
        for (variant, hours), group in paths.groupby(["variant", "hours"]):
            for book, roots in (("four", FOUR), ("five", ROOTS)):
                selected = group.loc[group.root.isin(roots)]
                ready = selected.groupby("session").ready.all()
                for price in ("gross_return", "net_return", "slippage_net_return"):
                    pnl = selected.groupby("session")[price].sum(min_count=len(roots)).where(ready)
                    metrics.append({"timing": label, "variant": variant, "hours": hours,
                                    "book": book, "cost_basis": price, **stats(pnl)})
    pd.DataFrame(metrics).to_csv(output / "intraday_timing_metrics.csv", index=False)

    lookup = {(r.source_table, r.symbol, r.ts): r for r in bars.itertuples()}
    decomposition = []
    for row in ref.loc[ref.root.isin(FOUR)].itertuples():
        stamp = entry_time(row.session)
        for route in ("stored_front", "daily_contract_control"):
            choices = [("bars_1m", row.root), ("bars_1m", MICROS[row.root])]
            if route == "daily_contract_control":
                choices += [("bars_1m_2m", row.root), ("bars_1m_2m", MICROS[row.root])]
            chosen = None
            for table, symbol in choices:
                candidate = lookup.get((table, symbol, stamp))
                if valid_price(candidate) and (route == "stored_front" or canonical(row.root, candidate) == row.contract_key):
                    chosen = candidate
                    break
            record = {"session": row.session, "root": row.root, "route": route,
                      "forecast_date": row.forecast_date, "entry_close_date": row.entry_close_date,
                      "held_weight": row.held_weight, "fresh_weight": row.fresh_weight,
                      "daily_contract": row.contract_key, "daily_price_return": row.daily_price_return,
                      "daily_model_net": row.daily_model_net, "status": "missing entry"}
            if chosen is not None:
                record.update(symbol=chosen.symbol, source_table=chosen.source_table,
                              local_symbol=chosen.local_symbol, entry_price=chosen.open,
                              same_daily_contract=canonical(row.root, chosen) == row.contract_key,
                              entry_trade_confirmed=chosen.volume>0 or chosen.bar_count>0)
                previous_stamp = chicago_close(row.entry_close_date)-pd.Timedelta(minutes=1)
                close_stamp = chicago_close(row.session)-pd.Timedelta(minutes=1)
                anchors = {"previous_close": previous_stamp, "exit": stamp+pd.Timedelta(hours=12),
                           "session_close": close_stamp, "next_morning": stamp+pd.Timedelta(hours=24)}
                valid = {}
                for name, endpoint in anchors.items():
                    item = lookup.get((chosen.source_table, chosen.symbol, endpoint))
                    value = getattr(item, "close" if name in ("previous_close", "session_close") else "open", np.nan)
                    if item is not None and np.isfinite(value) and value>0 and same_intraday_instrument(chosen, item):
                        record[name+"_price"] = value
                        valid[name] = value
                if "previous_close" in valid:
                    record["overnight_return"] = chosen.open/valid["previous_close"]-1.
                if "exit" in valid:
                    record["standalone_intraday_return"] = valid["exit"]/chosen.open-1.
                if "exit" in valid and "previous_close" in valid:
                    record["intraday_on_close_base"] = (valid["exit"]-chosen.open)/valid["previous_close"]
                if "exit" in valid and "session_close" in valid and "previous_close" in valid:
                    record["after_exit_on_close_base"] = (valid["session_close"]-valid["exit"])/valid["previous_close"]
                if "previous_close" in valid and "session_close" in valid:
                    record["minute_close_return"] = valid["session_close"]/valid["previous_close"]-1.
                if all(key in valid for key in ("previous_close", "exit", "session_close")):
                    record["status"] = "complete session decomposition"
                    np.testing.assert_allclose(record["overnight_return"]+record["intraday_on_close_base"]+
                                               record["after_exit_on_close_base"], record["minute_close_return"], atol=1e-12)
                if "next_morning" in valid:
                    record["morning_24h_return"] = valid["next_morning"]/chosen.open-1.
            decomposition.append(record)
    decomp = pd.DataFrame(decomposition)
    decomp.to_parquet(output / "per_asset_price_path_decomposition.parquet", index=False)
    comparison_rows, daily_paths = [], []
    for route, group in decomp.groupby("route"):
        for timing, weight in (("held", "held_weight"), ("fresh", "fresh_weight")):
            for component in ("daily_price_return", "overnight_return", "intraday_on_close_base",
                              "after_exit_on_close_base", "minute_close_return", "morning_24h_return"):
                group = group.copy()
                group["pnl"] = group[weight]*group.get(component, pd.Series(np.nan, index=group.index))
                group.loc[group[weight].abs().le(1e-12), "pnl"] = 0.
                result, pnl = portfolio_stats(group, "pnl", f"{route}/{timing}/{component}")
                comparison_rows.append(result)
                daily_paths.append(pd.DataFrame({"session": pnl.index, "return": pnl.to_numpy(), "case": result["case"]}))
        complete = group.loc[group.status.eq("complete session decomposition") & group.daily_price_return.notna()].copy()
        complete_dates = complete.groupby("session").root.nunique().eq(4)
        dates = complete_dates.index[complete_dates]
        for component in ("daily_price_return", "overnight_return", "intraday_on_close_base",
                          "after_exit_on_close_base", "minute_close_return"):
            for timing, weight in (("held", "held_weight"), ("fresh", "fresh_weight")):
                subset = complete.loc[complete.session.isin(dates)].copy()
                subset["pnl"] = subset[weight]*subset[component]
                result, _ = portfolio_stats(subset, "pnl", f"COMMON/{route}/{timing}/{component}")
                comparison_rows.append(result)
    pd.DataFrame(comparison_rows).to_csv(output / "price_path_component_metrics.csv", index=False)
    pd.concat(daily_paths, ignore_index=True).to_parquet(output / "component_daily_returns.parquet", index=False)
    standalone, standalone_paths = [], []
    for route, group in decomp.groupby("route"):
        for timing, weight in (("held", "held_weight"), ("fresh", "fresh_weight")):
            part = group.copy()
            part["gross"] = part[weight]*part.standalone_intraday_return
            part["commission"] = part[weight].abs()*ROUND_TRIP_FEE/(part.entry_price*part.root.map(MICRO_MULTIPLIERS))
            part["slippage"] = part[weight].abs()*2.*part.root.map(SLIPPAGE_PER_SIDE)/part.entry_price
            part["net"] = part.gross-part.commission
            part["slippage_net"] = part.net-part.slippage
            part.loc[part[weight].abs().le(1e-12), ["gross", "commission", "slippage", "net", "slippage_net"]] = 0.
            for column in ("gross", "net", "slippage_net"):
                result, pnl = portfolio_stats(part, column, f"{route}/{timing}/{column}")
                standalone.append(result)
                standalone_paths.append(pd.DataFrame({"session": pnl.index, "return": pnl.to_numpy(), "case": result["case"]}))
    standalone_paths = pd.concat(standalone_paths, ignore_index=True)
    standalone_paths.to_parquet(output / "standalone_12h_daily_returns.parquet", index=False)
    pd.DataFrame(standalone).to_csv(output / "standalone_12h_metrics.csv", index=False)
    common_route = []
    wide = standalone_paths.pivot(index="session", columns="case", values="return")
    dates = wide.dropna().index
    for name in wide:
        common_route.append({"case": name, **stats(wide.loc[dates, name])})
    pd.DataFrame(common_route).to_csv(output / "common_date_12h_route_and_timing_metrics.csv", index=False)
    decomp.groupby(["route", "root", "same_daily_contract"], dropna=False).size().rename("days").to_csv(
        output / "contract_alignment_coverage.csv")
    benchmark = []
    for book, roots in (("four", FOUR), ("five", ROOTS)):
        pnl = daily[list(roots)].sum(axis=1, min_count=len(roots))
        for name, dates in (("full_history", daily.index), ("intraday_calendar", sessions)):
            benchmark.append({"book": book, "period": name, **stats(pnl.reindex(dates))})
    pd.DataFrame(benchmark).to_csv(output / "dashboard_daily_benchmarks.csv", index=False)
    print("[3/4] Timing and additive price-path decomposition complete", flush=True)
    print(pd.DataFrame(benchmark).to_string(index=False), flush=True)
    print(pd.DataFrame(comparison_rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, default=Path("five_currents_dashboard/snapshots/000032_2026-10-09"))
    parser.add_argument("--anchor-cache", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    activation = Path("five_currents_dashboard/curve_policy_activation")
    source = Path("five_currents_front_intraday_price_edge_20261010")
    inputs = [cache, args.snapshot / "admin.json", args.snapshot / "weights.parquet",
              source / "daily_predictors.parquet", Path(__file__)]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    weights, reference, daily = prepare(args.snapshot, cache, activation, output)
    predictors = pd.read_parquet(source / "daily_predictors.parquet")
    sessions = pd.DatetimeIndex(predictors.session.unique()).sort_values()
    anchor_days = sessions.union(pd.DatetimeIndex(reference.loc[reference.mark_date.isin(sessions), "entry_close_date"]))
    if args.anchor_cache:
        bars = pd.read_parquet(args.anchor_cache)
    else:
        bars = read_anchors(anchor_days, output)
    analyze(weights, reference, daily, bars, source, output)
    if args.anchor_cache:
        hashes[str(args.anchor_cache.resolve())] = hashlib.sha256(args.anchor_cache.read_bytes()).hexdigest()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "database_access": "none; cached anchors" if args.anchor_cache else "read-only indexed anchors",
                "market_api_calls": False, "dashboard_changes": False, "new_parameter_selection": False,
                "dashboard_alphas_and_weights_reconstructed": True, "dashboard_daily_pnl_reproduced": True,
                "input_sha256": hashes,
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[4/4] Audit saved to " + str(output), flush=True)


if __name__ == "__main__":
    main()
