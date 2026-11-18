"""Pin the existing daily model and export indexed, cached minute endpoints only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
DEFAULT_OUT = ROOT / "five_currents_execution_timing_20261023"
PRODUCTS = ("CL", "NG", "GC", "SI", "HG")
MICROS = {"CL": "MCL", "NG": "MHNG", "GC": "MGC", "SI": "SIL", "HG": "MHG"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def signals(output, start, end):
    from five_currents_daily.dashboard_core import dollar_quote_replay
    from five_currents_opening_study import allocated_sleeves, lag_contract_fields, rebuild_concepts, PRICE_CONCEPTS
    from five_currents_active_contracts import daily_liquidity, select_active_contracts

    if (output / "weight_arrays.npz").exists():
        raise FileExistsError("Frozen study inputs already exist; use --bars-only to resume the cache export")
    pinned = output / "inputs"
    pinned.mkdir(exist_ok=True)
    cache = ROOT / "five_currents_dashboard/private_replay_cache.pkl"
    content = cache.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    (pinned / "daily_replay.pkl").write_bytes(content)
    activation = ROOT / "five_currents_dashboard/curve_policy_activation"
    shutil.copytree(activation, pinned / "curve_policy_activation", dirs_exist_ok=True)
    original = pickle.loads(content)
    print("[signals] Applying the existing versioned model; no alpha parameter search", flush=True)
    replay = dollar_quote_replay(original, curve_activation=pinned / "curve_policy_activation")
    latest = sorted((ROOT / "five_currents_dashboard/snapshots").glob("*"+str(replay.last_signal_date.date())))[-1]
    saved_weights = pd.read_parquet(latest / "weights.parquet")
    pd.testing.assert_frame_equal(replay.weights, saved_weights, check_exact=False, atol=1e-10, rtol=1e-10)
    for name in ("weights.parquet", "selection_audit.parquet", "product_scaling.parquet"):
        shutil.copy2(latest / name, pinned / name)
    replay.combined.mapping.to_parquet(pinned / "contract_mapping.parquet", index=False)
    replay.combined.wide.to_parquet(pinned / "daily_panel.parquet")
    replay.paths["base"].to_parquet(output / "daily_baseline.parquet")
    replay.product_returns["base"].to_parquet(output / "daily_baseline_asset_returns.parquet")
    lagged_path = output / "weight_arrays.npz"
    if lagged_path.exists():
        raise FileExistsError("Existing frozen weights must not be overwritten")
    print("[signals] Rebuilding the identical definitions with same-contract T-1 volume/OI", flush=True)
    lagged_wide = lag_contract_fields(original.combined.wide, original.combined.mapping, original.panel.metadata)
    lagged = rebuild_concepts(original, replay, lagged_wide)
    price_ids = [replay.concept_names.index(name) for name in PRICE_CONCEPTS]
    np.testing.assert_allclose(lagged[price_ids], replay.concept_weights[price_ids], atol=1e-10, equal_nan=True)
    settled_net, settled_sleeves, _ = allocated_sleeves(replay.concept_weights, replay.concept_names,
        replay.weights.index, replay.selection_audit, replay.product_coefficients)
    lagged_net, lagged_sleeves, _ = allocated_sleeves(lagged, replay.concept_names,
        replay.weights.index, replay.selection_audit, replay.product_coefficients)
    np.testing.assert_allclose(settled_net, replay.weights.to_numpy(), atol=1e-10)
    np.savez_compressed(lagged_path, raw_settled=replay.concept_weights, raw_lagged=lagged,
                        allocated_settled=settled_sleeves, allocated_lagged=lagged_sleeves,
                        net_settled=settled_net, net_lagged=lagged_net,
                        dates=replay.weights.index.asi8)
    sessions = replay.weights.index[(replay.weights.index >= pd.Timestamp(start))
                                    & (replay.weights.index <= pd.Timestamp(end))]
    pd.DataFrame({"session": sessions}).to_parquet(output / "sessions.parquet", index=False)
    print("[contracts] Same pre-session nearby-liquidity mask for every commodity", flush=True)
    candidates = daily_liquidity(replay.combined.wide, replay.combined.mapping)
    targets, ledger = select_active_contracts(candidates, sessions, pd.DatetimeIndex(candidates.date.unique()))
    targets.to_parquet(output / "contract_targets.parquet", index=False)
    ledger.to_parquet(output / "liquidity_candidates.parquet", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "start": str(sessions.min().date()),
        "end": str(sessions.max().date()), "signal_names": list(replay.concept_names), "products": PRODUCTS,
        "source_cache": str(cache), "source_cache_sha256": source_hash, "source_snapshot": str(latest),
        "frozen_parameters": True, "existing_admission_and_risk_fits_preserved": True,
        "historical_publication_vintages_available": False, "new_market_api_requests": False,
        "production_changes": False, "price_only_families": PRICE_CONCEPTS,
        "lagged_definition": "Prior observed panel session's same listed contract volume/OI; original ADV gate unchanged",
        "research_end_policy": "Only completed sessions through 2026-10-22; no incomplete 23 October outcomes",
        "daily_weight_reconciliation_max_error": float(np.max(np.abs(settled_net-replay.weights.to_numpy()))),
        "files_sha256": {p.name:digest(p) for p in output.glob("*.npz")}}
    save_json(output / "source_manifest.json", manifest)
    print("[signals] Frozen and reconciled", manifest["start"], "to", manifest["end"], flush=True)


def minute_stamps(sessions):
    # A regular five-minute grid supports independent +/- 5/15/30 minute clock tests.
    stamps = []
    for session in sessions:
        close = (pd.Timestamp(session).normalize()+pd.Timedelta(hours=16)).tz_localize("America/Chicago")
        begin = (pd.Timestamp(session).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17)).tz_localize("America/Chicago")
        stamps.extend(pd.date_range(begin.tz_convert("UTC"), close.tz_convert("UTC"), freq="5min", inclusive="left"))
    return pd.DatetimeIndex(stamps).unique().sort_values()


def bars(output):
    from five_currents_daily.db import readonly_connection
    from psycopg2 import sql

    sessions = pd.DatetimeIndex(pd.read_parquet(output / "sessions.parquet").session)
    stamps = minute_stamps(sessions)
    fields = ("ts", "open", "close", "volume", "bar_count", "local_symbol", "expiry")
    destination = output / "minute_cache"
    destination.mkdir(exist_ok=True)
    counts = []
    with readonly_connection(statement_timeout_ms=12000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET application_name='execution_timing_cached_readonly'")
            for table in ("bars_1m", "bars_1m_2m"):
                query = sql.SQL("SELECT ts,open,close,volume,bar_count,local_symbol,expiry FROM public.{} "
                               "WHERE symbol=%s AND ts=ANY(%s::timestamptz[])").format(sql.Identifier(table))
                for root in PRODUCTS:
                    for symbol in (MICROS[root], root):
                        path = destination / (table+"_"+symbol+".parquet")
                        if path.exists():
                            rows = len(pd.read_parquet(path, columns=["ts"]))
                        else:
                            pieces = []
                            for first in range(0, len(stamps), 4096):
                                cursor.execute(query, (symbol, stamps[first:first+4096].to_pydatetime().tolist()))
                                frame = pd.DataFrame(cursor.fetchall(), columns=fields)
                                pieces.append(frame)
                                time.sleep(.025)
                            frame = pd.concat(pieces, ignore_index=True)
                            frame["ts"] = pd.to_datetime(frame.ts, utc=True)
                            frame["table"], frame["symbol"], frame["root"] = table, symbol, root
                            if frame.duplicated("ts").any():
                                raise AssertionError("Duplicate stored minute endpoint")
                            frame.to_parquet(path, index=False)
                            rows = len(frame)
                        counts.append({"table":table,"root":root,"symbol":symbol,"rows":rows,"sha256":digest(path)})
                        print(f"[stored bars] {table} {symbol}: {rows:,} rows; local database only", flush=True)
    save_json(output / "minute_cache_manifest.json", {"exported_utc":datetime.now(timezone.utc).isoformat(),
        "read_only":True,"market_api_calls":False,"queries":"Indexed (symbol,ts) endpoints in 4096-stamp batches",
        "sampling_minutes":5,"stamps_requested_per_symbol":len(stamps),"sources":counts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2024-06-20")
    parser.add_argument("--end", default="2026-10-22")
    parser.add_argument("--signals-only", action="store_true")
    parser.add_argument("--bars-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        if not args.bars_only:
            signals(output, args.start, args.end)
        if not args.signals_only:
            bars(output)
    print("[prepared]", output, flush=True)


if __name__ == "__main__":
    main()
