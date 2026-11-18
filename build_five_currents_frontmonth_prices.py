"""Read-only, full-minute front-contract fills for the four-asset intraday study.

Daily signals are untouched. Entry uses rank one or, under a separately audited
causal roll policy, rank two when it was more liquid in completed prior sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2 import sql

from five_currents_daily.db import readonly_connection
from five_currents_execution_timing import regional_time, within_session
from five_currents_front_intraday import MICROS, ROOTS
from five_currents_group_clock_core import CLOCKS, DELAYS, HOLD_MINUTES, SOURCE, asia_time


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUT = ROOT / "five_currents_frontmonth_20261025"
PRODUCTS = ("CL", "NG", "GC", "HG")
TABLES = ("bars_1m", "bars_1m_2m")
MAX_DELAY_MINUTES = 10
ROLL_BUFFER_SESSIONS = 2
FETCH_PAUSE_SECONDS = 0.15
FIELDS = ("ts", "open", "close", "volume", "bar_count", "local_symbol", "expiry")
POLICIES = ("expiry_only", "prior_5_session_volume")


def session_calendar(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    observed = pd.DatetimeIndex(sessions).normalize().unique().sort_values()
    before = pd.bdate_range(observed.min() - pd.Timedelta(days=500),
                            observed.min() - pd.Timedelta(days=1))
    after = pd.bdate_range(observed.max() + pd.Timedelta(days=1),
                           observed.max() + pd.Timedelta(days=500))
    return before.union(observed).union(after).sort_values()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def chunk_path(output: Path, root: str, table: str, month: pd.Period) -> Path:
    return output / "minute_cache" / root / f"{table}_{month.strftime('%Y%m')}.parquet"


def download(output: Path, sessions: pd.DatetimeIndex) -> None:
    """Indexed monthly reads avoid loading a full table or touching collectors."""
    months = pd.period_range(sessions.min().to_period("M"),
                             sessions.max().to_period("M"), freq="M")
    ledger = []
    with readonly_connection(statement_timeout_ms=30000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET application_name='five_currents_frontmonth_cached_readonly'")
            for root in PRODUCTS:
                symbol = MICROS[root]
                for table in TABLES:
                    query = sql.SQL(
                        "SELECT ts,open,close,volume,bar_count,local_symbol,expiry "
                        "FROM public.{} WHERE symbol=%s AND ts>=%s AND ts<%s ORDER BY ts"
                    ).format(sql.Identifier(table))
                    for month in months:
                        path = chunk_path(output, root, table, month)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if not path.exists():
                            begin = month.start_time.tz_localize("UTC")
                            end = (month + 1).start_time.tz_localize("UTC")
                            cursor.execute(query, (symbol, begin.to_pydatetime(),
                                                   end.to_pydatetime()))
                            frame = pd.DataFrame(cursor.fetchall(), columns=FIELDS)
                            frame["ts"] = pd.to_datetime(frame.ts, utc=True)
                            if frame.ts.duplicated().any():
                                raise AssertionError(f"Duplicate {symbol} {table} {month} minute")
                            frame.to_parquet(path, index=False, compression="zstd")
                            time.sleep(FETCH_PAUSE_SECONDS)
                        else:
                            frame = pd.read_parquet(path, columns=["ts", "local_symbol"])
                        ledger.append({"root": root, "symbol": symbol, "table": table,
                                       "utc_month": str(month), "rows": len(frame),
                                       "path": str(path), "sha256": digest(path)})
                        print(f"[cached] {root} {table} {month}: {len(frame):,} full 1m rows",
                              flush=True)
    pd.DataFrame(ledger).to_parquet(output / "cache_ledger.parquet", index=False)


class FrontRouteBook:
    def __init__(self, frame: pd.DataFrame):
        self.routes = {}
        for table, part in frame.groupby("table", sort=False):
            part = part.sort_values("ts", kind="stable")
            self.routes[table] = {
                "ts": pd.to_datetime(part.ts, utc=True).dt.as_unit("ns").astype("int64").to_numpy(),
                "close": pd.to_numeric(part.close, errors="coerce").to_numpy(dtype=float),
                "volume": pd.to_numeric(part.volume, errors="coerce").to_numpy(dtype=float),
                "contract": part.local_symbol.fillna("").astype(str).to_numpy(),
                "expiry": part.expiry.to_numpy(),
            }

    def find(self, table: str, stamp: pd.Timestamp, contract: str | None = None):
        route = self.routes.get(table)
        if route is None:
            return None
        wanted = pd.Timestamp(stamp).value
        limit = wanted + pd.Timedelta(minutes=MAX_DELAY_MINUTES).value
        first = int(np.searchsorted(route["ts"], wanted, side="left"))
        end = int(np.searchsorted(route["ts"], limit, side="right"))
        for i in range(first, end):
            if not np.isfinite(route["volume"][i]) or route["volume"][i] <= 0:
                continue
            if not np.isfinite(route["close"][i]) or route["close"][i] <= 0:
                continue
            if not route["contract"][i]:
                continue
            if contract is not None and route["contract"][i] != contract:
                continue
            bar_start = pd.Timestamp(route["ts"][i], tz="UTC")
            return {"bar_start": bar_start, "fill_time": bar_start + pd.Timedelta(minutes=1),
                    "price": float(route["close"][i]), "contract": route["contract"][i],
                    "expiry": route["expiry"][i], "table": table}
        return None

    def find_same_contract(self, stamp: pd.Timestamp, contract: str):
        choices = [self.find(table, stamp, contract) for table in TABLES]
        choices = [choice for choice in choices if choice is not None]
        return min(choices, key=lambda row: row["bar_start"]) if choices else None

    def volume_between(self, table: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
        route = self.routes.get(table)
        if route is None:
            return 0.0
        left = int(np.searchsorted(route["ts"], pd.Timestamp(start).value, side="left"))
        right = int(np.searchsorted(route["ts"], pd.Timestamp(end).value, side="left"))
        values = route["volume"][left:right]
        return float(np.nansum(np.maximum(values, 0)))


def prior_volume_route(book: FrontRouteBook, session: pd.Timestamp,
                       prior_sessions: pd.DatetimeIndex):
    volumes = {}
    for table in TABLES:
        volumes[table] = sum(
            book.volume_between(table, regional_time(prior, "cme_open"),
                                regional_time(prior, "session_close") + pd.Timedelta(minutes=5))
            for prior in prior_sessions
        )
    first, second = (volumes[table] for table in TABLES)
    return ("bars_1m_2m" if second > first and second > 0 else "bars_1m", volumes)


def sessions_until_expiry(session: pd.Timestamp, raw_expiry, calendar: pd.DatetimeIndex):
    if pd.isna(raw_expiry):
        return None
    try:
        expiry = pd.to_datetime(str(int(raw_expiry)), format="%Y%m%d")
    except (ValueError, TypeError):
        return None
    return int(calendar.searchsorted(expiry) - calendar.searchsorted(session))


def fill_pair(book: FrontRouteBook, session: pd.Timestamp, root: str, clock: str,
              hold: int, delay: int, calendar: pd.DatetimeIndex,
              preferred_table: str = "bars_1m"):
    planned = (asia_time(session, "09:00" if clock == "asia_open" else "17:00")
               if clock.startswith("asia_") else regional_time(session, clock, root))
    planned += pd.Timedelta(minutes=delay)
    row = {"session": session, "root": root, "clock": clock, "hold_minutes": hold,
           "delay_minutes": delay, "planned_entry": planned, "status": "missing front entry"}
    if not within_session(planned, session):
        row["status"] = "entry outside session"
        return row
    first = book.find("bars_1m", planned)
    second = book.find("bars_1m_2m", planned)
    if first is None and preferred_table == "bars_1m":
        return row
    to_expiry = sessions_until_expiry(session, first["expiry"], calendar) if first else None
    roll_due = to_expiry is not None and to_expiry <= ROLL_BUFFER_SESSIONS
    if preferred_table == "bars_1m_2m" or roll_due:
        if second is None or (first is not None and pd.notna(second["expiry"])
                              and pd.notna(first["expiry"])
                              and int(second["expiry"]) <= int(first["expiry"])):
            row["status"] = "no safe second-month roll entry"
            return row
        entry = second
        row["roll_bridge"] = True
    else:
        entry = first
        row["roll_bridge"] = False
    row.update(entry_time=entry["fill_time"], entry_price=entry["price"],
               entry_contract=entry["contract"], entry_table=entry["table"],
               entry_delay_minutes=(entry["fill_time"] - planned).total_seconds() / 60,
               sessions_to_expiry=to_expiry)
    planned_exit = entry["fill_time"] + pd.Timedelta(minutes=hold)
    row["planned_exit"] = planned_exit
    if not within_session(planned_exit, session):
        row["status"] = "exit outside session"
        return row
    exit_bar = book.find_same_contract(planned_exit, entry["contract"])
    if exit_bar is None or not within_session(exit_bar["fill_time"], session):
        row["status"] = "missing same-contract exit"
        return row
    row.update(status="matched", exit_time=exit_bar["fill_time"],
               exit_price=exit_bar["price"], exit_table=exit_bar["table"],
               exit_delay_minutes=(exit_bar["fill_time"] - planned_exit).total_seconds() / 60,
               price_return=exit_bar["price"] / entry["price"] - 1)
    return row


def load_month_neighbours(output: Path, root: str, month: pd.Period):
    frames = []
    for table in TABLES:
        for offset in (-1, 0, 1):
            path = chunk_path(output, root, table, month + offset)
            if path.exists():
                frame = pd.read_parquet(path)
                if len(frame):
                    frame["table"] = table
                    frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No cached bars for {root} {month}")
    return FrontRouteBook(pd.concat(frames, ignore_index=True))


def build_prices(output: Path, sessions: pd.DatetimeIndex, policy: str = "expiry_only"):
    if policy not in POLICIES:
        raise ValueError(policy)
    suffix = "_liquidity" if policy == "prior_5_session_volume" else ""
    shape = (len(CLOCKS), len(HOLD_MINUTES), len(DELAYS), len(sessions), len(ROOTS))
    arrays = {name: np.full(shape, np.nan, dtype=float) for name in ("returns", "entry")}
    calendar = session_calendar(sessions)
    positions = {stamp: i for i, stamp in enumerate(sessions)}
    route_ledger = []
    for root in PRODUCTS:
        ai = ROOTS.index(root)
        for month in pd.period_range(sessions.min().to_period("M"),
                                     sessions.max().to_period("M"), freq="M"):
            book = load_month_neighbours(output, root, month)
            ledger = []
            active_sessions = sessions[sessions.to_period("M") == month]
            for session in active_sessions:
                di = positions[session]
                prior_sessions = sessions[max(0, di - 5):di]
                if policy == "prior_5_session_volume":
                    preferred_table, volumes = prior_volume_route(book, session, prior_sessions)
                else:
                    preferred_table = "bars_1m"
                    volumes = {table: np.nan for table in TABLES}
                route_ledger.append({"session": session, "root": root, "policy": policy,
                                     "preferred_table": preferred_table,
                                     "prior_sessions": len(prior_sessions),
                                     "latest_information_session": (prior_sessions[-1]
                                                                     if len(prior_sessions) else pd.NaT),
                                     "rank1_prior_volume": volumes["bars_1m"],
                                     "rank2_prior_volume": volumes["bars_1m_2m"]})
                for ci, clock in enumerate(CLOCKS):
                    for hi, hold in enumerate(HOLD_MINUTES):
                        for oi, delay in enumerate(DELAYS):
                            row = fill_pair(book, session, root, clock, int(hold), int(delay),
                                            calendar, preferred_table)
                            ledger.append(row)
                            if row["status"] == "matched":
                                arrays["returns"][ci, hi, oi, di, ai] = row["price_return"]
                                arrays["entry"][ci, hi, oi, di, ai] = row["entry_price"]
            path = output / f"fill_ledgers{suffix}" / root / f"{month.strftime('%Y%m')}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(ledger)
            frame.to_parquet(path, index=False, compression="zstd")
            print(f"[fills] {root} {month}: {frame.status.eq('matched').sum():,}/"
                  f"{len(frame):,} clock/hold/delay cells", flush=True)
    np.savez_compressed(output / f"front_clock_prices{suffix}.npz", **arrays)
    routes = pd.DataFrame(route_ledger)
    if not routes.empty and not (routes.latest_information_session.dropna() <
                                 routes.loc[routes.latest_information_session.notna(), "session"]).all():
        raise AssertionError("Route selection used information from the current or a future session")
    routes.to_parquet(output / f"route_selection{suffix}.parquet", index=False)
    return arrays


def verify(output: Path, sessions: pd.DatetimeIndex, arrays, policy: str = "expiry_only"):
    suffix = "_liquidity" if policy == "prior_5_session_volume" else ""
    if not np.isfinite(arrays["returns"][:, :, 0, :, :]).any():
        raise AssertionError("No matched front-contract fills")
    stats = []
    for ai, root in enumerate(ROOTS):
        if root not in PRODUCTS:
            continue
        for ci, clock in enumerate(CLOCKS):
            for hi, hold in enumerate(HOLD_MINUTES):
                values = arrays["returns"][ci, hi, 0, :, ai]
                stats.append({"root": root, "clock": clock, "hold_minutes": hold,
                              "matched_sessions": int(np.isfinite(values).sum()),
                              "sessions": len(sessions),
                              "coverage": float(np.isfinite(values).mean())})
    pd.DataFrame(stats).to_parquet(output / f"clock_coverage{suffix}.parquet", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": len(sessions), "session_start": str(sessions.min().date()),
        "session_end": str(sessions.max().date()), "products": PRODUCTS,
        "source_tables": TABLES, "symbols": {root: MICROS[root] for root in PRODUCTS},
        "database_access": "read-only indexed monthly SELECT, no API calls or writes",
        "price_cache": "full one-minute rows, not five-minute timestamp samples",
        "front_rule": ("rank one with rank two only within two observed sessions of expiry"
                       if policy == "expiry_only" else
                       "rank one or two selected from completed five-session volume; rank two also required within two observed sessions of rank-one expiry"),
        "policy": policy,
        "fill_rule": "first positive-volume minute within ten minutes; its close is an execution-price proxy timestamped at minute end; hold counted from that timestamp",
        "signal_source": str(SOURCE / "weight_arrays.npz"),
        "source_sha256": digest(SOURCE / "weight_arrays.npz"),
        "price_sha256": digest(output / f"front_clock_prices{suffix}.npz"),
    }
    (output / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--policy", choices=POLICIES, default="expiry_only")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sessions = pd.DatetimeIndex(pd.read_parquet(SOURCE / "sessions.parquet").session)
    if not args.skip_download:
        download(output, sessions)
    if args.download_only:
        return
    arrays = build_prices(output, sessions, args.policy)
    verify(output, sessions, arrays, args.policy)


if __name__ == "__main__":
    main()
