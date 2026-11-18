"""Read-only endpoint-coverage sensitivity for archived market-clock tests."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from psycopg2 import sql

from five_currents_daily.db import readonly_connection
from five_currents_front_intraday import MICROS, observed, same_intraday_instrument, valid_price
from run_five_currents_market_hours import closed_exit_reason


SOURCE = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_market_hours_20261013_reconciled"
ROUTES = {"front": "bars_1m", "nearby_proxy": "bars_1m_2m"}
FIELDS = ("ts", "open", "close", "volume", "bar_count", "symbol", "local_symbol", "expiry")


def read_missing_windows(table: str, moves: pd.DataFrame) -> pd.DataFrame:
    targets: dict[str, set[pd.Timestamp]] = defaultdict(set)
    for row in moves.loc[moves.status.isin(("missing entry", "missing exit"))].itertuples():
        stamps = (row.entry_utc, row.exit_utc)
        for symbol in (row.root, MICROS[row.root]):
            targets[symbol].update(stamps)

    query = sql.SQL(
        "SELECT ts,open,close,volume,bar_count,symbol,local_symbol,expiry "
        "FROM public.{} WHERE symbol=%s AND ts=ANY(%s::timestamptz[])"
    ).format(sql.Identifier(table))
    rows = []
    with readonly_connection(statement_timeout_ms=10000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET max_parallel_workers_per_gather=0")
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET application_name='market_clock_tolerance_readonly'")
            for symbol, symbol_targets in sorted(targets.items()):
                stamps = sorted({target - pd.Timedelta(minutes=minute)
                                 for target in symbol_targets for minute in range(-10, 11)})
                found = 0
                for offset in range(0, len(stamps), 384):
                    cursor.execute(query, (symbol, [s.to_pydatetime() for s in stamps[offset:offset + 384]]))
                    batch = cursor.fetchall()
                    rows.extend(dict(zip(FIELDS, values)) for values in batch)
                    found += len(batch)
                print(f"{table}/{symbol}: {len(symbol_targets)} missing endpoints, {found} nearby bars", flush=True)
    result = pd.DataFrame(rows, columns=FIELDS)
    result["ts"] = pd.to_datetime(result.ts, utc=True)
    return result


def bar_lookup(bars: pd.DataFrame):
    by_symbol = {}
    for symbol, group in bars.groupby("symbol"):
        records = [row for row in group.sort_values("ts").itertuples() if valid_price(row)]
        by_symbol[symbol] = ([row.ts for row in records], records)

    @lru_cache(maxsize=None)
    def eligible(symbol: str, stamp: pd.Timestamp, minutes: int, direction: str):
        if symbol not in by_symbol:
            return ()
        stamps, records = by_symbol[symbol]
        start = bisect_left(stamps, stamp - pd.Timedelta(minutes=minutes))
        end = bisect_right(stamps, stamp if direction == "prior" else stamp + pd.Timedelta(minutes=minutes))
        nearby = records[start:end]
        if direction == "prior":
            return tuple(reversed(nearby))
        return tuple(sorted(nearby, key=lambda row: (abs(row.ts - stamp), row.ts)))

    return eligible


def matched_row(row, eligible, minutes: int, direction: str):
    if row.status == "matched":
        return True, bool(row.entry_trade_confirmed and row.exit_trade_confirmed)
    if row.status not in ("missing entry", "missing exit"):
        return False, False
    if closed_exit_reason(row.exit_utc):
        return False, False
    entries = [*eligible(row.root, row.entry_utc, minutes, direction),
               *eligible(MICROS[row.root], row.entry_utc, minutes, direction)]
    if not entries:
        return False, False
    # A time-local choice, never selected by looking at the exit price.
    if direction == "prior":
        entries.sort(key=lambda candidate: candidate.ts, reverse=True)
    else:
        entries.sort(key=lambda candidate: (abs(candidate.ts - row.entry_utc), candidate.ts))
    entry = entries[0]
    exits = eligible(entry.symbol, row.exit_utc, minutes, direction)
    for candidate in exits:
        if same_intraday_instrument(entry, candidate):
            return True, observed(entry) and observed(candidate)
    return False, False


def run() -> None:
    output = []
    matched_sets = {}
    for route, table in ROUTES.items():
        moves = pd.read_parquet(SOURCE / f"{route}_price_moves.parquet")
        anchors = pd.read_parquet(SOURCE / f"{route}_minute_anchors.parquet")
        windows = read_missing_windows(table, moves)
        bars = pd.concat([anchors, windows], ignore_index=True).drop_duplicates(["symbol", "ts"])
        eligible = bar_lookup(bars)
        for (clock, root, hours), group in moves.groupby(["clock", "root", "hours"], sort=False):
            for direction, minutes in (("exact", 0), ("prior", 5), ("prior", 10), ("nearest", 5), ("nearest", 10)):
                if direction == "exact":
                    flags = [(row.status == "matched",
                              row.status == "matched" and bool(row.entry_trade_confirmed and row.exit_trade_confirmed))
                             for row in group.itertuples()]
                else:
                    flags = [matched_row(row, eligible, minutes, direction) for row in group.itertuples()]
                rule = f"{direction}{minutes}" if minutes else "exact"
                matched_sets[(route, clock, root, hours, rule)] = {
                    session for session, flag in zip(group.session, flags) if flag[0]
                }
                output.append({"route": route, "clock": clock, "root": root, "hours": hours,
                               "rule": rule,
                               "price_pairs": sum(flag[0] for flag in flags),
                               "trade_confirmed_pairs": sum(flag[1] for flag in flags),
                               "sessions": len(flags)})
    frame = pd.DataFrame(output)
    for route in ROUTES:
        print(f"\n{route}: 4-hour SIL coverage (out of 602):")
        print(frame.loc[(frame.route == route) & frame.root.eq("SI") & frame.hours.eq(4)]
              .pivot(index="clock", columns="rule", values="price_pairs").to_string())
        print(f"\n{route}: 4-hour SIL trade-confirmed pairs:")
        print(frame.loc[(frame.route == route) & frame.root.eq("SI") & frame.hours.eq(4)]
              .pivot(index="clock", columns="rule", values="trade_confirmed_pairs").to_string())
        print(f"\n{route}: 4-hour all-five same-session overlap:")
        for clock in ("london_open", "london_close", "us_open"):
            print(clock, {rule: len(set.intersection(*(matched_sets[(route, clock, root, 4, rule)]
                                                     for root in ("CL", "NG", "GC", "SI", "HG"))) )
                          for rule in ("exact", "prior5", "prior10", "nearest5", "nearest10")})
    print("\nAll horizons, silver only:")
    print(frame.loc[frame.root.eq("SI"), ["route", "clock", "hours", "rule", "price_pairs", "trade_confirmed_pairs"]]
          .sort_values(["route", "clock", "hours", "rule"]).to_string(index=False))


if __name__ == "__main__":
    run()
