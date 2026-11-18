"""Replay the deployed ten-rule intraday schedule from saved signals and minute bars."""
from __future__ import annotations

import base64
import hashlib
import json
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psycopg2 import sql
from threadpoolctl import threadpool_limits

from calculate_five_currents_intraday_positions import BOOK, SOURCE as RULE_SOURCE, planned_entry
from build_five_currents_frontmonth_prices import (
    FrontRouteBook, fill_pair, prior_volume_route, session_calendar, TABLES, FIELDS,
)
from five_currents_daily.curve_policy import CURVE_RANK_LIMITS
from five_currents_daily.db import readonly_connection
from five_currents_execution_timing import ROOTS, SLIPPAGE_PRICE_PER_SIDE, asset_pnl
from five_currents_front_intraday import MICROS, MICRO_MULTIPLIERS
from five_currents_group_clock_core import CLOCKS, HOLD_MINUTES, DELAYS, SOURCE, weights_for_class
from five_currents_opening_study import lag_contract_fields, rebuild_concepts_variants
from run_five_currents_group_clock import bundle


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
FRONT = ROOT / "five_currents_frontmonth_20261025"
OUT = ROOT / "five_currents_intraday_direction_audit_20261027"
COLORS = {"CL": "#b46140", "NG": "#56845b", "GC": "#ab862a", "HG": "#376b91"}
NOW = pd.Timestamp.now(tz="UTC")


def independent_weights(raw, dates, sessions, fi, treatment):
    """Explicit treatment of prior-close votes, independent of the production helper."""
    indices = dates.searchsorted(sessions, side="left") - 1
    assert (indices >= 1).all()
    assert (dates[indices].to_numpy() < sessions.to_numpy()).all()
    base, style = treatment.split("__")
    current, previous = raw[fi, indices], raw[fi, indices-1]
    if base == "gross":
        values = current.copy()
    elif base == "gross_change":
        values = current - previous
    else:
        raise ValueError(f"Unexpected deployed treatment: {treatment}")
    if style.endswith("sign"):
        values = np.where(np.abs(values) > 1e-12, np.sign(values)/len(ROOTS), 0.)
    if style.startswith("reverse_"):
        values = -values
    return values / np.maximum(1., np.sum(np.abs(values), axis=1))[:, None]


def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    active = x[np.abs(x) > 1e-14]
    equity = np.r_[1., np.cumprod(1+x)]
    std = x.std(ddof=1) if len(x) > 1 else np.nan
    without = x.copy()
    positives = np.flatnonzero(x > 0)
    if len(positives):
        without = np.delete(without, positives[np.argsort(x[positives])[-5:]])
    sd5 = without.std(ddof=1) if len(without) > 1 else np.nan
    return {"observed_days": len(x), "active_days": len(active),
            "net_sharpe": x.mean()/std*np.sqrt(252) if std > 0 else np.nan,
            "annual_vol": std*np.sqrt(252), "total_return": equity[-1]-1,
            "max_drawdown": np.min(equity/np.maximum.accumulate(equity)-1),
            "win_rate": np.mean(active > 0) if len(active) else np.nan,
            "median_gain": np.median(active[active > 0]) if (active > 0).any() else np.nan,
            "median_loss": np.median(active[active < 0]) if (active < 0).any() else np.nan,
            "worst_day": x.min() if len(x) else np.nan,
            "sr_without_top5": without.mean()/sd5*np.sqrt(252) if sd5 > 0 else np.nan}


def costs(weight, entry, root):
    return abs(weight)*(2.16/(entry*MICRO_MULTIPLIERS[root])
                        + 2*SLIPPAGE_PRICE_PER_SIDE[root]/entry)


def independent_clock(session, clock):
    zone, hour = {"asia_open": ("Asia/Singapore", "09:00"),
                  "asia_close": ("Asia/Singapore", "17:00"),
                  "london_open": ("Europe/London", "08:00"),
                  "london_close": ("Europe/London", "16:30"),
                  "us_open": ("America/New_York", "09:30")}[clock]
    return pd.Timestamp(str(pd.Timestamp(session).date())+" "+hour).tz_localize(zone).tz_convert("UTC")


def verify_minute_endpoints(ledger, archive_end, recent_frames):
    endpoints = []
    matched = ledger.loc[ledger.status.eq("matched")]
    for side in ("entry", "exit"):
        frame = matched[["root", "session", f"{side}_time", f"{side}_table",
                         f"{side}_price", "entry_contract"]].copy()
        frame = frame.rename(columns={f"{side}_time": "fill_time", f"{side}_table": "table",
                                      f"{side}_price": "price", "entry_contract": "expected_contract"})
        frame["bar_start"] = pd.to_datetime(frame.fill_time, utc=True)-pd.Timedelta(minutes=1)
        frame["archived"] = frame.session.le(archive_end)
        frame["month"] = frame.bar_start.dt.strftime("%Y%m")
        endpoints.append(frame)
    endpoints = pd.concat(endpoints, ignore_index=True).drop_duplicates(
        ["root", "table", "bar_start", "expected_contract", "price", "archived"])
    verified = 0
    for (root, table, month, archived), wanted in endpoints.groupby(["root", "table", "month", "archived"]):
        if archived:
            bars = pd.read_parquet(FRONT / "minute_cache" / root / f"{table}_{month}.parquet",
                                   columns=["ts", "close", "volume", "local_symbol"])
        else:
            bars = recent_frames[root].loc[recent_frames[root].table.eq(table),
                                           ["ts", "close", "volume", "local_symbol"]]
        checked = wanted.merge(bars, left_on="bar_start", right_on="ts", how="left", validate="many_to_one")
        assert checked.local_symbol.eq(checked.expected_contract).all(), (root, month, "contract mismatch")
        np.testing.assert_allclose(checked.close, checked.price, atol=1e-10, rtol=1e-12)
        assert pd.to_numeric(checked.volume).gt(0).all()
        verified += len(checked)
    print(f"[audit] Independently checked {verified:,} same-contract minute endpoints", flush=True)
    return verified


def load_recent_minutes(first, last):
    frames = {}
    with readonly_connection(statement_timeout_ms=30_000) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout='1s'")
            cursor.execute("SET max_parallel_workers_per_gather=0")
            for root in BOOK:
                parts = []
                for table in TABLES:
                    query = sql.SQL("SELECT ts,open,close,volume,bar_count,local_symbol,expiry "
                                    "FROM public.{} WHERE symbol=%s AND ts>=%s AND ts<%s ORDER BY ts").format(sql.Identifier(table))
                    cursor.execute(query, (MICROS[root], first.to_pydatetime(), last.to_pydatetime()))
                    frame = pd.DataFrame(cursor.fetchall(), columns=FIELDS)
                    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
                    frame["table"] = table
                    frame.to_parquet(OUT / f"recent_{root}_{table}.parquet", index=False)
                    parts.append(frame)
                frames[root] = pd.concat(parts, ignore_index=True)
                print(f"[minute read] {root}: {len(frames[root]):,} recent rows", flush=True)
    return frames


def plot_series(series, title, destination, *, panels=False):
    names = list(series)
    fig, axes = plt.subplots(len(names) if panels else 1, 1,
                             figsize=(13, 3.15*len(names) if panels else 5.2), squeeze=False)
    for i, (name, values) in enumerate(series.items()):
        ax = axes[i, 0] if panels else axes[0, 0]
        valid = values.notna()
        eq = (1+values.fillna(0)).cumprod()*100
        ax.plot(eq.index, eq.where(valid), label=name, color=COLORS.get(name, "#176b5c"), lw=1.45)
        ax.axhline(100, color="#b8bfb8", lw=.7)
        ax.grid(alpha=.18)
        ax.set_ylabel("Growth of 100")
        if panels:
            ax.set_title(name, loc="left", fontsize=12, weight="bold")
        else:
            ax.legend(frameon=False, ncol=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Completed audit exists; preserve it")
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(RULE_SOURCE / "recomputed_votes.npz") as saved:
        pinned = saved["gc4_lagged"]
        np.testing.assert_array_equal(saved["dates"], dates.to_numpy(dtype="datetime64[ns]"))
    with np.load(FRONT / "front_clock_prices_liquidity.npz") as saved:
        prices = {key: saved[key] for key in saved.files}
    candidates = pd.read_parquet(RULE_SOURCE / "individual_hindsight_winners.parquet")
    selected = []
    for root, families in BOOK.items():
        for i, family in enumerate(families):
            one = candidates.loc[candidates.root.eq(root) & candidates.family.eq(family)]
            assert len(one) == 1
            row = one.iloc[0].to_dict()
            row["book"] = "fixed" if i == 0 else "volume"
            selected.append(row)
    rules = pd.DataFrame(selected)
    rules.to_csv(OUT / "frozen_rules.csv", index=False)

    print("[audit] Reconstructing all daily votes from the latest saved panel", flush=True)
    replay_path = ROOT / "five_currents_dashboard/private_replay_cache.pkl"
    replay_bytes = replay_path.read_bytes()
    replay_digest = hashlib.sha256(replay_bytes).hexdigest()
    replay = pickle.loads(replay_bytes)
    fresh_dates = pd.DatetimeIndex(replay.combined.wide.index)
    lagged = lag_contract_fields(replay.combined.wide, replay.combined.mapping, replay.panel.metadata)
    vote_cache = OUT / "fresh_votes.npz"
    digest_file = OUT / "fresh_votes_source.sha256"
    if vote_cache.exists() and digest_file.exists() and digest_file.read_text() == replay_digest:
        with np.load(vote_cache) as saved:
            raw_full, raw_preview = saved["raw"], saved["preview"]
    else:
        with threadpool_limits(limits=2):
            base, gc4, gc5 = rebuild_concepts_variants(
                replay, replay, lagged, (CURVE_RANK_LIMITS,
                                        {**CURVE_RANK_LIMITS, "GC": 4},
                                        {**CURVE_RANK_LIMITS, "GC": 5}))
        raw_full, raw_preview = base.copy(), gc5.copy()
        gold = ROOTS.index("GC")
        raw_full[:, :, gold] = gc4[:, :, gold]
        raw_preview[:, :, gold] = gc4[:, :, gold]
        np.savez_compressed(vote_cache, raw=raw_full, preview=raw_preview,
                            dates=fresh_dates.to_numpy(dtype="datetime64[ns]"))
        digest_file.write_text(replay_digest)
    assert fresh_dates[:len(dates)].equals(dates)
    difference = np.abs(raw_full[:, :len(dates)] - pinned)
    changed = np.any(difference > 1e-10, axis=(0, 2))
    comparison = pd.DataFrame({"date": dates, "changed": changed,
                               "max_abs_difference": np.nanmax(difference, axis=(0, 2))})
    comparison.to_csv(OUT / "daily_vote_revision_audit.csv", index=False)
    print(f"[audit] Historical votes: {int(changed.sum())} revised dates / {len(dates)}", flush=True)

    next_session = fresh_dates[-1] + pd.offsets.BDay(1)
    extensions = fresh_dates[(fresh_dates > sessions[-1])]
    completed = sessions.union(extensions)
    all_sessions = completed.union(pd.DatetimeIndex([next_session]))
    calendar = session_calendar(all_sessions)
    recent_first = (sessions[-1] - pd.Timedelta(days=12)).tz_localize("UTC")
    minute_frames = load_recent_minutes(recent_first, NOW + pd.Timedelta(seconds=1))
    recent_books = {root: FrontRouteBook(frame) for root, frame in minute_frames.items()}
    recent_routes = {}
    for root in BOOK:
        for session in all_sessions[all_sessions > sessions[-1]]:
            prior = all_sessions[all_sessions < session][-5:]
            preferred, volume = prior_volume_route(recent_books[root], session, prior)
            recent_routes[root, session] = preferred

    ledger_parts, summary, archived_checks = [], [], []
    leg_paths, delay_paths, fresh_paths = {}, {}, {}
    weight_checks = []
    today_rows = []
    for rule in rules.itertuples(index=False):
        key = f"{rule.root}:{rule.family}"
        ai, fi = ROOTS.index(rule.root), names.index(rule.family)
        w = independent_weights(pinned, dates, sessions, fi, rule.signal_class)
        w_reference = weights_for_class(pinned, pinned, dates, sessions, family=fi,
                                        signal_class=rule.signal_class)
        np.testing.assert_allclose(w, w_reference, rtol=0, atol=1e-12)
        fw = independent_weights(raw_full, fresh_dates, all_sessions, fi, rule.signal_class)
        fresh_reference = weights_for_class(raw_full, raw_full, fresh_dates, all_sessions,
                                            family=fi, signal_class=rule.signal_class)
        np.testing.assert_allclose(fw, fresh_reference, rtol=0, atol=1e-12)
        ci, hi = CLOCKS.index(rule.clock), HOLD_MINUTES.index(rule.hold_minutes)
        idx = dates.searchsorted(sessions)-1
        leg_frames = []
        for path in sorted((FRONT / "fill_ledgers_liquidity" / rule.root).glob("*.parquet")):
            frame = pd.read_parquet(path)
            frame = frame.loc[frame.clock.eq(rule.clock)
                              & frame.hold_minutes.eq(rule.hold_minutes)
                              & frame.delay_minutes.isin([0, 15])].copy()
            leg_frames.append(frame)
        history_ledger = pd.concat(leg_frames, ignore_index=True)
        history_ledger["family"] = rule.family
        history_ledger["signal_class"] = rule.signal_class
        history_ledger["book"] = rule.book
        history_ledger["weight"] = history_ledger.session.map(pd.Series(w[:, ai], index=sessions))
        history_ledger["signal_date"] = history_ledger.session.map(pd.Series(dates[idx], index=sessions))
        assert (history_ledger.signal_date < history_ledger.session).all()
        expected_clock = history_ledger.apply(lambda row: independent_clock(row.session, rule.clock)
                                              + pd.Timedelta(minutes=int(row.delay_minutes)), axis=1)
        assert history_ledger.planned_entry.eq(expected_clock).all()
        matched = history_ledger.status.eq("matched")
        assert (history_ledger.loc[matched, "entry_time"] >= history_ledger.loc[matched, "planned_entry"]).all()
        assert (history_ledger.loc[matched, "exit_time"] > history_ledger.loc[matched, "entry_time"]).all()
        history_ledger["gross"] = history_ledger.weight*(history_ledger.exit_price/history_ledger.entry_price-1)
        history_ledger["cost"] = costs(history_ledger.weight, history_ledger.entry_price, rule.root)
        history_ledger.loc[~matched, ["gross", "cost"]] = np.nan
        inactive = history_ledger.weight.abs().le(1e-12)
        history_ledger.loc[inactive, ["gross", "cost"]] = 0.
        history_ledger["net"] = history_ledger.gross-history_ledger.cost
        extra_rows = []
        for session in all_sessions[all_sessions > sessions[-1]]:
            di = all_sessions.get_loc(session)
            weight = float(fw[di, ai])
            for delay in (0, 15):
                entry = fill_pair(recent_books[rule.root], session, rule.root, rule.clock,
                                  int(rule.hold_minutes), delay, calendar, recent_routes[rule.root, session])
                entry.update(family=rule.family, signal_class=rule.signal_class, book=rule.book,
                             weight=weight, signal_date=fresh_dates[fresh_dates.searchsorted(session)-1])
                valid = entry["status"] == "matched"
                entry["gross"] = weight*entry["price_return"] if valid else np.nan
                entry["cost"] = costs(weight, entry["entry_price"], rule.root) if valid else np.nan
                if abs(weight) <= 1e-12:
                    entry["gross"], entry["cost"] = 0., 0.
                entry["net"] = entry["gross"]-entry["cost"]
                extra_rows.append(entry)
                if session == next_session and delay == 0:
                    detail = dict(entry)
                    detail["planned_exit_nominal"] = detail["planned_entry"] + pd.Timedelta(minutes=int(rule.hold_minutes))
                    detail["status_now"] = ("flat" if abs(weight) <= 1e-12 else "not_started"
                                            if detail["planned_entry"] > NOW else "closed" if valid else "open")
                    if detail["status_now"] == "open" and "entry_price" in detail:
                        frame = minute_frames[rule.root]
                        tail = frame.loc[frame.local_symbol.eq(detail["entry_contract"])
                                         & frame.ts.ge(detail["entry_time"])
                                         & frame.ts.lt(NOW-pd.Timedelta(minutes=1))
                                         & pd.to_numeric(frame.volume, errors="coerce").gt(0)].sort_values("ts")
                        if len(tail):
                            mark = tail.iloc[-1]
                            detail.update(mark_price=float(mark.close), mark_time=mark.ts+pd.Timedelta(minutes=1))
                    elif valid:
                        detail.update(mark_price=entry["exit_price"], mark_time=entry["exit_time"])
                    if "mark_price" in detail and "entry_price" in detail and abs(weight) > 1e-12:
                        detail["gross_mtm"] = weight*(detail["mark_price"]/detail["entry_price"]-1)
                        detail["gross_usd_per_micro_direction"] = np.sign(weight)*(detail["mark_price"]-detail["entry_price"])*MICRO_MULTIPLIERS[rule.root]
                    today_rows.append(detail)

        combined = pd.concat([history_ledger, pd.DataFrame(extra_rows)], ignore_index=True)
        combined["hold_minutes"] = int(rule.hold_minutes)
        ledger_parts.append(combined)
        main = combined.loc[combined.delay_minutes.eq(0)].set_index("session").net.reindex(completed)
        delayed = combined.loc[combined.delay_minutes.eq(15)].set_index("session").net.reindex(completed)
        leg_paths[key], delay_paths[key] = main, delayed
        weights_fresh_hist = fw[:len(sessions), ai]
        moves = prices["returns"][ci, hi, 0, :, ai]
        entries = prices["entry"][ci, hi, 0, :, ai]
        reconstructed_net = w[:, ai]*moves-costs(w[:, ai], entries, rule.root)
        reconstructed_net[np.abs(w[:, ai]) <= 1e-12] = 0.
        np.testing.assert_allclose(main.reindex(sessions), reconstructed_net, rtol=0, atol=1e-12, equal_nan=True)
        gross_ref, fee_ref, slip_ref = asset_pnl(w, prices["returns"][ci, hi, 0], prices["entry"][ci, hi, 0])
        np.testing.assert_allclose(reconstructed_net, (gross_ref-fee_ref-slip_ref)[:, ai], atol=1e-12, equal_nan=True)
        archived_sr = stats(reconstructed_net)["net_sharpe"]
        np.testing.assert_allclose(archived_sr, rule.net_sr, atol=1e-9)
        fresh_net = weights_fresh_hist*moves-costs(weights_fresh_hist, entries, rule.root)
        fresh_net[np.abs(weights_fresh_hist) <= 1e-12] = 0.
        fresh_paths[key] = pd.Series(fresh_net, index=sessions)
        archived_checks.append({"root": rule.root, "family": rule.family,
                                "archived_sr": rule.net_sr, "replayed_sr": archived_sr,
                                "weight_error": float(np.max(np.abs(w-w_reference))),
                                "changed_historical_weight_dates": int(np.sum(np.abs(weights_fresh_hist-w[:,ai])>1e-10))})
        summary.append({"series": key, **stats(main), "planned_days": int(combined.loc[combined.delay_minutes.eq(0)
                        & combined.session.isin(completed), "weight"].abs().gt(1e-12).sum()),
                        "delay15_sharpe": stats(delayed)["net_sharpe"], "frozen_archive_sr": rule.net_sr,
                        "fresh_history_sr": stats(fresh_net)["net_sharpe"]})
        print(f"[replay] {key}: archived SR {archived_sr:.4f}, sign/clock/price checks passed", flush=True)

    ledger = pd.concat(ledger_parts, ignore_index=True)
    verified_endpoints = verify_minute_endpoints(ledger, sessions[-1], minute_frames)
    ledger.to_parquet(OUT / "full_trade_audit.parquet", index=False)
    pd.DataFrame(archived_checks).to_csv(OUT / "archived_reconciliation.csv", index=False)
    pd.DataFrame(today_rows).to_json(OUT / "today_legs.json", orient="records", date_format="iso", indent=2)
    paths = pd.DataFrame(leg_paths, index=completed)
    delay_frame = pd.DataFrame(delay_paths, index=completed)
    paths.to_parquet(OUT / "leg_daily_returns.parquet")
    books = {}
    for root in BOOK:
        keys = [key for key in paths if key.startswith(root+":")]
        books[root] = paths[keys].sum(axis=1, min_count=len(keys))
    fixed = [f"{root}:{families[0]}" for root, families in BOOK.items()]
    volume = [key for key in paths if key not in fixed]
    books["Portfolio"] = paths.sum(axis=1, min_count=len(paths.columns))
    books["Fixed four legs"] = paths[fixed].sum(axis=1, min_count=len(fixed))
    books["Volume six legs"] = paths[volume].sum(axis=1, min_count=len(volume))
    books["Original fixed-book scale"] = books["Fixed four legs"]/4
    book_frame = pd.DataFrame(books)
    book_frame.to_parquet(OUT / "portfolio_daily_returns.parquet")
    for key, values in books.items():
        columns = ([c for c in paths if c.startswith(key+":")] if key in BOOK else
                   list(paths) if key == "Portfolio" else fixed if key in ("Fixed four legs", "Original fixed-book scale") else volume)
        delayed = delay_frame[columns].sum(axis=1, min_count=len(columns))
        if key == "Original fixed-book scale":
            delayed /= 4
        summary.append({"series": key, **stats(values), "delay15_sharpe": stats(delayed)["net_sharpe"]})
    metrics = pd.DataFrame(summary)
    metrics.to_csv(OUT / "performance_metrics.csv", index=False)
    years = []
    for key, values in books.items():
        for year, part in values.groupby(values.index.year):
            years.append({"series": key, "year": year, **stats(part)})
    pd.DataFrame(years).to_csv(OUT / "yearly_metrics.csv", index=False)

    issuance = []
    today_checks = []
    with sqlite3.connect(ROOT / "five_currents_dashboard/publications.sqlite3") as connection:
        publications = connection.execute("SELECT id,issued_utc,admin_json FROM publications WHERE signal_date=? ORDER BY id",
                                          (str(fresh_dates[-1].date()),)).fetchall()
    for ordinal, issued, payload in publications:
        payload = json.loads(payload)
        schedule = payload.get("intraday_exposure", {})
        for row in payload.get("intraday_alpha_legs", []):
            ai, fi = ROOTS.index(row["root"]), names.index(row["family"])
            source = raw_preview if schedule.get("provisional") else raw_full
            weight = float(independent_weights(source, fresh_dates, pd.DatetimeIndex([next_session]), fi, row["signal_class"])[0, ai])
            error = weight-row["archived_normalized_weight"]
            today_checks.append({"publication": ordinal, "issued_utc": issued, "root": row["root"],
                                 "family": row["family"], "published_weight": row["archived_normalized_weight"],
                                 "replayed_weight": weight, "difference": error})
        for part in schedule.get("intervals", []):
            issuance.append({"publication": ordinal, "issued_utc": issued, **part})
    pd.DataFrame(today_checks).to_csv(OUT / "today_issued_weight_reconciliation.csv", index=False)
    pd.DataFrame(issuance).to_csv(OUT / "publication_intervals.csv", index=False)
    failures = [row for row in today_checks if abs(row["difference"]) > 1e-9]

    plot_series({"Portfolio": books["Portfolio"], "Fixed four legs": books["Fixed four legs"],
                 "Volume six legs": books["Volume six legs"]},
                "Deployed intraday rules | combined and component books", OUT / "portfolio_equity.png")
    plot_series({root: books[root] for root in BOOK},
                "Intraday performance by commodity | displayed exposure scale", OUT / "asset_equity.png", panels=True)

    cols = ["series", "observed_days", "net_sharpe", "total_return", "max_drawdown", "win_rate", "median_gain", "median_loss", "delay15_sharpe"]
    display = metrics.loc[metrics.series.isin(["Portfolio", *BOOK, "Fixed four legs", "Volume six legs"]), cols].copy()
    for column in ("total_return", "max_drawdown", "win_rate", "median_gain", "median_loss"):
        display[column] *= 100
    today = pd.DataFrame(today_rows)
    today_display = today[["root", "family", "weight", "planned_entry", "planned_exit_nominal", "status_now",
                           "entry_price", "mark_price", "mark_time", "gross_mtm", "gross_usd_per_micro_direction"]].copy()
    for col in ("weight", "gross_mtm"):
        today_display[col] *= 100
    for col in ("planned_entry", "planned_exit_nominal", "mark_time"):
        today_display[col] = pd.to_datetime(today_display[col], utc=True).dt.tz_convert("Asia/Singapore").astype(str)
    active = ledger.loc[ledger.weight.abs().gt(1e-12) & ledger.delay_minutes.eq(0) & ledger.session.isin(completed)]
    coverage = active.groupby("root").agg(planned=("session", "size"), matched=("net", "count"))
    coverage["coverage_pct"] = coverage.matched/coverage.planned*100
    coverage.to_csv(OUT / "coverage.csv")
    ledger.loc[ledger.weight.abs().gt(1e-12) & ledger.net.isna()].to_parquet(OUT / "missing_active_fills.parquet", index=False)
    timing = ledger.loc[ledger.status.eq("matched")].copy()
    for column in ("entry_time", "exit_time", "planned_entry"):
        timing[column] = pd.to_datetime(timing[column], utc=True)
    timing["entry_delay"] = (timing.entry_time-timing.planned_entry).dt.total_seconds()/60
    timing["hold_extra"] = (timing.exit_time-timing.entry_time).dt.total_seconds()/60-timing.hold_minutes
    timing.groupby("root")[["entry_delay", "hold_extra"]].agg(["median", "max"]).to_csv(OUT / "actual_fill_delays.csv")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "mark_cutoff_utc": NOW.isoformat(),
                "history_start": str(completed[0].date()), "history_end": str(completed[-1].date()),
                "archive_sessions": len(sessions), "completed_sessions": len(completed),
                "independently_verified_minute_endpoints": verified_endpoints,
                "fresh_daily_revision_dates": int(changed.sum()), "issued_weight_discrepancies": failures,
                "selection": "Existing same-history winners frozen; no reselection or tuning in this audit",
                "sizing": "Sum the ten published unit-sleeve weights; no dollar VaR sizing. Fees charged separately per leg, conservatively before netting.",
                "commission_micro_roundtrip_usd": 2.16, "database_access": "Read-only indexed recent one-minute bars, no broker API",
                "missing": "Per-asset performance uses its own observations. Portfolio unknown if any active component lacks a fill; missing outcomes excluded, no zero imputation.",
                "replay_cache_sha256": replay_digest}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    def image(name):
        return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Intraday Direction Audit</title><style>
    body{margin:0;background:#f5f2eb;color:#203c3b;font:15px/1.6 Georgia,serif}main{max-width:1250px;margin:40px auto;padding:0 28px 60px}h1{font-size:38px}h2{margin-top:38px}img{width:100%;background:white}table{border-collapse:collapse;background:white;font:11px/1.5 Verdana,sans-serif;width:100%}th,td{padding:8px;border-bottom:1px solid #dce0d7;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.note{padding:18px;border-left:4px solid #b18d47;background:#fffdf7}code{font-size:12px}</style></head><body><main>"""
    html += f'<h1>Intraday signals: direction, clocks and full replay</h1><p>{manifest["history_start"]} through {manifest["history_end"]}; today marked separately at {NOW}.</p>'
    html += '<div class="note">These ten individual rules were chosen with hindsight from the same history. Their combined displayed book was not previously validated as an allocated portfolio. This audit retains the exact rules, sums the displayed exposures, and charges $2.16 per micro contract round trip plus the archived slippage. Historical returns retain unknown active fills as gaps.</div>'
    html += '<h2>Reconciliation</h2><p>Independent signs and prior-close dates matched the archived weight helper. All ten archived rule Sharpes and minute-ledger returns were reproduced. '+f'Current issued-weight discrepancies: {len(failures)}. Revised historical daily-vote dates: {int(changed.sum())}.</p>'
    html += '<h2>Today\'s legs and model marks</h2><div class="scroll">'+today_display.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)+'</div>'
    html += '<h2>Portfolio and component books</h2>'+image('portfolio_equity.png')
    html += '<div class="scroll">'+display.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)+'</div>'
    html += '<h2>Each commodity</h2>'+image('asset_equity.png')
    html += '<h2>All deployed rules</h2><div class="scroll">'+rules[['root','family','signal_class','clock','hold_minutes','net_sr','priced_trade_coverage']].to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')+'</div>'
    html += '<h2>Priced active legs</h2>'+coverage.to_html(float_format=lambda x:f'{x:.2f}',border=0)
    html += '<p>Statistics exclude unresolved active trades. Flat days are retained. Actual fills use the first traded minute close within ten minutes of the nominal clock; exits retain the same contract. Entry fills are normally one minute after the labelled clock and exit fills can be later than the displayed nominal end. The chart uses signed model weights, with no calibrated lot allocation or intraday dollar-VaR claim.</p></main></body></html>'
    (OUT / "intraday_direction_audit.html").write_text(html,encoding="utf-8")
    print(display.to_string(index=False),flush=True)
    print('TODAY_GC',today_display.loc[today_display.root.eq('GC')].to_string(index=False),flush=True)
    print('ISSUED_DISCREPANCIES',len(failures),flush=True)
    print(OUT / "intraday_direction_audit.html",flush=True)


if __name__ == "__main__":
    main()
