"""Independent economic-book intraday research; immutable cached daily signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from five_currents_execution_timing import ROOTS, regional_time, asset_pnl
from run_five_currents_market_hours import closed_exit_reason, crosses_maintenance


GROUPS = {"energy": (0, 1), "precious": (2, 3), "copper": (4,)}
ANCHORS = ("asia_open", "asia_close", "london_open", "london_close", "us_open", "us_close")
CONTROLS = ("cme_open", "early_utc", "settlement")
CLOCKS = ANCHORS+CONTROLS
HOLDS = (30, 60, 90, 120, 180, 240, 360, 720)
DELAYS = (-10, -5, 0, 5, 15, 30)
FIT_DELAYS = (DELAYS.index(0), DELAYS.index(5), DELAYS.index(15))
BASE_DELAY = DELAYS.index(0)
COHORT_GROSS_CEILING = .5


def clock_time(session, clock, root="CL"):
    if clock.startswith("asia_"):
        # Shanghai commodity day-session anchors, same wall clock as Singapore.
        local = pd.Timestamp(session).normalize()+pd.Timedelta(hours=9 if clock == "asia_open" else 15)
        return local.tz_localize("Asia/Shanghai").tz_convert("UTC")
    return regional_time(session, clock, root)


def entry_and_exits(book, target, clock, delay):
    """Choose an entry route once, before examining any requested exit."""
    planned = clock_time(target.session, clock, target.root)+pd.Timedelta(minutes=delay)
    closed = closed_exit_reason(planned)
    entry = None if closed else book.find(target.root, target.month, planned)
    for hold in HOLDS:
        row = {"session": target.session, "root": target.root, "clock": clock, "hold": hold,
            "delay": delay, "planned_entry": planned, "month": target.month,
            "identity_basis": target.identity_basis, "rank": target.rank,
            "status": "closed entry" if closed else "missing entry", "crosses_maintenance": False}
        if entry is None:
            yield row
            continue
        row.update(entry=entry.price, entry_ts=entry.ts, symbol=entry.symbol)
        if closed_exit_reason(entry.ts):
            row["status"] = "delayed entry in closed market"
            yield row
            continue
        exit_stamp = entry.ts+pd.Timedelta(minutes=hold)
        row.update(planned_exit=exit_stamp, crosses_maintenance=crosses_maintenance(entry.ts, exit_stamp))
        reason = closed_exit_reason(exit_stamp)
        if reason:
            row["status"] = reason
            yield row
            continue
        last = book.find(target.root, target.month, exit_stamp, symbol=entry.symbol)
        if last is None:
            row["status"] = "missing same-instrument exit"
        elif closed_exit_reason(last.ts):
            row["status"] = "delayed exit in closed market"
        elif last.ts <= entry.ts:
            row["status"] = "nonpositive holding interval"
        else:
            row.update(status="matched", exit=last.price, exit_ts=last.ts,
                price_return=last.price/entry.price-1., elapsed_minutes=(last.ts-entry.ts).total_seconds()/60)
        yield row


def class_library(source, timeline, *, extra_lag=0, include_controls=True):
    """Differentiate original unnormalised daily values before applying risk caps."""
    held = timeline == "older_settled"
    suffix = "settled" if held else "lagged"
    saved = source.saved
    raw, allocated = saved["raw_"+suffix], saved["allocated_"+suffix]
    np.testing.assert_allclose(allocated.sum(axis=0), saved["net_"+suffix], atol=1e-12)
    names = source.manifest["signal_names"]
    series = [(name, basis, values[f]) for f, name in enumerate(names)
              for basis, values in (("raw_family", raw), ("allocated_family", allocated))]
    if include_controls:
        series.extend([("portfolio_net", "net_control", allocated.sum(axis=0)),
                       ("portfolio_gross", "unsigned_control", np.abs(allocated).sum(axis=0))])
    positions = source.dates.searchsorted(source.sessions, side="left")-1-int(held)-extra_lag
    if (positions < 2).any():
        raise ValueError("Insufficient daily history for the second difference")
    metadata, weights = [], []
    for name, basis, values in series:
        for order in (0, 1, 2):
            current = values[positions].copy()
            if order >= 1:
                current -= values[positions-1]
            if order == 2:
                current -= values[positions-1]-values[positions-2]
            for kind in ("magnitude", "sign"):
                amount = np.sign(current)/len(ROOTS) if kind == "sign" else current.copy()
                amount /= np.maximum(np.abs(amount).sum(axis=1), 1.)[:, None]
                for orientation in (1, -1):
                    metadata.append({"slot": len(metadata), "family": name, "basis": basis,
                        "difference_order": order, "kind": kind, "orientation": orientation,
                        "class": f"{basis}_d{order}_{kind}_"+("direct" if orientation == 1 else "inverse"),
                        "is_control": name.startswith("portfolio_")})
                    weights.append(amount*orientation)
    dates = source.dates[positions]
    assert (dates.to_numpy() < source.sessions.to_numpy()).all()
    return pd.DataFrame(metadata), np.stack(weights)


def pool_outcomes(weights, cube, indices, delay):
    """slot x cell x day; aggregate only the requested economic subgroup."""
    slots, days, _ = weights.shape
    cells = cube["returns"].shape[0]
    gross, cost = np.empty((slots, cells, days)), np.empty((slots, cells, days))
    for slot in range(slots):
        g, fee, slip = asset_pnl(COHORT_GROSS_CEILING*weights[slot][None],
                                cube["returns"][:, delay], cube["entry"][:, delay])
        good = np.isfinite(g[:, :, indices]).all(axis=2)
        gross[slot] = np.where(good, g[:, :, indices].sum(axis=2), np.nan)
        cost[slot] = np.where(good, (fee+slip)[:, :, indices].sum(axis=2), np.nan)
    return gross, cost


def replay_classes(weights, cube, choices, alpha, indices, *, delay=BASE_DELAY):
    """A sparse combination may contain several different classes of one family."""
    days, max_classes = choices.shape
    cells = cube["returns"].shape[0]
    positions = np.zeros((days, cells, len(ROOTS)))
    independent = positions.copy()
    for j in range(max_classes):
        d = np.flatnonzero((choices[:, j] >= 0) & (alpha[:, j] > 0))
        slot, cell = np.divmod(choices[d, j], cells)
        amount = weights[slot, d]*alpha[d, j, None]
        amount[:, [i for i in range(5) if i not in indices]] = 0
        for ai in indices:
            np.add.at(positions, (d, cell, np.full(len(d), ai)), amount[:, ai])
            np.add.at(independent, (d, cell, np.full(len(d), ai)), np.abs(amount[:, ai]))
    cap = np.maximum(np.abs(positions).sum(axis=(1, 2)), 1.)
    positions *= COHORT_GROSS_CEILING/cap[:, None, None]
    independent *= COHORT_GROSS_CEILING/cap[:, None, None]
    moves = cube["returns"][:, delay].transpose(1, 0, 2)
    entries = cube["entry"][:, delay].transpose(1, 0, 2)
    g, fee, slip = asset_pnl(positions, moves, entries)
    _, ff, ss = asset_pnl(independent, moves, entries)
    gross, cost = g.sum(axis=1), (fee+slip).sum(axis=1)
    return {"net": (gross-cost).sum(axis=1), "gross": gross.sum(axis=1),
            "cost": cost.sum(axis=1), "stressed": (gross-2*cost).sum(axis=1),
            "independent_net": gross.sum(axis=1)-(ff+ss).sum(axis=(1, 2)),
            "asset_net": gross-cost, "exposure": np.abs(positions).sum(axis=(1, 2)),
            "positions": positions}
