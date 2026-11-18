"""Audit the frozen intraday rules against the latest completed daily replay.

This reads the dashboard cache and archived study; it never contacts a broker,
database, or dashboard worker. The output is model exposure, not an order.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_daily.curve_policy import CURVE_RANK_LIMITS
from five_currents_execution_timing import ROOTS, regional_time
from five_currents_group_clock_core import asia_time, weights_for_class
from five_currents_opening_study import lag_contract_fields, rebuild_concepts_variants


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
SOURCE = ROOT / "five_currents_frontmonth_20261025" / "alpha_audit"
REPLAY = ROOT / "five_currents_dashboard" / "private_replay_cache.pkl"
BOOK = {
    "CL": ("carry_oi_reversal", "aggregate_volume_carry", "volume_spike_reversion"),
    "NG": ("carry_oi_reversal", "volume_spike_reversion"),
    "GC": ("oi_direction_reversion", "volume_dryup_continuation", "volume_spike_reversion"),
    "HG": ("curve_quality_reversal", "volume_spike_reversion"),
}
SGT = ZoneInfo("Asia/Singapore")


def selected_vote(raw, family_index, asset_index, signal_class):
    base, style = signal_class.split("__")
    current = float(raw[family_index, -1, asset_index])
    previous = float(raw[family_index, -2, asset_index])
    unstyled = current if base == "gross" else current - previous
    styled = np.sign(unstyled) / len(ROOTS) if style.endswith("sign") else unstyled
    if style.startswith("reverse_"):
        styled = -styled
    return current, previous, unstyled, float(styled)


def calculate_rows(replay, session: pd.Timestamp, *, rank_limits=None) -> list[dict]:
    dates = pd.DatetimeIndex(replay.combined.wide.index)
    if dates[-1] >= session:
        raise ValueError(f"Latest daily signal {dates[-1]} does not precede {session}")
    lagged_wide = lag_contract_fields(replay.combined.wide, replay.combined.mapping,
                                     replay.panel.metadata)
    limits = dict(CURVE_RANK_LIMITS if rank_limits is None else rank_limits)
    gold_limits = {**limits, "GC": min(limits["GC"], 4)}
    with threadpool_limits(limits=2):
        baseline, gc_four = rebuild_concepts_variants(
            replay, replay, lagged_wide, (limits, gold_limits))
    raw = baseline.copy()
    raw[:, :, ROOTS.index("GC")] = gc_four[:, :, ROOTS.index("GC")]

    winners = pd.read_parquet(SOURCE / "individual_hindsight_winners.parquet")
    names = list(replay.concept_names)
    rows = []
    for root, families in BOOK.items():
        asset_index = ROOTS.index(root)
        for family in families:
            matches = winners.loc[winners.root.eq(root) & winners.family.eq(family)]
            if len(matches) != 1:
                raise AssertionError(f"Expected one frozen rule for {root}/{family}")
            winner = matches.iloc[0]
            family_index = names.index(family)
            current, previous, unstyled, styled = selected_vote(
                raw, family_index, asset_index, winner.signal_class)
            weight = weights_for_class(raw, raw, dates, pd.DatetimeIndex([session]),
                                       lag=0, family=family_index,
                                       signal_class=winner.signal_class)[0, asset_index]
            if not np.isfinite(weight):
                raise AssertionError(f"Non-finite exposure for {root}/{family}")
            if not np.isclose(np.sign(weight), np.sign(styled)):
                raise AssertionError(f"Style/sign mismatch for {root}/{family}")
            rows.append({"root": root, "family": family,
                         "book": "fixed" if family == families[0] else "volume",
                         "signal_class": winner.signal_class, "clock": winner.clock,
                         "hold_minutes": int(winner.hold_minutes),
                         "raw_friday": current, "raw_thursday_revised": previous,
                         "unstyled": unstyled, "styled": styled,
                         "archived_normalized_weight": float(weight),
                         "historical_rule_sr": float(winner.net_sr)})
    return rows


def planned_entry(session: pd.Timestamp, root: str, clock: str) -> pd.Timestamp:
    if clock == "asia_open":
        return asia_time(session, "09:00")
    if clock == "asia_close":
        return asia_time(session, "17:00")
    return regional_time(session, clock, root)


def active_intervals(rows: list[dict], session: pd.Timestamp) -> dict[str, list[tuple]]:
    events = defaultdict(lambda: defaultdict(float))
    for row in rows:
        weight = float(row["archived_normalized_weight"])
        if abs(weight) < 1e-12:
            continue
        start = planned_entry(session, row["root"], row["clock"]).tz_convert(SGT)
        end = start + pd.Timedelta(minutes=int(row["hold_minutes"]))
        events[row["root"]][start] += weight
        events[row["root"]][end] -= weight
    result = {}
    for root, schedule in events.items():
        weight = 0.0
        previous = None
        parts = []
        for stamp, change in sorted(schedule.items()):
            if previous is not None and stamp > previous and abs(weight) >= 1e-12:
                parts.append((previous, stamp, weight))
            weight += change
            previous = stamp
        if abs(weight) > 1e-10:
            raise AssertionError(f"Unclosed planned exposure for {root}")
        result[root] = parts
    return result


def build_exposure(replay, session: pd.Timestamp | None = None, *, rank_limits=None) -> tuple[dict, list[dict]]:
    signal_date = pd.Timestamp(replay.combined.wide.index[-1])
    if signal_date != pd.Timestamp(replay.last_signal_date):
        raise ValueError("Intraday signal inputs do not match the published daily close")
    session = pd.Timestamp(session) if session is not None else signal_date + pd.offsets.BDay(1)
    rows = calculate_rows(replay, session, rank_limits=rank_limits)
    intervals = active_intervals(rows, session)
    parts = [
        {"root": root, "start_sgt": start.isoformat(), "end_sgt": end.isoformat(),
         "weight": float(weight)}
        for root in BOOK for start, end, weight in intervals.get(root, [])
    ]
    return {
        "session": str(session.date()), "signal_date": str(signal_date.date()),
        "timezone": "Asia/Singapore", "assets": list(BOOK), "intervals": parts,
        "weight_basis": "Signed normalized model weight; not contract lots or a VaR-sized order",
        "execution_status": "Planned exposure, not confirmed fills",
    }, rows


def run(session: pd.Timestamp, output: Path):
    replay = pickle.loads(REPLAY.read_bytes())
    exposure, rows = build_exposure(replay, session)
    payload = {
        "session": str(session.date()), "signal_date": exposure["signal_date"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_replay": str(REPLAY), "frozen_rules": str(SOURCE),
        "exposure": exposure,
        "positions": rows,
        "warning": "Archived normalized weights are study exposures, not lot counts; "
                   "the ten-leg aggregate has not been backtested or sized to a VaR budget.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"[audit] {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=pd.Timestamp, default=pd.Timestamp("2026-10-26"))
    parser.add_argument("--output", type=Path, default=ROOT / "five_currents_frontmonth_20261025"
                        / "monday_20261026_signal_audit.json")
    args = parser.parse_args()
    run(args.session, args.output)
