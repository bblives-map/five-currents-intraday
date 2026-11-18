"""Create new regional price pairs from already-exported bar caches, without DB/API access."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import PriceBook, ROOTS, MICROS
from five_currents_subgroup_intraday import CLOCKS, HOLDS, DELAYS, entry_and_exits
from prepare_five_currents_execution_timing import DEFAULT_OUT, digest, save_json


OUTPUT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_subgroup_execution_20261024"


def prepare(source, output):
    output.mkdir(exist_ok=True, parents=True)
    if (output / "prices.npz").exists():
        raise FileExistsError("Price pairs are immutable; use the existing cache or a fresh output directory")
    sessions = pd.DatetimeIndex(pd.read_parquet(source / "sessions.parquet").session)
    targets = pd.read_parquet(source / "contract_targets.parquet")
    shape = (len(CLOCKS)*len(HOLDS), len(DELAYS), len(sessions), len(ROOTS))
    arrays = {name: np.full(shape, np.nan) for name in ("entry", "returns")}
    arrays["crosses_maintenance"] = np.zeros(shape, dtype=bool)
    records, sources = [], {}
    location = {date: i for i, date in enumerate(sessions)}
    for ai, root in enumerate(ROOTS):
        paths = list((source / "minute_cache").glob("*_"+root+".parquet"))
        paths += list((source / "minute_cache").glob("*_"+MICROS[root]+".parquet"))
        for path in paths:
            sources[str(path.relative_to(source))] = digest(path)
        book = PriceBook(pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True))
        latest = max((stamps[-1] for stamps, rows in book.groups.values() if len(stamps)), default=0)
        for target in targets.loc[targets.root.eq(root)].itertuples():
            day = location[target.session]
            for ci, clock in enumerate(CLOCKS):
                for oi, delay in enumerate(DELAYS):
                    for hi, row in enumerate(entry_and_exits(book, target, clock, delay)):
                        key = (ci*len(HOLDS)+hi, oi, day, ai)
                        if row["status"] == "matched":
                            arrays["entry"][key], arrays["returns"][key] = row["entry"], row["price_return"]
                            arrays["crosses_maintenance"][key] = row["crosses_maintenance"]
                        elif row["status"].startswith("missing") and row.get("planned_exit", row["planned_entry"]).value > latest:
                            row["status"] = "cache ends before requested fill"
                        records.append(row)
        print("[cached regional prices]", root, f"{len(records):,} cumulative price pairs", flush=True)
    np.savez_compressed(output / "prices.npz", **arrays)
    ledger = pd.DataFrame(records)
    ledger.to_parquet(output / "price_ledger.parquet", index=False)
    ledger.groupby(["root", "clock", "hold", "delay", "status"], dropna=False).size().rename("sessions").reset_index().to_csv(output / "coverage.csv", index=False)
    metadata = pd.DataFrame([{"cell": ci*len(HOLDS)+hi, "clock": clock, "hold": hold,
        "primary_clock": ci < 6} for ci, clock in enumerate(CLOCKS) for hi, hold in enumerate(HOLDS)])
    metadata.to_csv(output / "cells.csv", index=False)
    save_json(output / "price_manifest.json", {"source": str(source), "source_sha256": sources,
        "sessions": len(sessions), "start": str(sessions[0].date()), "end": str(sessions[-1].date()),
        "clocks": CLOCKS, "holds": HOLDS, "delays": DELAYS,
        "regional_reference_anchors": "Asia=Shanghai 09:00/15:00 (01:00/07:00 UTC), London=08:00/16:30 local, NY=09:30/16:00 local; reference-clock events, not a guarantee the reference cash exchange opens on a holiday",
        "exit_policy": "Hold starts at actual forward-only entry. Position may cross maintenance, but an entry or exit during scheduled closure is rejected. Same delivery month and price instrument throughout. At most +10 minutes fill tolerance.",
        "official_sources": ["https://www.shfe.com.cn/eng/services/investor/", "https://www.londonstockexchange.com/en/resources/faqs/personal-investing", "https://www.nyse.com/trade/trading-information", "https://www.cmegroup.com/education/courses/introduction-to-crude-oil/product-overview/wti-overview"],
        "calendar_limit": "Standard Globex maintenance/weekends enforced; special holiday closures need actual bars, never stale fills. Regional cash holidays are not excluded from reference-clock tests.",
        "market_api_calls": False, "database_access": False, "production_changes": False})
    print("[price preparation complete]", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        prepare(args.source.resolve(), args.output.resolve())
