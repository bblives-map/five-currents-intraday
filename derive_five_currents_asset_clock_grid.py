"""Export the same descriptive family/clock/hold grid for each contract alone."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import SOURCE, OUTPUT
from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_group_clock import bundle, prepare_candidates, TIMELINES


def run():
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(OUTPUT / "clock_prices.npz") as saved:
        cube = {field: saved[field] for field in saved.files}
    groups = []
    for ai, root in enumerate(ROOTS):
        for timeline in TIMELINES:
            _, _, rows = prepare_candidates(arrays, dates, sessions, cube, (ai,),
                timeline=timeline, group=root, names=names)
            groups.append(pd.DataFrame(rows))
            print(f"[asset grid] {root} / {timeline}: {len(rows):,} rows", flush=True)
    frame = pd.concat(groups, ignore_index=True)
    destination = OUTPUT / "asset_family_clock_grid.parquet"
    frame.to_parquet(destination, index=False)
    save_json(OUTPUT / "asset_grid_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "source_weight_sha256": digest(SOURCE / "weight_arrays.npz"),
        "price_cube_sha256": digest(OUTPUT / "clock_prices.npz"),
        "asset_grid_sha256": digest(destination), "assets": ROOTS,
        "rows": len(frame), "new_market_api_requests": False,
        "interpretation": "Descriptive hindsight grid only; not an independently forward-selected portfolio"})
    print(f"[complete] {len(frame):,} contract-level rule rows", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
