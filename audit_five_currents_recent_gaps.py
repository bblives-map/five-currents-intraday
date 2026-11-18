from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_daily.dashboard_core import json_bytes
from five_currents_daily.dashboard_store import PublicationStore
from five_currents_daily.reconciliation import MarkBook, canonical_contract
from us_commodity_research_core import PRODUCT_ORDER


def main():
    parser=argparse.ArgumentParser(description="Audit local replay repairs; no API or database collection.")
    parser.add_argument("--directory",type=Path,default=Path("five_currents_dashboard"))
    parser.add_argument("--old-publication",type=int,default=5)
    args=parser.parse_args()
    store=PublicationStore(args.directory)
    current=store.latest(admin=True)
    old_path=next((args.directory/"snapshots").glob(f"{args.old_publication:06d}_*/admin.json"))
    old=json.loads(old_path.read_text())
    before=dict(zip(old["history"]["dates"],old["history"]["returns"]))
    after=dict(zip(current["history"]["dates"],current["history"]["returns"]))
    common=[d for d,v in before.items() if d<="2026-07-16" and v is not None]
    assert all(after.get(d) is not None for d in common), "A repair introduced an archived-history gap"
    error=max(abs(before[d]-after[d]) for d in common)
    assert error<1e-10,"The frozen prefix changed"
    path=args.directory/"private_replay_cache.pkl"
    content=path.read_bytes()
    assert hashlib.sha256(content).hexdigest()==json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    replay=pickle.loads(content)
    book=MarkBook(replay.combined.valuation_marks,replay.weights.index)
    snapshot=args.directory/"snapshots"/f"{current['publication_id']:06d}_{current['signal_date']}"
    weights=pd.read_parquet(snapshot/"weights.parquet")
    entries=book.calendar.searchsorted(weights.index)+1
    unresolved=[]
    for day,value in after.items():
        if day<"2026-07-21" or value is not None:
            continue
        position=int(book.calendar.get_loc(pd.Timestamp(day)))
        forecast=int(np.searchsorted(entries,position-1,side="right")-1)
        failures=[]
        if forecast>=0 and position>0:
            for column,root in enumerate(PRODUCT_ORDER):
                if abs(weights.iloc[forecast,column])<1e-12:
                    continue
                contract=replay.execution.identity[forecast,column]
                key=canonical_contract(root,contract,pd.Timestamp(day))
                if not np.isfinite(book.return_between(root,key,book.calendar[position-1],book.calendar[position])):
                    failures.append({"product":root,"held_contract":str(contract),"delivery":key,
                                     "entry_date":str(book.calendar[position-1].date()),"mark_date":day})
        unresolved.append({"date":day,"missing_marks":failures})
    summed=pd.DataFrame({root:current["history"]["assets"][root]["returns"] for root in PRODUCT_ORDER}).sum(axis=1,min_count=5)
    returns=pd.Series(current["history"]["returns"],dtype=float)
    contribution_error=float((summed-returns).abs().max())
    assert contribution_error<1e-10
    result={"previous_publication":args.old_publication,"new_publication":current["publication_id"],
            "previous_metrics":old["history"]["metrics"],"new_metrics":current["history"]["metrics"],
            "signal_date":current["signal_date"],"performance_date":current["performance_date"],
            "frozen_prefix_return_maximum_error":error,"asset_contribution_maximum_error":contribution_error,
            "recovered_chart_dates":[d for d,v in before.items() if d>="2026-07-21" and v is None and after.get(d) is not None],
            "restored_calendar_dates":sorted(set(after)-set(before)),"unresolved_recent_marks":unresolved,
            "verification":current["verification"],"execution_ready":current["execution_ready"]}
    (args.directory/"gap_repair_audit.json").write_bytes(json_bytes(result))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
