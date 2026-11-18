"""Validate frozen inputs, causal cutoffs, replay equivalence and report assets."""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_five_currents_execution_timing import DEFAULT_OUT,digest,save_json


def verify(output):
    source=json.loads((output / "source_manifest.json").read_text())
    assert digest(output / "inputs/daily_replay.pkl")==source["source_cache_sha256"]
    for name,sha in source["files_sha256"].items():assert digest(output / name)==sha
    minute=json.loads((output / "minute_cache_manifest.json").read_text())
    for row in minute["sources"]:
        assert digest(output / "minute_cache" / (row["table"]+"_"+row["symbol"]+".parquet"))==row["sha256"]
    audit=json.loads((output / "arithmetic_audit.json").read_text())
    assert audit["passed"] and audit["maximum_absolute_error"]<1e-10
    assert audit["daily_weights_reconcile_to_archived_dashboard"]<1e-10
    checked_pairs=0
    for path in (output / "price_ledgers").glob("*.parquet"):
        frame=pd.read_parquet(path,columns=["status","entry_ts","planned_entry","exit_ts","planned_exit"])
        frame=frame.loc[frame.status.eq("matched")]
        assert frame.entry_ts.ge(frame.planned_entry).all()
        assert frame.exit_ts.ge(frame.planned_exit).all()
        assert frame.exit_ts.gt(frame.entry_ts).all()
        assert (frame.entry_ts-frame.planned_entry).le(pd.Timedelta(minutes=10)).all()
        assert (frame.exit_ts-frame.planned_exit).le(pd.Timedelta(minutes=10)).all()
        checked_pairs+=len(frame)
    for name in ("timing_selection_ledger.csv","regional_block_selection_ledger.csv"):
        ledger=pd.read_csv(output / name)
        assert (ledger.train_end_exclusive<ledger.start).all()
    equivalence=[]
    for baseline,stress in (("walk_forward_metrics.csv","frozen_schedule_stress.csv"),
                           ("regional_block_causal_metrics.csv","regional_frozen_schedule_stress.csv")):
        original=pd.read_csv(output / baseline)
        delayed=pd.read_csv(output / stress)
        assert delayed.same_recipes_not_retuned.eq(True).all()
        original=original[["signal","book","fit","sharpe"]]
        zero=delayed.loc[delayed.lag.eq(0)&delayed.offset.eq(0)]
        joined=original.merge(zero,on=["signal","book","fit"],suffixes=("_original","_zero"),validate="one_to_one")
        np.testing.assert_allclose(joined.sharpe_original,joined.sharpe_zero,atol=1e-10,equal_nan=True)
        equivalence.append({"source":baseline,"comparisons":len(joined),"passed":True})
    path=output / "Five_Currents_Execution_Timing_Report.html"
    content=path.read_text(encoding="utf-8")
    images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',content)
    assert len(images)>=10
    for encoded in images:assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
    references=re.findall(r'href="([^"]+)"',content)
    for reference in references:
        if "://" not in reference and not reference.startswith("#"):
            assert (output / reference).is_file(),reference
    assert len(content)>100000
    result={"passed":True,"verified_utc":datetime.now(timezone.utc).isoformat(),
        "frozen_input_hashes_valid":True,"forward_only_matched_price_pairs_checked":checked_pairs,
        "replay_equivalence":equivalence,"embedded_pngs":len(images),"report_bytes":path.stat().st_size,
        "report_sha256":digest(path),"source_snapshot":source["source_snapshot"],
        "scalar_pnl_comparisons":audit["scalar_comparisons"],
        "runtime":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__},
        "reproduction_order":["prepare_five_currents_execution_timing.py (fresh output directory)",
            "run_five_currents_execution_timing_study.py","analyse_five_currents_execution_timing.py --reference-grid",
            "run_five_currents_regional_blocks.py","stress_five_currents_execution_timing.py",
            "diagnose_five_currents_regional_timing.py","report_five_currents_execution_timing.py",
            "verify_five_currents_execution_timing.py"],
        "code_sha256":{name:digest(Path(__file__).parent / name) for name in (
            "prepare_five_currents_execution_timing.py","five_currents_execution_timing.py",
            "run_five_currents_execution_timing_study.py","analyse_five_currents_execution_timing.py",
            "run_five_currents_regional_blocks.py","stress_five_currents_execution_timing.py",
            "diagnose_five_currents_regional_timing.py","report_five_currents_execution_timing.py",
            "test_five_currents_execution_timing.py","verify_five_currents_execution_timing.py")},
        "existing_dependency_sha256":{name:digest(Path(__file__).parent / name) for name in (
            "five_currents_front_intraday.py","five_currents_active_contracts.py",
            "five_currents_opening_study.py","five_currents_daily/engine.py",
            "five_currents_daily/curve_activation.py","five_currents_daily/curve_signals.py",
            "five_currents_daily/dashboard_core.py")},
        "outputs_sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file() and p.suffix in (".csv",".npz",".parquet")}}
    save_json(output / "verification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("code_sha256","existing_dependency_sha256","outputs_sha256","reproduction_order")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    verify(args.output_dir.resolve())
