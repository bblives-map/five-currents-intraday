"""Freeze every OOS recipe decision, then perturb fills and signal availability."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import (
    ROOTS,CLOCKS,HOLDS,PRIMARY_VARIANTS,forecast_matrix,asset_pnl,choose_path,summarize_matrix,
)
from run_five_currents_execution_timing_study import signals,pnl_grid,BOOKS,aggregate,metric_row
from run_five_currents_regional_blocks import WINDOWS,DELAY
from prepare_five_currents_execution_timing import DEFAULT_OUT


def frozen_stress(output,source,*,regional):
    if regional:
        cube=np.load(output / "regional_block_prices.npz")
        cube={name:cube[name] for name in cube.files}
        paths=pd.read_parquet(output / "regional_block_causal_paths.parquet")
        path=output / "regional_frozen_schedule_stress.csv"
        selected_cells=list(range(len(WINDOWS)))
    else:
        cube=np.load(output / "price_cube.npz")
        cube={name:cube[name] for name in cube.files}
        paths=pd.read_parquet(output / "walk_forward_paths.parquet")
        path=output / "frozen_schedule_stress.csv"
        selected_cells=[i for i,(_,hold) in enumerate((c,h) for c in CLOCKS for h in HOLDS) if hold>0]
    records=[]
    for signal in source.values:
        schedules=paths.loc[paths.signal.eq(signal)]
        grouped={(book,fit):group.sort_values("session") for (book,fit),group in schedules.groupby(["book","fit"])}
        for lag in (0,1,2):
            for offset in DELAY:
                fields={book:{name:[] for name in ("net","gross","cost")} for book in BOOKS}
                for variant in PRIMARY_VARIANTS:
                    if regional:
                        held=signal.startswith("held__")
                        base=signal.removeprefix("held__")
                        values=source.settled[base] if held else source.values[base]
                        w,_=forecast_matrix(values,source.dates,source.sessions,lag=int(held)+lag,variant=variant)
                        oi=DELAY.index(offset)
                        gross,fee,slip=asset_pnl(w[None,:,:],cube["returns"][:,oi],cube["entry"][:,oi])
                    else:
                        gross,fee,slip,_=pnl_grid(source,cube,signal,variant,lag=lag,offset=offset)
                    for book,indices in BOOKS.items():
                        for name,values in (("net",gross-fee-slip),("gross",gross),("cost",fee+slip)):
                            fields[book][name].append(aggregate(values,indices)[:,selected_cells])
                for (book,fit),schedule in grouped.items():
                    choices=schedule.choice.to_numpy(dtype=int)
                    calibrated=~schedule.calibrating.to_numpy()
                    arrays={name:np.concatenate(parts,axis=1) for name,parts in fields[book].items()}
                    net=choose_path(arrays["net"],choices)
                    gross=choose_path(arrays["gross"],choices)
                    cost=choose_path(arrays["cost"],choices)
                    baseline=schedule.net.to_numpy()
                    common=np.isfinite(net)&np.isfinite(baseline)&calibrated
                    if lag==0 and offset==0:
                        np.testing.assert_allclose(net,baseline,atol=1e-12,equal_nan=True)
                    row=metric_row(net[calibrated],signal=signal,book=book,fit=fit,lag=lag,offset=offset,
                        common_days=int(common.sum()),paired_sharpe=summarize_matrix(net[common])["sharpe"][0],
                        paired_baseline_sharpe=summarize_matrix(baseline[common])["sharpe"][0],
                        stressed_sharpe=summarize_matrix((gross-2*cost)[calibrated])["sharpe"][0],
                        same_recipes_not_retuned=True)
                    records.append(row)
        print("[frozen-schedule stress]",("regional" if regional else "fixed hold"),signal,flush=True)
    pd.DataFrame(records).to_csv(path,index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--regional-only",action="store_true")
    args=parser.parse_args()
    output=args.output_dir.resolve()
    source=signals(output)
    with threadpool_limits(limits=1):
        if not args.regional_only:frozen_stress(output,source,regional=False)
        frozen_stress(output,source,regional=True)
