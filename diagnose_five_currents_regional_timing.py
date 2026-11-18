"""Concentration and constant-direction controls, with OOS timing decisions frozen."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import PRIMARY_VARIANTS,asset_pnl,choose_path,summarize_matrix
from run_five_currents_regional_blocks import weight
from run_five_currents_execution_timing_study import signals,BOOKS,aggregate,metric_row
from prepare_five_currents_execution_timing import DEFAULT_OUT


def run(output):
    source=signals(output)
    archive=np.load(output / "regional_block_prices.npz")
    cube={name:archive[name] for name in archive.files}
    paths=pd.read_parquet(output / "regional_block_causal_paths.parquet")
    paths=paths.loc[paths.fit.eq("rolling252_monthly")]
    controls,checks=[],[]
    for signal in source.values:
        fields={book:{name:[] for name in ("long","short")} for book in BOOKS}
        for variant in PRIMARY_VARIANTS:
            original=weight(source,signal,variant)
            for direction,name in ((1,"long"),(-1,"short")):
                w=np.abs(original)*direction
                gross,fee,slip=asset_pnl(w[None,:,:],cube["returns"][:,0],cube["entry"][:,0])
                for book,indices in BOOKS.items():
                    fields[book][name].append(aggregate(gross-fee-slip,indices))
        for book in BOOKS:
            selected=paths.loc[paths.signal.eq(signal)&paths.book.eq(book)].sort_values("session")
            choices=selected.choice.to_numpy(dtype=int)
            start=~selected.calibrating.to_numpy()
            actual=selected.net.to_numpy()
            long=choose_path(np.concatenate(fields[book]["long"],axis=1),choices)
            short=choose_path(np.concatenate(fields[book]["short"],axis=1),choices)
            paired=np.isfinite(actual)&np.isfinite(long)&np.isfinite(short)&start
            controls.append({"signal":signal,"book":book,"common_days":int(paired.sum()),
                "actual_sharpe":summarize_matrix(actual[paired])["sharpe"][0],
                "always_long_same_exposure_sharpe":summarize_matrix(long[paired])["sharpe"][0],
                "always_short_same_exposure_sharpe":summarize_matrix(short[paired])["sharpe"][0],
                "actual_minus_long_mean_bps":np.mean(actual[paired]-long[paired])*10000 if paired.any() else np.nan,
                "actual_minus_short_mean_bps":np.mean(actual[paired]-short[paired])*10000 if paired.any() else np.nan,
                "rule":"Identical fitted timing, absolute exposure, instrument and friction; only direction is fixed"})
            sample=actual[start]
            dates=source.sessions[start]
            masks={"full":np.ones(len(sample),dtype=bool)}
            finite=np.flatnonzero(np.isfinite(sample))
            for n in (1,3,5,10):
                keep=np.ones(len(sample),dtype=bool)
                keep[finite[np.argsort(sample[finite])[-n:]]]=False
                masks["exclude_best_"+str(n)+"_days"]=keep
            for day in range(5):masks["exclude_weekday_"+str(day)]=dates.weekday!=day
            summer=np.array([pd.Timestamp(d).tz_localize("America/New_York").utcoffset().total_seconds()==-4*3600 for d in dates])
            masks["US_summer"],masks["US_winter"]=summer,~summer
            for year in sorted(set(dates.year)):masks[str(year)]=dates.year==year
            for test,mask in masks.items():
                checks.append(metric_row(sample[mask],signal=signal,book=book,test=test,
                    method="Original OOS choices frozen; concentration check, not retraining on filtered returns"))
    pd.DataFrame(controls).to_csv(output / "regional_direction_controls.csv",index=False)
    pd.DataFrame(checks).to_csv(output / "regional_oos_concentration_checks.csv",index=False)
    print("[diagnosed] frozen OOS concentration, seasons and direction controls",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output_dir.resolve())
