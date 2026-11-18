"""Four parsimonious region-to-region execution windows, using the pinned clock-study inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import (
    ROOTS,MICROS,PriceBook,regional_time,within_session,forecast_matrix,asset_pnl,
    VARIANTS,PRIMARY_VARIANTS,walk_forward_choices,choose_path,causal_choice,summarize_matrix,
)
from run_five_currents_execution_timing_study import signals,BOOKS,FITS,metric_row,aggregate
from prepare_five_currents_execution_timing import DEFAULT_OUT,save_json


WINDOWS={"CME_to_London":("cme_open","london_open"),
         "London_to_US_open":("london_open","us_open"),
         "US_open_to_settlement":("us_open","settlement"),
         "settlement_to_US_close":("settlement","us_close")}
DELAY=(0,5,15,30)


def event_pair(book,target,first,last,offset):
    start=regional_time(target.session,first,target.root)+pd.Timedelta(minutes=offset)
    end=regional_time(target.session,last,target.root)+pd.Timedelta(minutes=offset)
    result={"session":target.session,"root":target.root,"offset":offset,"planned_entry":start,
            "planned_exit":end,"status":"missing entry","month":target.month,"identity_basis":target.identity_basis}
    if not within_session(start,target.session) or not within_session(end,target.session):
        result["status"]="outside session"
        return result
    entry=book.find(target.root,target.month,start)
    if entry is None:return result
    result.update(entry=entry.price,entry_ts=entry.ts,symbol=entry.symbol)
    exit=book.find(target.root,target.month,end,symbol=entry.symbol)
    if exit is None:result["status"]="missing same-instrument exit"
    elif exit.ts<=entry.ts or not within_session(exit.ts,target.session):result["status"]="invalid exit"
    else:result.update(status="matched",exit=exit.price,exit_ts=exit.ts,price_return=exit.price/entry.price-1.)
    return result


def prices(output,source):
    path=output / "regional_block_prices.npz"
    if path.exists():
        with np.load(path) as saved:return {name:saved[name] for name in saved.files}
    targets=pd.read_parquet(output / "contract_targets.parquet")
    shape=(len(WINDOWS),len(DELAY),len(source.sessions),len(ROOTS))
    result={name:np.full(shape,np.nan) for name in ("returns","entry")}
    location={date:i for i,date in enumerate(source.sessions)}
    records=[]
    for ai,root in enumerate(ROOTS):
        paths=list((output / "minute_cache").glob("*_"+root+".parquet"))+list((output / "minute_cache").glob("*_"+MICROS[root]+".parquet"))
        book=PriceBook(pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True))
        for target in targets.loc[targets.root.eq(root)].itertuples():
            di=location[target.session]
            for wi,(window,(first,last)) in enumerate(WINDOWS.items()):
                for oi,offset in enumerate(DELAY):
                    row=event_pair(book,target,first,last,offset)
                    row["window"]=window
                    records.append(row)
                    if row["status"]=="matched":
                        result["returns"][wi,oi,di,ai]=row["price_return"]
                        result["entry"][wi,oi,di,ai]=row["entry"]
        print("[regional cached prices]",root,flush=True)
    pd.DataFrame(records).to_parquet(output / "regional_block_price_ledger.parquet",index=False)
    np.savez_compressed(path,**result)
    return result


def weight(source,signal,variant):
    held=signal.startswith("held__")
    base=signal.removeprefix("held__")
    values=source.settled[base] if held else source.values[base]
    return forecast_matrix(values,source.dates,source.sessions,lag=int(held),variant=variant)[0]


def run(output):
    source=signals(output)
    cube=prices(output,source)
    grid,metrics,paths,decisions,robust,fixed_checks=[],[],[],[],[],[]
    for signal in source.values:
        by_book={book:{col:[] for col in ("net","gross","cost","neighbors","exposure")} for book in BOOKS}
        meta=[]
        for variant in VARIANTS:
            w=weight(source,signal,variant)
            delay_paths=[]
            for oi,offset in enumerate(DELAY):
                gross,fee,slip=asset_pnl(w[None,:,:],cube["returns"][:,oi],cube["entry"][:,oi])
                delay_paths.append((gross,fee+slip))
            gross,cost=delay_paths[0]
            for book,indices in BOOKS.items():
                for wi,window in enumerate(WINDOWS):
                    path=aggregate(gross-cost,indices)[:,wi]
                    grid.append(metric_row(path,signal=signal,book=book,window=window,variant=variant,
                        stressed_sharpe=summarize_matrix(aggregate(gross-2*cost,indices)[:,wi])["sharpe"][0]))
                    for era,mask in [(str(y),source.sessions.year==y) for y in sorted(set(source.sessions.year))]+[
                        ("tail_after_first_"+str(int(fraction*100))+"pct",np.arange(len(path))>=int(len(path)*fraction))
                        for fraction in (.3,.4,.5)]:
                        fixed_checks.append(metric_row(path[mask],signal=signal,book=book,window=window,variant=variant,
                            period=era,stressed_sharpe=summarize_matrix(aggregate(gross-2*cost,indices)[mask,wi])["sharpe"][0]))
                    for oi,offset in enumerate(DELAY):
                        g,c=delay_paths[oi]
                        part=aggregate(g-c,indices)[:,wi]
                        pair=np.isfinite(path)&np.isfinite(part)
                        robust.append(metric_row(part,signal=signal,book=book,window=window,variant=variant,
                            offset=offset,paired_days=int(pair.sum()),paired_sharpe=summarize_matrix(part[pair])["sharpe"][0],
                            paired_base_sharpe=summarize_matrix(path[pair])["sharpe"][0]))
                if variant in PRIMARY_VARIANTS:
                    fields=by_book[book]
                    for col,values in (("net",gross-cost),("gross",gross),("cost",cost)):
                        fields[col].append(aggregate(values,indices))
                    fields["neighbors"].append(np.stack([aggregate(g-2*c,indices) for g,c in delay_paths[:3]],axis=2))
                    fields["exposure"].append(np.broadcast_to(np.abs(w[:,indices]).sum(axis=1)[:,None],(len(w),len(WINDOWS))))
            if variant in PRIMARY_VARIANTS:
                meta.extend({"variant":variant,"window":window} for window in WINDOWS)
        metadata=pd.DataFrame(meta)
        for book,fields in by_book.items():
            data={col:np.concatenate(parts,axis=1) for col,parts in fields.items()}
            schedules=[]
            for fit,(lookback,refit) in FITS.items():
                choice,audit=walk_forward_choices(data["neighbors"],lookback=lookback,refit=refit)
                schedules.append((fit,choice,audit))
            for fraction in (.3,.4,.5):
                cut=int(len(source.sessions)*fraction)
                selected,score,eligible=causal_choice(data["neighbors"][:cut-1],lookback=None)
                choice=np.full(len(source.sessions),-1,dtype=int)
                choice[cut:]=selected
                audit=pd.DataFrame([{"start":cut,"end":len(choice),"train_start":0,"train_end_exclusive":cut-1,
                    "candidate":selected,"train_median_stressed_sharpe":score,"eligible":eligible}])
                schedules.append((f"frozen_{int(fraction*100)}_{100-int(fraction*100)}",choice,audit))
            for fit,choice,audit in schedules:
                net=choose_path(data["net"],choice)
                gross=choose_path(data["gross"],choice)
                cost=choose_path(data["cost"],choice)
                exposure=choose_path(data["exposure"],choice)
                start=int(audit.start.min())
                metrics.append(metric_row(net[start:],signal=signal,book=book,fit=fit,selected_sessions=int((choice[start:]>=0).sum()),
                    stressed_sharpe=summarize_matrix((gross-2*cost)[start:])["sharpe"][0]))
                paths.append(pd.DataFrame({"session":source.sessions,"signal":signal,"book":book,"fit":fit,
                    "net":net,"gross":gross,"cost":cost,"exposure":exposure,"choice":choice,
                    "calibrating":np.arange(len(net))<start}))
                for row in audit.itertuples():
                    record={"signal":signal,"book":book,"fit":fit,**row._asdict()}
                    record["last_training_session"]=source.sessions[row.train_end_exclusive-1]
                    if row.candidate>=0:record.update(metadata.iloc[row.candidate].to_dict())
                    decisions.append(record)
        print("[regional fit]",signal,flush=True)
    pd.DataFrame(grid).to_csv(output / "regional_block_full_grid.csv",index=False)
    pd.DataFrame(metrics).to_csv(output / "regional_block_causal_metrics.csv",index=False)
    pd.DataFrame(decisions).to_csv(output / "regional_block_selection_ledger.csv",index=False)
    pd.DataFrame(robust).to_csv(output / "regional_block_delay_checks.csv",index=False)
    pd.DataFrame(fixed_checks).to_csv(output / "regional_fixed_rule_era_checks.csv",index=False)
    pd.concat(paths,ignore_index=True).to_parquet(output / "regional_block_causal_paths.parquet",index=False)
    save_json(output / "regional_block_manifest.json",{"windows":WINDOWS,"offsets":DELAY,
        "rule":"Same research protocol; only four economically defined regional windows, no independent holding-period fit",
        "development_note":"Boundary-to-boundary check complements the fixed-duration grid; no fresh never-seen data are claimed",
        "data":"Same pinned daily signals and local minute cache; no database or market API requests"})
    print("[regional complete]",output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output_dir.resolve())
