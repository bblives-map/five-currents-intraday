"""Independent arithmetic, cost, timing-capture and data-quality checks for the clock study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import (
    CLOCKS,HOLDS,OFFSETS,ROOTS,MICRO_MULTIPLIERS,SLIPPAGE_PRICE_PER_SIDE,forecast_matrix,summarize_matrix,
)
from run_five_currents_execution_timing_study import signals,pnl_grid,aggregate,metric_row,BOOKS,complete_grid
from prepare_five_currents_execution_timing import DEFAULT_OUT,save_json


def arithmetic_audit(output,source,cube):
    rng=np.random.default_rng(20261023)
    records=[]
    all_error=[]
    for signal in source.values:
        for variant in ("net_magnitude","change_magnitude","reverse_change_magnitude"):
            gross,fee,slip,weight=pnl_grid(source,cube,signal,variant)
            possible=np.argwhere(np.isfinite(gross)&(np.abs(gross)>1e-12))
            if not len(possible):
                continue
            for chosen in rng.choice(len(possible),size=min(12,len(possible)),replace=False):
                cell,day,asset=possible[chosen]
                clock=cell//len(HOLDS)
                hold=cell%len(HOLDS)
                root=ROOTS[asset]
                price=cube["entry"][clock,hold,OFFSETS.index(0),day,asset]
                move=cube["returns"][clock,hold,OFFSETS.index(0),day,asset]
                quantity=weight[day,asset]/(price*MICRO_MULTIPLIERS[root])
                manual_gross=quantity*MICRO_MULTIPLIERS[root]*(price*(1+move)-price)
                manual_fee=abs(quantity)*2.16
                manual_slip=abs(quantity)*2*SLIPPAGE_PRICE_PER_SIDE[root]*MICRO_MULTIPLIERS[root]
                errors=[abs(manual_gross-gross[cell,day,asset]),abs(manual_fee-fee[cell,day,asset]),
                        abs(manual_slip-slip[cell,day,asset])]
                all_error.extend(errors)
                records.append({"signal":signal,"variant":variant,"root":root,"session":source.sessions[day],
                    "clock":CLOCKS[clock],"hold":HOLDS[hold],"micro_quantity_per_unit_capital":quantity,
                    "vector_gross":gross[cell,day,asset],"scalar_gross":manual_gross,
                    "vector_fee":fee[cell,day,asset],"scalar_fee":manual_fee,"max_error":max(errors)})
    pd.DataFrame(records).to_csv(output / "independent_scalar_audit.csv",index=False)
    np.testing.assert_allclose(source.saved["allocated_lagged"].sum(axis=0),source.saved["net_lagged"],atol=1e-10)
    np.testing.assert_allclose(source.saved["allocated_settled"].sum(axis=0),source.saved["net_settled"],atol=1e-10)
    if max(all_error,default=0)>1e-10:
        raise AssertionError("Independent lot-level PnL/cost check failed")
    save_json(output / "arithmetic_audit.json",{"passed":True,"scalar_comparisons":len(records),
        "maximum_absolute_error":max(all_error,default=0),"allocated_sleeves_sum_to_net":True,
        "commission_usd_per_micro_round_trip":2.16,"silver_slippage_price_per_side_usd_per_oz":.01,
        "silver_exchange_minimum_tick_usd_per_oz":.005,
        "daily_weights_reconcile_to_archived_dashboard":source.manifest["daily_weight_reconciliation_max_error"]})


def execution_capture(output,source,cube):
    targets=pd.read_parquet(output / "contract_targets.parquet")
    panel=pd.read_parquet(output / "inputs/daily_panel.parquet")
    mapping=pd.read_parquet(output / "inputs/contract_mapping.parquet")
    close_reference=np.full((len(source.sessions),len(ROOTS)),np.nan)
    same_contract=np.zeros(close_reference.shape,dtype=bool)
    for ai,root in enumerate(ROOTS):
        group=targets.loc[targets.root.eq(root)].set_index("session").reindex(source.sessions)
        same_contract[:,ai]=(group.month==group.month.shift(1)).to_numpy()
        for di,(date,row) in enumerate(group.iterrows()):
            if pd.isna(row["rank"]):
                continue
            rank=int(row["rank"])
            refdate=pd.Timestamp(row.liquidity_date)
            key=(f"{root}{rank} Comdty","PX_SETTLE")
            value=panel.at[refdate,key] if refdate in panel.index and key in panel else np.nan
            if not np.isfinite(value) or value<=0:
                key=(f"{root}{rank} Comdty","PX_LAST")
                value=panel.at[refdate,key] if refdate in panel.index and key in panel else np.nan
            close_reference[di,ai]=value*(.01 if root=="HG" else 1.)
    current,_=forecast_matrix(source.saved["net_lagged"],source.dates,source.sessions)
    held,_=forecast_matrix(source.saved["net_settled"],source.dates,source.sessions,lag=1)
    delta=current-held
    settle_ci=CLOCKS.index("settlement")
    oi=OFFSETS.index(0)
    benchmark=cube["entry"][settle_ci,0,oi]
    benchmark_micro=cube["micro"][settle_ci,0,oi]
    paths,summary=[],[]
    for ci,clock in enumerate(CLOCKS):
        for offset in (0,5,15,30):
            entry=cube["entry"][ci,0,OFFSETS.index(offset)]
            same_instrument=cube["micro"][ci,0,OFFSETS.index(offset)]==benchmark_micro
            valid=same_contract&same_instrument&np.isfinite(benchmark)&np.isfinite(entry)&np.isfinite(close_reference)
            capture=delta*(benchmark-entry)/close_reference
            capture=np.where(np.abs(delta)<=1e-12,0.,np.where(valid,capture,np.nan))
            # Identical signed order quantities, commissions and tick slippage at
            # both execution times: these costs cancel in the timing difference.
            for book,indices in BOOKS.items():
                x=aggregate(capture[None,:,:],indices)[:,0]
                summary.append(metric_row(x,clock=clock,offset=offset,book=book,
                    comparison="fixed-order rebalance preview versus settlement-window bar proxy, non-roll days"))
                paths.append(pd.DataFrame({"session":source.sessions,"clock":clock,"offset":offset,"book":book,
                                          "incremental_timing_capture":x}))
    pd.DataFrame(summary).to_csv(output / "rebalance_timing_capture.csv",index=False)
    pd.concat(paths,ignore_index=True).to_parquet(output / "rebalance_timing_capture_paths.parquet",index=False)


def session_decomposition(output,source,cube):
    order=("cme_open","early_utc","london_open","us_open","london_close","settlement","us_close")
    oi=OFFSETS.index(0)
    starts=np.stack([cube["entry"][CLOCKS.index(clock),0,oi] for clock in order])
    micro=np.stack([cube["micro"][CLOCKS.index(clock),0,oi] for clock in order])
    first=cube["entry"][CLOCKS.index("cme_open"),HOLDS.index(0),oi]
    end=first*(1+cube["returns"][CLOCKS.index("cme_open"),HOLDS.index(0),oi])
    prices=np.concatenate([starts,end[None,:,:]],axis=0)
    same_source=(micro==micro[0:1]).all(axis=0)
    complete=np.isfinite(prices).all(axis=0)&same_source&np.isclose(first,starts[0],equal_nan=False)
    summaries,paths=[],[]
    maximum_error=0.
    for signal in source.values:
        held=signal.startswith("held__")
        base=signal.removeprefix("held__")
        values=source.settled[base] if held else source.values[base]
        weight,_=forecast_matrix(values,source.dates,source.sessions,lag=int(held))
        contribution=weight[None,:,:]*(prices[1:]-prices[:-1])/first[None,:,:]
        contribution=np.where(complete[None,:,:],contribution,np.nan)
        total=weight*(end-first)/first
        error=np.abs(np.sum(contribution,axis=0)-total)
        if np.isfinite(error).any():
            maximum_error=max(maximum_error,float(np.nanmax(error)))
        for book,indices in BOOKS.items():
            for i in range(len(order)):
                label=order[i]+" -> "+(order[i+1] if i+1<len(order) else "session_close")
                x=aggregate(contribution[i:i+1],indices)[:,0]
                summaries.append(metric_row(x,signal=signal,book=book,interval=label,
                    basis="Fixed quantity across the same session; gross before one shared round-trip cost"))
                if signal.removeprefix("held__")=="portfolio_net":
                    paths.append(pd.DataFrame({"session":source.sessions,"signal":signal,"book":book,
                                              "interval":label,"gross_contribution":x}))
    pd.DataFrame(summaries).to_csv(output / "session_return_decomposition.csv",index=False)
    pd.concat(paths,ignore_index=True).to_parquet(output / "session_return_decomposition_paths.parquet",index=False)
    if maximum_error>1e-10:
        raise AssertionError("Intraday windows do not reconcile to the full same-contract session")
    save_json(output / "session_decomposition_audit.json",{"passed":True,"maximum_absolute_error":maximum_error,
        "rule":"Same contract, same price instrument, same fixed quantity for all seven disjoint windows",
        "cost_rule":"No fictitious exit/re-entry or repeated commission between adjacent windows"})


def selected_recipe_checks(output,source,cube):
    ledger=pd.read_csv(output / "timing_selection_ledger.csv")
    latest=ledger.loc[ledger.fit.eq("rolling252_monthly")&ledger.book.isin(ROOTS)]
    latest=latest.sort_values("start").groupby(["signal","book"]).tail(1)
    full=pd.read_csv(output / "walk_forward_metrics.csv")
    full=full.loc[full.period.eq("full_replay")]
    rows=[]
    paths=[]
    target=pd.read_parquet(output / "contract_targets.parquet")
    for row in latest.itertuples():
        record={"signal":row.signal,"asset":row.book,"clock":getattr(row,"clock",None),"hold":getattr(row,"hold",None),
            "variant":getattr(row,"variant",None),"train_until":row.last_training_session,
            "train_stressed_sharpe":row.train_median_stressed_sharpe,"candidate":row.candidate}
        if row.candidate<0:
            record["assessment"]="Cash: no eligible positive training plateau"
            rows.append(record)
            continue
        selected=full.loc[full.signal.eq(row.signal)&full.book.eq(row.book)]
        base=selected.loc[selected.fit.eq("rolling252_monthly")].iloc[0]
        frozen=selected.loc[selected.fit.str.startswith("frozen")]
        alternatives=selected.loc[selected.fit.isin(("rolling504_monthly","rolling252_weekly","expanding_monthly"))]
        record.update(oos_sharpe=base.sharpe,oos_stressed_sharpe=base.stressed_sharpe,
            oos_days=base.days,oos_coverage=base.coverage,
            frozen_positive=int(frozen.sharpe.gt(0).sum()),frozen_checks=len(frozen),
            alternate_positive=int(alternatives.sharpe.gt(0).sum()),alternate_checks=len(alternatives))
        record["assessment"]=("Paper-trade candidate; not a validated deployment rule" if base.sharpe>0
            and base.stressed_sharpe>0 and (frozen.sharpe>0).sum()>=2 and (alternatives.sharpe>0).sum()>=2
            else "Exploratory / unstable across chronological checks")
        rows.append(record)
        cell=CLOCKS.index(row.clock)*len(HOLDS)+HOLDS.index(int(row.hold))
        ai=ROOTS.index(row.book)
        g,fee,slip,weight=pnl_grid(source,cube,row.signal,row.variant,cell=cell)
        base_path=(g-fee-slip)[0,:,ai]
        information=target.loc[target.root.eq(row.book)].set_index("session").reindex(source.sessions)
        verified=information.identity_basis.eq("verified_month").to_numpy()
        recent_active=verified&np.isfinite(base_path)&(np.abs(weight[:,ai])>1e-12)
        record["verified_active_observations"]=int(recent_active.sum())
        record["verified_active_sharpe"]=summarize_matrix(base_path[recent_active])["sharpe"][0]
        ny_summer=np.array([pd.Timestamp(day).tz_localize("America/New_York").utcoffset().total_seconds()==-4*3600
                            for day in source.sessions])
        same_dst_season=np.array([(pd.Timestamp(day).tz_localize("Europe/London").utcoffset()
            -pd.Timestamp(day).tz_localize("America/New_York").utcoffset()).total_seconds()==5*3600
            for day in source.sessions])
        masks={"verified_delivery_month":information.identity_basis.eq("verified_month").to_numpy(),
            "inferred_delivery_month":information.identity_basis.eq("inferred_delivery_month").to_numpy(),
            "exclude_last_five_expiry_sessions":information.sessions_to_expiry.gt(5).to_numpy(),
            "exclude_delivery_calendar_month":np.array([m>str(s.to_period('M')) if isinstance(m,str) else False
                for m,s in zip(information.month,source.sessions)]),
            "micro_price_only":cube["micro"][CLOCKS.index(row.clock),HOLDS.index(int(row.hold)),OFFSETS.index(0),:,ai]}
        masks.update({"US_summer":ny_summer,"US_winter":~ny_summer,"exclude_UK_US_DST_mismatch":same_dst_season})
        for label,mask in masks.items():
            paths.append(metric_row(base_path[mask],signal=row.signal,asset=row.book,clock=row.clock,
                                     hold=row.hold,variant=row.variant,check=label))
    pd.DataFrame(rows).to_csv(output / "latest_causal_timing_candidates.csv",index=False)
    pd.DataFrame(paths).to_csv(output / "contract_expiry_and_proxy_checks.csv",index=False)


def personalized_portfolios(output,source):
    original=pd.read_parquet(output / "walk_forward_paths.parquet")
    families=source.manifest["signal_names"]
    allowed=set(families)|{"held__"+name for name in families}
    assets=original.loc[original.book.isin(ROOTS)&original.signal.isin(allowed)]
    all_family_paths={}
    details=[]
    for (fit,signal),group in assets.groupby(["fit","signal"],sort=False):
        for book,roots in (("four",("CL","NG","GC","HG")),("five",ROOTS)):
            pivot={col:group.pivot(index="session",columns="book",values=col).reindex(
                index=source.sessions,columns=roots) for col in ("net","gross","cost","exposure","calibrating")}
            gross_budget=pivot["exposure"].sum(axis=1)
            scale=1/np.maximum(1,gross_budget.to_numpy())
            values={col:np.where(np.isfinite(pivot[col]).all(axis=1),pivot[col].sum(axis=1)*scale,np.nan)
                    for col in ("net","gross","cost")}
            calibrated=~pivot["calibrating"].all(axis=1).to_numpy()
            part=pd.DataFrame({"session":source.sessions,"signal":signal,"fit":fit,"book":book,
                "net":values["net"],"gross":values["gross"],"cost":values["cost"],"calibrating":~calibrated,
                "known_gross_ceiling":gross_budget*scale})
            all_family_paths.setdefault((fit,book,"held__" if signal.startswith("held__") else ""),[]).append(part)
    paths=[]
    for (fit,book,prefix),parts in all_family_paths.items():
        if len(parts)!=10:
            raise AssertionError("Personalized book does not contain exactly ten prespecified families")
        arrays={col:np.column_stack([p[col] for p in parts]) for col in ("net","gross","cost")}
        values={col:np.where(np.isfinite(x).all(axis=1),x.sum(axis=1)/10,np.nan) for col,x in arrays.items()}
        calibrated=~parts[0].calibrating.to_numpy()
        label=prefix+"asset_specific_equal_ten_intraday"
        path=pd.DataFrame({"session":source.sessions,"signal":label,"fit":fit,"book":book,
                           "calibrating":~calibrated,**values})
        paths.append(path)
        details.append(metric_row(values["net"][calibrated],signal=label,fit=fit,book=book,
            stressed_sharpe=summarize_matrix((values["gross"]-2*values["cost"])[calibrated])["sharpe"][0],
            gross_sharpe=summarize_matrix(values["gross"][calibrated])["sharpe"][0]))
    combined=pd.concat(paths,ignore_index=True)
    combined.to_parquet(output / "asset_specific_portfolio_paths.parquet",index=False)
    pd.DataFrame(details).to_csv(output / "asset_specific_portfolio_metrics.csv",index=False)
    comparisons=[]
    for fit in combined.fit.unique():
        for book in ("four","five"):
            for prefix in ("","held__"):
                first=combined.loc[combined.fit.eq(fit)&combined.book.eq(book)
                    &combined.signal.eq(prefix+"asset_specific_equal_ten_intraday")].set_index("session")
                shared=original.loc[original.fit.eq(fit)&original.book.eq(book)
                    &original.signal.eq(prefix+"equal_ten_intraday")].set_index("session")
                mask=~first.calibrating&np.isfinite(first.net)&np.isfinite(shared.net)
                a,b=first.loc[mask,"net"],shared.loc[mask,"net"]
                comparisons.append({"fit":fit,"book":book,"timeline":"held" if prefix else "fresh_lagged",
                    "common_days":len(a),"personalized_sharpe":summarize_matrix(a)["sharpe"][0],
                    "shared_family_sharpe":summarize_matrix(b)["sharpe"][0],"correlation":a.corr(b),
                    "incremental_mean_bps":(a-b).mean()*10000})
    pd.DataFrame(comparisons).to_csv(output / "personalized_vs_shared_common_dates.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--reference-grid",action="store_true")
    args=parser.parse_args()
    output=args.output_dir.resolve()
    source=signals(output)
    with np.load(output / "price_cube.npz") as archive:
        cube={name:archive[name] for name in archive.files}
    with threadpool_limits(limits=1):
        arithmetic_audit(output,source,cube)
        execution_capture(output,source,cube)
        session_decomposition(output,source,cube)
        selected_recipe_checks(output,source,cube)
        personalized_portfolios(output,source)
        if args.reference_grid:
            reference=SimpleNamespace(**vars(source))
            reference.values=source.settled
            frame=complete_grid(output,reference,cube,filename="availability_assumed_settled_reference_grid.csv")
            # Current-session final flows must never be represented as CME-open tradable.
            frame=frame.loc[frame.clock.ne("cme_open")]
            frame.to_csv(output / "availability_assumed_settled_reference_grid.csv",index=False)
    print("[audited] independent arithmetic, rebalance capture and latest causal timings",output,flush=True)


if __name__=="__main__":
    main()
