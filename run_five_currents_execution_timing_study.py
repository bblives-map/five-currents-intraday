"""Offline regional-clock study of the existing Five Currents signal roster."""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from five_currents_execution_timing import (
    CLOCKS, HOLDS, OFFSETS, VARIANTS, PRIMARY_VARIANTS, ROOTS, PriceBook, trade_pair,
    forecast_matrix, asset_pnl, summarize_matrix, walk_forward_choices, causal_choice,
    choose_path, block_interval,
)
from prepare_five_currents_execution_timing import DEFAULT_OUT, digest, save_json


BOOKS = {**{r:(i,) for i,r in enumerate(ROOTS)}, "four":(0,1,2,4), "five":(0,1,2,3,4)}
FITS = {"rolling252_monthly":(252,21), "rolling504_monthly":(504,21),
        "rolling252_weekly":(252,5), "expanding_monthly":(None,21)}


def build_prices(output):
    targets = pd.read_parquet(output / "contract_targets.parquet")
    sessions = pd.DatetimeIndex(pd.read_parquet(output / "sessions.parquet").session)
    location = {date:i for i,date in enumerate(sessions)}
    shape = (len(CLOCKS),len(HOLDS),len(OFFSETS),len(sessions),len(ROOTS))
    arrays = {name:np.full(shape,np.nan) for name in ("returns","entry","mark_returns","mark_entry")}
    arrays.update({name:np.zeros(shape,dtype=bool) for name in ("micro","verified")})
    audits = output / "price_ledgers"
    audits.mkdir(exist_ok=True)
    coverage = []
    for ai,root in enumerate(ROOTS):
        frames = [pd.read_parquet(p) for p in (output / "minute_cache").glob("*_"+root+".parquet")]
        from five_currents_front_intraday import MICROS
        frames += [pd.read_parquet(p) for p in (output / "minute_cache").glob("*_"+MICROS[root]+".parquet")]
        bars = pd.concat(frames,ignore_index=True)
        book = PriceBook(bars)
        records = []
        for target in targets.loc[targets.root.eq(root)].itertuples(index=False):
            di = location[target.session]
            for ci,clock in enumerate(CLOCKS):
                for oi,offset in enumerate(OFFSETS):
                    for hi,hold in enumerate(HOLDS):
                        record = trade_pair(book,target,clock,hold,offset)
                        key = (ci,hi,oi,di,ai)
                        if record["status"] == "matched":
                            arrays["returns"][key] = record["price_return"]
                            arrays["entry"][key] = record["entry_price"]
                            arrays["micro"][key] = record["symbol"] == MICROS[root]
                            arrays["verified"][key] = target.identity_basis == "verified_month"
                            mark = record
                        else:
                            mark = trade_pair(book,target,clock,hold,offset,require_trade=False)
                        if mark["status"] == "matched":
                            arrays["mark_returns"][key] = mark["price_return"]
                            arrays["mark_entry"][key] = mark["entry_price"]
                        record["mark_status"] = mark["status"]
                        records.append(record)
        frame = pd.DataFrame(records)
        frame.to_parquet(audits / (root+".parquet"),index=False)
        summary = frame.groupby(["root","clock","hold","offset","status"],dropna=False).size().rename("sessions").reset_index()
        coverage.append(summary)
        print(f"[prices] {root}: {frame.status.eq('matched').sum():,}/{len(frame):,} traded pairs; failures retained",flush=True)
    np.savez_compressed(output / "price_cube.npz",**arrays)
    pd.concat(coverage,ignore_index=True).to_csv(output / "price_coverage.csv",index=False)


def signals(output):
    manifest = json.loads((output / "source_manifest.json").read_text())
    saved = np.load(output / "weight_arrays.npz")
    dates = pd.DatetimeIndex(saved["dates"].astype("datetime64[ns]"))
    sessions = pd.DatetimeIndex(pd.read_parquet(output / "sessions.parquet").session)
    values = {name:saved["raw_lagged"][i] for i,name in enumerate(manifest["signal_names"])}
    settled = {name:saved["raw_settled"][i] for i,name in enumerate(manifest["signal_names"])}
    values["portfolio_net"] = saved["net_lagged"]
    settled["portfolio_net"] = saved["net_settled"]
    for name,array in list(settled.items()):
        values["held__"+name] = array
    return SimpleNamespace(manifest=manifest,saved=saved,dates=dates,sessions=sessions,values=values,settled=settled)


def pnl_grid(source,cube,signal,variant,*,offset=0,lag=0,policy="lagged",marks=False,cell=None):
    held = signal.startswith("held__")
    base = signal.removeprefix("held__")
    data = source.settled if held or policy != "lagged" else source.values
    effective_lag = lag+(1 if held else 0)+(1 if policy == "held_settled" else 0)
    weight,_ = forecast_matrix(data[base],source.dates,source.sessions,lag=effective_lag,variant=variant)
    oi = OFFSETS.index(offset)
    price = cube["mark_entry" if marks else "entry"][:,:,oi].reshape(-1,len(source.sessions),len(ROOTS))
    moves = cube["mark_returns" if marks else "returns"][:,:,oi].reshape(price.shape)
    if cell is not None:
        price,moves = price[cell:cell+1],moves[cell:cell+1]
    gross,fee,slip = asset_pnl(weight[None,:,:],moves,price)
    return gross,fee,slip,weight


def aggregate(values,indices):
    selected = values[:,:,indices]
    return np.where(np.isfinite(selected).all(axis=2),np.sum(selected,axis=2),np.nan).T


def metric_row(values,**metadata):
    result = {k:v[0] for k,v in summarize_matrix(values).items()}
    valid = np.asarray(values)[np.isfinite(values)]
    result["median_gain"] = float(np.median(valid[valid>0])) if (valid>0).any() else np.nan
    result["median_loss"] = float(np.median(valid[valid<0])) if (valid<0).any() else np.nan
    return {**metadata,**result}


def complete_grid(output,source,cube,*,filename="full_grid.csv"):
    rows = []
    cells = list(product(CLOCKS,HOLDS))
    metadata = pd.DataFrame([{"clock":clock,"hold":hold,"book":book}
        for clock,hold in cells for book in BOOKS])
    for signal in source.values:
        for variant in VARIANTS:
            gross,fee,slip,weight = pnl_grid(source,cube,signal,variant)
            cost = fee+slip
            net = gross-cost
            matrices = []
            for a in (gross,net,gross-2*cost):
                matrices.append(np.stack([aggregate(a,indices) for indices in BOOKS.values()],axis=2)
                                .reshape(len(source.sessions),-1))
            measures = summarize_matrix(matrices[1])
            measures["gross_sharpe"] = summarize_matrix(matrices[0])["sharpe"]
            measures["stressed_sharpe"] = summarize_matrix(matrices[2])["sharpe"]
            measures["commission_drag"] = np.stack([np.nansum(aggregate(fee,idx),axis=0) for idx in BOOKS.values()],axis=1).ravel()
            measures["slippage_drag"] = np.stack([np.nansum(aggregate(slip,idx),axis=0) for idx in BOOKS.values()],axis=1).ravel()
            frame = metadata.copy()
            for name,values in measures.items():
                frame[name] = values
            frame["signal"],frame["variant"] = signal,variant
            rows.append(frame)
        print("[grid]",signal,"all clocks, eight signal classes, nine horizons",flush=True)
    metrics = pd.concat(rows,ignore_index=True)
    metrics.to_csv(output / filename,index=False)
    return metrics


def candidate_matrices(source,cube,signal):
    cells = list(product(CLOCKS,HOLDS))
    selected_cells = [i for i,(_,hold) in enumerate(cells) if hold>0]
    metadata = pd.DataFrame([{"variant":v,"clock":cells[i][0],"hold":cells[i][1],"grid_cell":i}
        for v in PRIMARY_VARIANTS for i in selected_cells])
    result = {book:{name:[] for name in ("net","gross","cost","neighbors","exposure")}
              for book in BOOKS}
    for variant in PRIMARY_VARIANTS:
        neighbors = []
        for offset in (0,5,15):
            gross,fee,slip,weight = pnl_grid(source,cube,signal,variant,offset=offset)
            cost = fee+slip
            neighbors.append((gross,cost,weight))
        gross,cost,weight = neighbors[0]
        for book,indices in BOOKS.items():
            for name,values in (("net",gross-cost),("gross",gross),("cost",cost)):
                result[book][name].append(aggregate(values,indices)[:,selected_cells])
            result[book]["neighbors"].append(np.stack([
                aggregate(g-2*c,indices)[:,selected_cells] for g,c,_ in neighbors],axis=2))
            exposure = np.abs(weight[:,indices]).sum(axis=1)
            result[book]["exposure"].append(np.broadcast_to(exposure[:,None],(len(exposure),len(selected_cells))))
    return metadata,{book:{name:np.concatenate(parts,axis=1) for name,parts in fields.items()}
                     for book,fields in result.items()}


def causal_replay(output,source,cube):
    metrics,paths,decisions,yearly,attribution = [],[],[],[],[]
    ensemble = {}
    chosen_recipes = set()
    for signal in source.values:
        metadata,books = candidate_matrices(source,cube,signal)
        for book,fields in books.items():
            schedules = []
            for label,(lookback,refit) in FITS.items():
                choices,audit = walk_forward_choices(fields["neighbors"],lookback=lookback,refit=refit)
                schedules.append((label,choices,audit))
            for fraction in (.3,.4,.5):
                cut = int(len(source.sessions)*fraction)
                chosen,fitness,eligible = causal_choice(fields["neighbors"][:cut-1],lookback=None)
                choices = np.full(len(source.sessions),-1,dtype=int)
                choices[cut:] = chosen
                audit = pd.DataFrame([{"start":cut,"end":len(choices),"train_start":0,
                    "train_end_exclusive":cut-1,"candidate":chosen,"train_median_stressed_sharpe":fitness,"eligible":eligible}])
                schedules.append((f"frozen_{int(fraction*100)}_{100-int(fraction*100)}",choices,audit))
            for fit,choices,audit in schedules:
                net = choose_path(fields["net"],choices)
                gross = choose_path(fields["gross"],choices)
                cost = choose_path(fields["cost"],choices)
                exposure = choose_path(fields["exposure"],choices)
                started = int(audit.start.min()) if len(audit) else len(choices)
                masks = {"full_replay":np.arange(len(net))>=started,
                         "2024":source.sessions.year==2024,"2025":source.sessions.year==2025,"2026":source.sessions.year==2026}
                common = {"signal":signal,"book":book,"fit":fit}
                for period,mask in masks.items():
                    actual = mask & (np.arange(len(net))>=started)
                    if not actual.any():
                        continue
                    row = metric_row(net[actual],**common,period=period,
                        live_from=str(source.sessions[started].date()), selected_sessions=int((choices[actual]>=0).sum()),
                        total_sessions=int(actual.sum()),mean_gross_exposure=float(np.mean(exposure[actual])),
                        gross_sharpe=summarize_matrix(gross[actual])["sharpe"][0],
                        stressed_sharpe=summarize_matrix((gross-2*cost)[actual])["sharpe"][0])
                    (metrics if period=="full_replay" else yearly).append(row)
                if fit == "rolling252_monthly":
                    metrics[-1].update(block_interval(net[started:]))
                path = pd.DataFrame({"session":source.sessions,"net":net,"gross":gross,"cost":cost,
                    "exposure":exposure,"choice":choices,"calibrating":np.arange(len(net))<started,**common})
                paths.append(path)
                audit = audit.copy()
                for name,value in common.items():
                    audit[name] = value
                audit["forecast_start"] = source.sessions[audit.start.to_numpy()]
                audit["last_training_session"] = source.sessions[(audit.train_end_exclusive-1).to_numpy()]
                for i,row in audit.iterrows():
                    if row.candidate>=0:
                        chosen = metadata.iloc[int(row.candidate)]
                        for col in ("variant","clock","hold"):
                            audit.loc[i,col] = chosen[col]
                        chosen_recipes.add((signal,book,str(chosen.variant),str(chosen.clock),int(chosen.hold)))
                decisions.append(audit)
                if signal.removeprefix("held__") != "portfolio_net" and book in ("four","five"):
                    key = (fit,book,"held__" if signal.startswith("held__") else "")
                    ensemble.setdefault(key,[]).append(path)
                if signal.removeprefix("held__") != "portfolio_net" and book in ROOTS:
                    attribution.append(path)
        print("[causal]",signal,"monthly/weekly fits, 30:70 / 40:60 / 50:50 frozen checks",flush=True)
    for (fit,book,prefix),parts in ensemble.items():
        values = np.column_stack([p.net for p in parts])
        costs = np.column_stack([p.cost for p in parts])
        gross = np.column_stack([p.gross for p in parts])
        net = np.where(np.isfinite(values).all(axis=1),values.sum(axis=1)/10,np.nan)
        cost = np.where(np.isfinite(costs).all(axis=1),costs.sum(axis=1)/10,np.nan)
        g = np.where(np.isfinite(gross).all(axis=1),gross.sum(axis=1)/10,np.nan)
        started = np.logical_or.reduce([~p.calibrating.to_numpy() for p in parts])
        row = metric_row(net[started],signal=prefix+"equal_ten_intraday",book=book,fit=fit,period="full_replay",
                         total_sessions=int(started.sum()),stressed_sharpe=summarize_matrix((g-2*cost)[started])["sharpe"][0])
        row.update(block_interval(net[started]))
        metrics.append(row)
        paths.append(pd.DataFrame({"session":source.sessions,"signal":prefix+"equal_ten_intraday","book":book,"fit":fit,
                                  "net":net,"gross":g,"cost":cost,"calibrating":~started}))
    pd.DataFrame(metrics).to_csv(output / "walk_forward_metrics.csv",index=False)
    pd.DataFrame(yearly).to_csv(output / "walk_forward_yearly.csv",index=False)
    pd.concat(paths,ignore_index=True).to_parquet(output / "walk_forward_paths.parquet",index=False)
    pd.concat(decisions,ignore_index=True).to_csv(output / "timing_selection_ledger.csv",index=False)
    return pd.DataFrame(metrics),chosen_recipes


def robustness(output,source,cube,grid,recipes):
    # Full-history winners are retained as explicitly descriptive, never as OOS choices.
    eligible = grid.loc[grid.coverage.ge(.4)&grid.days.ge(100)&grid.hold.gt(0)]
    top = eligible.sort_values("sharpe",ascending=False).groupby(["signal","book","variant"],sort=False).head(1)
    top.to_csv(output / "descriptive_winners.csv",index=False)
    selected = set(recipes)
    selected.update(tuple(r) for r in top[["signal","book","variant","clock","hold"]].itertuples(index=False,name=None))
    output_rows,year_rows,ablation_rows = [],[],[]
    for ordinal,(signal,book,variant,clock,hold) in enumerate(sorted(selected)):
        indices = BOOKS[book]
        ci,hi = CLOCKS.index(clock),HOLDS.index(hold)
        cell = ci*len(HOLDS)+hi
        base_paths = {}
        for lag in (0,1,2):
            for offset in OFFSETS:
                gross,fee,slip,weight = pnl_grid(source,cube,signal,variant,offset=offset,lag=lag,cell=cell)
                g = aggregate(gross,indices)[:,0]
                c = aggregate(fee+slip,indices)[:,0]
                net = g-c
                base_paths[(lag,offset)] = (net,g,c)
        base = base_paths[(0,0)][0]
        common_neighbors = np.isfinite(np.column_stack([base_paths[(0,o)][0] for o in (0,5,15)])).all(axis=1)
        for (lag,offset),(net,g,c) in base_paths.items():
            paired = np.isfinite(base)&np.isfinite(net)
            output_rows.append(metric_row(net,signal=signal,book=book,variant=variant,clock=clock,hold=hold,
                lag=lag,offset=offset,paired_days=int(paired.sum()),paired_base_sharpe=summarize_matrix(base[paired])["sharpe"][0],
                paired_sharpe=summarize_matrix(net[paired])["sharpe"][0],
                common_delay_days=int(common_neighbors.sum()),
                common_delay_sharpe=summarize_matrix(net[common_neighbors])["sharpe"][0],
                stressed_sharpe=summarize_matrix(g-2*c)["sharpe"][0]))
        staggered = np.column_stack([base_paths[(0,o)][0] for o in (0,5,15)])
        staggered_net = np.where(np.isfinite(staggered).all(axis=1),staggered.mean(axis=1),np.nan)
        output_rows.append(metric_row(staggered_net,signal=signal,book=book,variant=variant,clock=clock,
                                      hold=hold,lag=0,offset=0,policy="staggered_0_5_15"))
        for policy in ("held_settled","settled_reference"):
            g,fee,slip,_ = pnl_grid(source,cube,signal,variant,policy=policy,cell=cell)
            net = aggregate(g-fee-slip,indices)[:,0]
            output_rows.append(metric_row(net,signal=signal,book=book,variant=variant,clock=clock,hold=hold,
                                          lag=0,offset=0,policy=policy))
        for year in sorted(set(source.sessions.year)):
            year_rows.append(metric_row(base[source.sessions.year==year],signal=signal,book=book,
                variant=variant,clock=clock,hold=hold,year=int(year)))
        for weekday in range(5):
            keep = source.sessions.weekday!=weekday
            ablation_rows.append(metric_row(base[keep],signal=signal,book=book,variant=variant,clock=clock,
                                            hold=hold,test="exclude_weekday",removed=weekday))
        finite = np.flatnonzero(np.isfinite(base))
        keep = np.ones(len(base),dtype=bool)
        if len(finite):
            keep[finite[np.argsort(base[finite])[-5:]]] = False
        ablation_rows.append(metric_row(base[keep],signal=signal,book=book,variant=variant,clock=clock,
                                        hold=hold,test="exclude_top_5_profit_days",removed=5))
        if ordinal%100==0:
            print(f"[robustness] {ordinal+1}/{len(selected)} recipes",flush=True)
    pd.DataFrame(output_rows).to_csv(output / "delay_lag_cost_robustness.csv",index=False)
    pd.DataFrame(year_rows).to_csv(output / "recipe_yearly.csv",index=False)
    pd.DataFrame(ablation_rows).to_csv(output / "recipe_ablations.csv",index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--prices-only",action="store_true")
    parser.add_argument("--skip-prices",action="store_true")
    parser.add_argument("--skip-robustness",action="store_true")
    parser.add_argument("--robustness-only",action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        if not args.skip_prices and not args.robustness_only:
            build_prices(output)
        if args.prices_only:
            return
        source = signals(output)
        with np.load(output / "price_cube.npz") as archive:
            cube = {name:archive[name] for name in archive.files}
        if args.robustness_only:
            grid = pd.read_csv(output / "full_grid.csv")
            ledger = pd.read_csv(output / "timing_selection_ledger.csv")
            part = ledger.loc[ledger.candidate.ge(0),["signal","book","variant","clock","hold"]].drop_duplicates()
            recipes = set((str(r.signal),str(r.book),str(r.variant),str(r.clock),int(r.hold)) for r in part.itertuples())
        else:
            grid = complete_grid(output,source,cube)
            metrics,recipes = causal_replay(output,source,cube)
        if not args.skip_robustness:
            robustness(output,source,cube,grid,recipes)
        manifest = {"completed_utc":datetime.now(timezone.utc).isoformat(),"clocks":CLOCKS,"holds_minutes":HOLDS,
            "zero_hold_means":"to 15:55 Chicago session-close execution proxy","clock_offsets_minutes":OFFSETS,
            "signal_classes":VARIANTS,"fitting_classes":PRIMARY_VARIANTS,
            "forward_fill_tolerance_minutes":10,"price_grid_minutes":5,
            "primary_prices":"Cached bar opens with positive volume or bar_count; activity indication, not spread/depth-confirmed fills; same delivery month and price instrument on exit",
            "costs":"$2.16 per micro round trip, no standard-lot multiplication; one exchange tick per side except SIL $0.01/oz (two $0.005 ticks) per side",
            "selection_fitness":"Median stressed (2x friction) SR at offsets 0,+5,+15; positive means in both training halves",
            "training_requirements":"126 calendar panel sessions, 60 observations and 55% coverage; one session embargo",
            "signal_lags":"0=fresh prior-session close forecast; 1=one observation older/daily-held convention; 2=two older",
            "primary_inputs":"Previous observed same-contract volume/OI with current completed daily prices",
            "settled_reference":"Historical publication vintages unavailable: finalized-flow fresh results are not a verified tradable simulation",
            "gross_ceiling":1,"change_cap":"Ex-ante sum(abs(change)) above 1 is scaled down; no additional leverage",
            "monthly_fit_sessions":21,"weekly_fit_sessions":5,"lookbacks":(252,504,"expanding"),
            "frozen_train_test_fractions":((.3,.7),(.4,.6),(.5,.5)),"new_market_api_calls":False,
            "database_writes":False,"production_changes":False,"missing_outcomes":"NaN, never zeroed or price-imputed",
            "historical_roster_selection_bias":"Existing ten concepts and this research agenda were informed by earlier studies; chronological OOS here is not pristine never-seen holdout",
            "outputs_sha256":{p.name:digest(p) for p in output.glob("*.csv")}}
        save_json(output / "study_manifest.json",manifest)
        print("[complete]",output,flush=True)


if __name__ == "__main__":
    main()
