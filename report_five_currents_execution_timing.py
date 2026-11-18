"""Build an embedded, reproducible research report without changing the live dashboard."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from five_currents_execution_timing import CLOCKS,HOLDS,ROOTS
from five_currents_execution_timing import block_interval,summarize_matrix
from prepare_five_currents_execution_timing import DEFAULT_OUT,save_json


NAMES={"obv_divergence":"OBV divergence","carry_oi_reversal":"Carry / OI reversal",
    "curve_quality_reversal":"Curve quality reversal","carry_confirmed_momentum":"Carry-confirmed momentum",
    "aggregate_volume_carry":"Aggregate-volume carry","volume_dryup_continuation":"Volume dry-up continuation",
    "oi_direction_reversion":"OI directional reversion","carry_basis_reversal":"Carry / basis reversal",
    "volume_spike_continuation":"Volume-spike continuation","volume_spike_reversion":"Volume-spike reversion",
    "portfolio_net":"Combined daily signal","equal_ten_intraday":"Equal ten-family intraday research book"}
CLOCK_LABEL={"cme_open":"CME reopen","early_utc":"06:00/07:00 UTC","london_open":"London open",
    "london_close":"London close","us_open":"US cash open","settlement":"Product settlement","us_close":"US cash close"}
VARIANT_LABEL={"net_magnitude":"Position / size","change_magnitude":"Change / size",
    "reverse_net_magnitude":"Reversed position / size","reverse_change_magnitude":"Reversed change / size",
    "net_sign":"Position / sign","change_sign":"Change / sign","reverse_net_sign":"Reversed position / sign",
    "reverse_change_sign":"Reversed change / sign"}
COLORS=["#176b5c","#bb744b","#6f8796","#b69b4c","#8a6270"]


def label(signal):
    return ("Held: " if signal.startswith("held__") else "Fresh lagged-flow: ")+NAMES.get(signal.removeprefix("held__"),signal)


def fmt(value,kind="number"):
    if pd.isna(value):return "--"
    if kind=="pct":return f"{value*100:.2f}%"
    if kind=="bps":return f"{value*10000:+.2f}"
    if kind=="int":return f"{value:,.0f}"
    return f"{value:.2f}" if isinstance(value,(float,np.floating)) else html.escape(str(value))


def table(frame,columns,limit=None):
    if limit:frame=frame.head(limit)
    head="".join("<th>"+html.escape(title)+"</th>" for _,title,_ in columns)
    body="".join("<tr>"+"".join("<td>"+fmt(row.get(key),kind)+"</td>" for key,_,kind in columns)+"</tr>"
                 for row in frame.to_dict("records"))
    return '<div class="table"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def image(output,name,title,caption=""):
    data=base64.b64encode((output / "figures" / name).read_bytes()).decode()
    return '<figure><h3>'+html.escape(title)+'</h3><img alt="'+html.escape(title)+'" src="data:image/png;base64,'+data+'"><figcaption>'+caption+'</figcaption></figure>'


def savefig(output,name,fig):
    fig.savefig(output / "figures" / name,dpi=145,bbox_inches="tight",facecolor="#faf8f2")
    plt.close(fig)


def equity(values):
    return (1+pd.Series(np.asarray(values,dtype=float))).cumprod()*100


def line_panel(ax,frame,field="net",name=""):
    ax.plot(pd.to_datetime(frame.session),equity(frame[field]),label=name,lw=1.5)
    if "calibrating" in frame and frame.calibrating.any() and not getattr(ax,"_research_calibration_shaded",False):
        ax.axvspan(pd.to_datetime(frame.session).min(),pd.to_datetime(frame.loc[frame.calibrating,"session"]).max(),
                   color="#6f8796",alpha=.08,lw=0)
        ax._research_calibration_shaded=True
    ax.grid(axis="y",alpha=.15)
    ax.spines[["top","right"]].set_visible(False)
    locator=mdates.AutoDateLocator(minticks=3,maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def heatmap(ax,matrix,xlabels,ylabels,title,*,limit=2.5):
    ax.set_facecolor("#ece9df")
    data=np.asarray(matrix,float)
    plot=ax.imshow(np.ma.masked_invalid(data),aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit)
    ax.set_xticks(range(len(xlabels)),xlabels,rotation=35,ha="right")
    ax.set_yticks(range(len(ylabels)),ylabels)
    ax.set_title(title,loc="left",fontweight="bold",pad=12)
    for i,j in np.ndindex(data.shape):
        if np.isfinite(data[i,j]):
            ax.text(j,i,f"{data[i,j]:.1f}",ha="center",va="center",fontsize=8,
                    color="white" if abs(data[i,j])>limit*.65 else "#172e2b")
    ax.spines[:].set_visible(False)
    return plot


def make_figures(output,data,history):
    (output / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":"#faf8f2","figure.facecolor":"#faf8f2",
                         "axes.labelcolor":"#203a35","text.color":"#203a35","axes.titlepad":12,"font.size":9})
    fig,ax=plt.subplots(figsize=(11,3.6))
    ax.plot(pd.to_datetime(history["dates"]),history["portfolio"],color=COLORS[0],lw=1.7)
    ax.axvspan(pd.Timestamp("2024-06-20"),pd.Timestamp("2026-10-22"),color=COLORS[2],alpha=.1)
    ax.set_title("The daily core is unchanged; execution research uses the available intraday history",loc="left",fontweight="bold")
    ax.set_ylabel("Growth of 100, modelled net")
    ax.grid(axis="y",alpha=.15)
    ax.spines[["top","right"]].set_visible(False)
    savefig(output,"daily_core.png",fig)

    wf=data["paths"]
    for book in ("four","five"):
        fig,axes=plt.subplots(1,2,figsize=(12,3.7),sharey=True)
        for ax,signal in zip(axes,("portfolio_net","held__portfolio_net")):
            for fit in ("rolling252_monthly","rolling504_monthly","rolling252_weekly","expanding_monthly"):
                part=wf.loc[wf.signal.eq(signal)&wf.book.eq(book)&wf.fit.eq(fit)].sort_values("session")
                line_panel(ax,part,name=fit.replace("rolling","").replace("_"," "))
            ax.set_title("Fresh, lagged flows" if not signal.startswith("held__") else "Older settled / daily-held convention",loc="left")
            ax.legend(frameon=False,fontsize=7)
        axes[0].set_ylabel("Observed-return growth of 100")
        savefig(output,"causal_"+book+".png",fig)

    family=data["wf"].loc[data["wf"].fit.eq("rolling252_monthly")&data["wf"].book.isin(ROOTS)]
    names=list(NAMES)[:10]
    for prefix,title in (("","Fresh forecasts with lagged flow inputs"),("held__","Older settled forecasts / daily-held information")):
        chosen=family.loc[family.signal.isin([prefix+n for n in names])]
        matrix=chosen.pivot(index="signal",columns="book",values="sharpe").reindex(index=[prefix+n for n in names],columns=ROOTS)
        fig,ax=plt.subplots(figsize=(8.4,6))
        im=heatmap(ax,matrix,ROOTS,[NAMES[n] for n in names],title)
        fig.colorbar(im,ax=ax,pad=.02,label="Causal net Sharpe")
        savefig(output,"families_"+("held" if prefix else "fresh")+".png",fig)

    coverage=data["coverage"]
    subset=coverage.loc[coverage.offset.eq(0)&coverage.hold.eq(60)].copy()
    all_counts=subset.groupby(["root","clock"]).sessions.sum()
    matched=subset.loc[subset.status.eq("matched")].groupby(["root","clock"]).sessions.sum()
    fractions=(matched/all_counts).unstack("clock").reindex(index=ROOTS,columns=CLOCKS)*100
    fig,ax=plt.subplots(figsize=(10,3.7))
    im=ax.imshow(fractions,aspect="auto",cmap="YlGnBu",vmin=0,vmax=100)
    ax.set_yticks(range(5),ROOTS)
    ax.set_xticks(range(len(CLOCKS)),[CLOCK_LABEL[c] for c in CLOCKS],rotation=28,ha="right")
    for i,j in np.ndindex(fractions.shape):
        value=fractions.iloc[i,j]
        if np.isfinite(value):ax.text(j,i,f"{value:.0f}%",ha="center",va="center",color="white" if value>65 else "#203a35")
    ax.spines[:].set_visible(False)
    fig.colorbar(im,ax=ax,pad=.02,label="Active-bar entry + exit coverage (%)")
    savefig(output,"coverage.png",fig)

    grid=data["grid"]
    for root in ROOTS:
        fig,axes=plt.subplots(1,2,figsize=(12,4.6))
        for ax,variant in zip(axes,("net_magnitude","reverse_change_magnitude")):
            part=grid.loc[grid.signal.eq("portfolio_net")&grid.book.eq(root)&grid.variant.eq(variant)&grid.hold.gt(0)]
            matrix=part.pivot(index="clock",columns="hold",values="sharpe").reindex(index=CLOCKS,columns=HOLDS[:-1])
            im=heatmap(ax,matrix,[f"{h/60:g}h" for h in HOLDS[:-1]],[CLOCK_LABEL[c] for c in CLOCKS],VARIANT_LABEL[variant])
        savefig(output,"atlas_"+root+".png",fig)

    capture=data["capture"].loc[data["capture"].offset.eq(0)&data["capture"].book.isin(ROOTS)]
    matrix=capture.pivot(index="book",columns="clock",values="mean_daily").reindex(index=ROOTS,columns=CLOCKS)*10000
    fig,ax=plt.subplots(figsize=(10,3.7))
    limit=max(1.,float(np.nanmax(np.abs(matrix.to_numpy()))))
    im=heatmap(ax,matrix,[CLOCK_LABEL[c] for c in CLOCKS],ROOTS,"Rebalance timing capture, fixed quantity; basis points per session",limit=limit)
    fig.colorbar(im,ax=ax,pad=.02)
    savefig(output,"capture.png",fig)

    decomposition=data["decomposition"].loc[data["decomposition"].signal.eq("held__portfolio_net")&data["decomposition"].book.isin(ROOTS)]
    intervals=decomposition.interval.drop_duplicates().tolist()
    matrix=decomposition.pivot(index="interval",columns="book",values="mean_daily").reindex(index=intervals,columns=ROOTS)*10000
    fig,ax=plt.subplots(figsize=(12,4.4))
    matrix.plot.bar(ax=ax,color=COLORS,width=.78)
    ax.set_xticklabels([s.replace(" -> ","\nto\n").replace("_"," ") for s in intervals],rotation=0,fontsize=7)
    ax.axhline(0,color="#203a35",lw=.6)
    ax.set_ylabel("Mean gross contribution (bps / session)")
    ax.legend(frameon=False,ncol=5)
    ax.spines[["top","right"]].set_visible(False)
    savefig(output,"decomposition.png",fig)

    latest=data["latest"]
    eligible=latest.loc[latest.candidate.ge(0)&latest.oos_sharpe.notna()]
    selected=eligible.sort_values("oos_sharpe",ascending=False).groupby("asset").head(1)
    robust=data["robust"]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for row in selected.itertuples():
        base=robust.loc[robust.signal.eq(row.signal)&robust.book.eq(row.asset)&robust.variant.eq(row.variant)
            &robust.clock.eq(row.clock)&robust.hold.eq(row.hold)&robust.policy.isna()]
        delay=base.loc[base.lag.eq(0)].sort_values("offset")
        axes[0].plot(delay.offset,delay.paired_sharpe,marker="o",label=row.asset)
        lag=base.loc[base.offset.eq(0)].sort_values("lag")
        axes[1].plot(lag.lag,lag.paired_sharpe,marker="o",label=row.asset)
    for ax in axes:
        ax.axhline(0,color="#203a35",lw=.7)
        ax.grid(axis="y",alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_xlabel("Minutes relative to regional event")
    axes[1].set_xlabel("Additional signal observation lags")
    axes[0].set_ylabel("Net Sharpe on paired dates")
    axes[0].set_title("Clock perturbations",loc="left")
    axes[1].set_title("Information delays",loc="left")
    savefig(output,"robustness.png",fig)
    selected.to_csv(output / "illustrative_candidates.csv",index=False)
    cases=[("GC","held__curve_quality_reversal","regional","Gold / older curve quality / regional blocks"),
           ("GC","curve_quality_reversal","fixed","Gold / fresh curve quality / fixed holds"),
           ("HG","volume_dryup_continuation","fixed","Copper / fresh volume dry-up / fixed holds"),
           ("CL","held__volume_dryup_continuation","regional","Crude / older volume dry-up / regional blocks")]
    fig,axes=plt.subplots(2,2,figsize=(12,7.3))
    for ax,(book,signal,kind,title) in zip(axes.ravel(),cases):
        source=data["regional_paths"] if kind=="regional" else wf
        for fit in ("rolling252_monthly","rolling504_monthly","rolling252_weekly"):
            part=source.loc[source.book.eq(book)&source.signal.eq(signal)&source.fit.eq(fit)].sort_values("session")
            line_panel(ax,part,name=fit.replace("rolling","").replace("_"," "))
        ax.set_title(title,loc="left",fontsize=9)
        ax.legend(frameon=False,fontsize=7)
    savefig(output,"case_curves.png",fig)
    stress=data["regional_stress"].loc[data["regional_stress"].signal.eq("held__curve_quality_reversal")
        &data["regional_stress"].book.eq("GC")&data["regional_stress"].fit.eq("rolling252_monthly")]
    fig,ax=plt.subplots(figsize=(9,3.7))
    for lag,part in stress.groupby("lag"):
        part=part.sort_values("offset")
        ax.plot(part.offset,part.paired_sharpe,marker="o",label=f"Additional {lag} signal-day lag")
    ax.axhline(0,color="#203a35",lw=.7)
    ax.set_xlabel("Execution delay in minutes, with all original timing decisions frozen")
    ax.set_ylabel("Paired-date net Sharpe")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    savefig(output,"frozen_oos_delays.png",fig)
    selected=data["regional_paths"].loc[data["regional_paths"].signal.eq("held__curve_quality_reversal")
        &data["regional_paths"].book.eq("GC")&data["regional_paths"].fit.eq("rolling252_monthly")].sort_values("session")
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    x=selected.net.to_numpy()
    finite=np.flatnonzero(np.isfinite(x)&~selected.calibrating.to_numpy())
    for n in (0,1,5):
        value=x.copy()
        if n:value[finite[np.argsort(x[finite])[-n:]]]=0.
        axes[0].plot(selected.session,equity(value),label="Actual OOS" if not n else f"Miss best {n} day(s)")
    axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_ylabel("Observed-return growth of 100")
    locator=mdates.AutoDateLocator(minticks=3,maxticks=6)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    control=data["direction"].loc[data["direction"].signal.eq("held__curve_quality_reversal")&data["direction"].book.eq("GC")].iloc[0]
    axes[1].bar(["Actual signal","Always long","Always short"],
        [control.actual_sharpe,control.always_long_same_exposure_sharpe,control.always_short_same_exposure_sharpe],
        color=[COLORS[0],COLORS[1],COLORS[2]],width=.6)
    axes[1].set_ylabel("Net Sharpe / identical timing and absolute exposure")
    for ax in axes:
        ax.grid(axis="y",alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    savefig(output,"concentration_and_controls.png",fig)


def report(output):
    data={name:pd.read_csv(output / path) for name,path in {
        "grid":"full_grid.csv","wf":"walk_forward_metrics.csv","coverage":"price_coverage.csv",
        "latest":"latest_causal_timing_candidates.csv","robust":"delay_lag_cost_robustness.csv",
        "capture":"rebalance_timing_capture.csv","decomposition":"session_return_decomposition.csv",
        "yearly":"walk_forward_yearly.csv","recipe_yearly":"recipe_yearly.csv",
        "regional":"regional_block_causal_metrics.csv","regional_stress":"regional_frozen_schedule_stress.csv",
        "personalized":"personalized_vs_shared_common_dates.csv",
        "direction":"regional_direction_controls.csv","concentration":"regional_oos_concentration_checks.csv"}.items()}
    data["paths"]=pd.read_parquet(output / "walk_forward_paths.parquet")
    data["regional_paths"]=pd.read_parquet(output / "regional_block_causal_paths.parquet")
    shared=data["paths"].loc[data["paths"].signal.str.contains("equal_ten")&data["paths"].fit.eq("rolling252_monthly")]
    personal=pd.read_parquet(output / "asset_specific_portfolio_paths.parquet")
    personal=personal.loc[personal.fit.eq("rolling252_monthly")]
    friction_rows=[]
    for (signal,book),group in pd.concat([shared,personal]).groupby(["signal","book"]):
        sample=group.loc[~group.calibrating]
        common=np.isfinite(sample.net)&np.isfinite(sample.gross)
        gross=summarize_matrix(sample.loc[common,"gross"])
        net=summarize_matrix(sample.loc[common,"net"])
        friction_rows.append({"signal":signal,"book":book,"observed_days":int(common.sum()),
            "gross_sharpe":gross["sharpe"][0],"net_sharpe":net["sharpe"][0],
            "observed_gross_return":gross["observed_total_return"][0],"observed_net_return":net["observed_total_return"][0]})
    friction=pd.DataFrame(friction_rows)
    friction.to_csv(output / "research_ensemble_friction_sensitivity.csv",index=False)
    manifest=json.loads((output / "source_manifest.json").read_text())
    history=json.loads((Path(manifest["source_snapshot"]) / "client.json").read_text())["history"]
    save_json(output / "inputs/daily_dashboard_history.json",history)
    make_figures(output,data,history)
    wf=data["wf"]
    core=wf.loc[wf.signal.isin(("portfolio_net","held__portfolio_net"))&wf.book.isin(("four","five"))].copy()
    core["timeline"]=np.where(core.signal.str.startswith("held__"),"Older settled / held","Fresh / lagged flows")
    core["fit"]=core.fit.str.replace("rolling","").str.replace("_"," ")
    cols=[("timeline","Signal timeline","text"),("book","Markets","text"),("fit","Timing fit","text"),
          ("selected_sessions","Deployed recipe days","int"),
          ("days","Observed days","int"),("coverage","Coverage","pct"),("sharpe","Net SR","number"),
          ("stressed_sharpe","2x cost SR","number"),("observed_total_return","Observed return","pct"),
          ("observed_max_drawdown","Observed DD","pct")]
    latest=data["latest"].copy()
    latest["alpha"]=latest.signal.map(label)
    latest["clock"]=latest.clock.map(CLOCK_LABEL)
    latest["variant"]=latest.variant.map(VARIANT_LABEL)
    latest["hold_hours"]=latest.hold/60
    candidates=latest.loc[latest.assessment.str.startswith("Paper-trade",na=False)].sort_values("oos_sharpe",ascending=False)
    count=len(candidates)
    bestcore=core.loc[core.fit.eq("252 monthly")&core.book.eq("four")]
    narrative="No asset/family clears the cross-check screen; retain the daily core and treat individual clock winners as exploratory." if not count else (
        f"{count} asset/family timeline cells pass the simple cross-check screen. These are paper-trade candidates, not {count} independent alphas or an optimized production portfolio.")
    fee_audit=json.loads((output / "arithmetic_audit.json").read_text())
    decomp_audit=json.loads((output / "session_decomposition_audit.json").read_text())
    evidence=[]
    for asset,signal,kind,assessment in [
        ("GC","held__curve_quality_reversal","regional","Strongest consistency; recent exact-month active validation insufficient"),
        ("GC","curve_quality_reversal","fixed","Positive timing plateau; activity is dormant in the recent exact-month slice"),
        ("GC","held__carry_basis_reversal","fixed","Secondary paper candidate; weaker under long lookbacks"),
        ("HG","volume_dryup_continuation","fixed","Weekly fit and 50:50 split fail: cadence-sensitive"),
        ("CL","held__volume_dryup_continuation","regional","252/weekly positive; longer windows and frozen splits disagree")]:
        source=data["regional"] if kind=="regional" else data["wf"]
        part=source.loc[source.book.eq(asset)&source.signal.eq(signal)].set_index("fit")
        row=part.loc["rolling252_monthly"]
        series=data["regional_paths"] if kind=="regional" else data["paths"]
        path=series.loc[series.book.eq(asset)&series.signal.eq(signal)&series.fit.eq("rolling252_monthly")]
        bounds=block_interval(path.loc[~path.calibrating,"net"].to_numpy(),draws=1000)
        evidence.append({"asset":asset,"alpha":label(signal),"grid":kind,"net_sr":row.sharpe,
            "stress_sr":row.stressed_sharpe,"weekly_sr":part.loc["rolling252_weekly","sharpe"],
            "long_sr":part.loc["rolling504_monthly","sharpe"],
            "30_70":part.loc["frozen_30_70","sharpe"],"40_60":part.loc["frozen_40_60","sharpe"],
            "50_50":part.loc["frozen_50_50","sharpe"],"ci_low":bounds["sharpe_low"],"ci_high":bounds["sharpe_high"],
            "observed_return":row.observed_total_return,"observed_dd":row.observed_max_drawdown,
            "assessment":assessment})
    evidence=pd.DataFrame(evidence)
    evidence.to_csv(output / "key_evidence.csv",index=False)
    portions=[]
    portions.append('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Five Currents | When The Edge Trades</title><style>
    :root{--ink:#203a35;--paper:#faf8f2;--green:#176b5c;--clay:#bb744b;--line:#d8d8cb;--muted:#63736b}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 Georgia,'Palatino Linotype',serif}main{max-width:1200px;margin:auto;padding:44px 40px}h1{font-size:52px;line-height:1.05;margin:12px 0 20px;letter-spacing:-2px}h2{font-size:29px;margin:48px 0 14px;border-top:1px solid var(--line);padding-top:24px}h3{font-size:18px;margin:10px 0}.kicker,th,.tag{font-family:Verdana,sans-serif;text-transform:uppercase;letter-spacing:1.5px;font-size:10px}.kicker{color:var(--green)}.intro{font-size:19px;max-width:900px}.note{border-left:4px solid var(--clay);padding:13px 20px;background:#f0eadc}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0}.card{border:1px solid var(--line);padding:20px}.card strong{display:block;font-size:31px;color:var(--green)}.quiet,figcaption{font-family:Verdana,sans-serif;font-size:11px;color:var(--muted);line-height:1.6}figure{margin:25px 0;border:1px solid var(--line);padding:18px;background:#fffef9;break-inside:avoid}img{width:100%;height:auto}.table{overflow:auto;max-height:650px;border:1px solid var(--line);margin:20px 0}table{width:100%;border-collapse:collapse;font:11px/1.4 Verdana,sans-serif}th{text-align:left;position:sticky;top:0;background:#e7ece4;letter-spacing:.3px;font-size:9px}td,th{padding:10px;border-bottom:1px solid var(--line)}td{white-space:nowrap}details{padding:14px 0}summary{cursor:pointer;font-weight:bold}code{font:12px Consolas,monospace}.formula{background:#e9eee7;padding:14px;font-family:Consolas,monospace;white-space:normal}a{color:var(--green)}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}footer{border-top:1px solid var(--line);margin-top:35px;padding-top:15px}@media(max-width:700px){main{padding:24px 16px}h1{font-size:36px}.cards,.two{grid-template-columns:1fr}figure{padding:8px}}@media print{h2{break-before:page}.table{max-height:none;overflow:visible}details{display:block}body{font-size:11px}main{padding:0}h1{font-size:38px}}
    </style></head><body><main><div class="kicker">Five Currents / Execution Research / 23 October 2026</div><h1>When The Edge Trades</h1><p class="intro">A regional execution atlas for the existing ten daily alpha families across crude oil, natural gas, gold, silver and copper.</p>''')
    portions.append('<div class="cards"><div class="card"><span class="tag">Unchanged daily core, 2020-2026</span><strong>'+fmt(history["metrics"]["sharpe"])+
        '</strong>Modelled net Sharpe</div><div class="card"><span class="tag">Intraday source span</span><strong>2024-2026</strong>20 June 2024 - 22 October 2026</div><div class="card"><span class="tag">Daily families</span><strong>10</strong>Two causal information timelines; no new alpha definitions</div></div>')
    portions.append('<div class="note"><b>Read the causal results before the clock winners.</b> '+narrative+
        ' The original daily core remains unchanged. No trading orders, collector changes or new market-data requests were made.</div>')
    portions.append('<h3>Research conclusion</h3><p><b>The strongest consistency is in gold, not a uniform all-market intraday overlay.</b> The older curve-quality forecast, traded in reverse from CME reopen toward London open, has a 1.09 causal net Sharpe and 1.04 at doubled costs. Its frozen split checks remain positive. Crude-oil and copper volume sleeves can look equally strong in one fit but fail important cadence or split checks. Silver lacks enough matched recent minute data; natural gas has no comparably consistent result.</p><p>These findings support a small, asset-specific paper programme, not a wholesale change to the daily core. Gold curve quality currently has little usable active validation in the recent exact-month sample, so even the leading historical candidate is not an immediate deployment mandate.</p>')
    portions.append('<div class="note"><b>The leading gold result is tail-concentrated.</b> Removing its five best OOS days changes the net Sharpe from 1.09 to -0.13 and the observed return from +5.41% to -0.33%. This does not prove the payoff is spurious, but it prevents a claim of a broadly reliable smooth intraday income stream.</div>')
    portions.append(image(output,"daily_core.png","The interday benchmark is preserved","The full daily history is shown. The shaded interval is the available intraday study span, not an erased calibration period. Daily performance uses the archived dashboard\'s actual mark-session dates."))
    portions.append('<h2>1. What Was Tested</h2><div class="two"><div><h3>Signal families</h3><p>'+', '.join(NAMES[n] for n in list(NAMES)[:10])+'.</p><p>Each family is examined standalone and the frozen combined daily signal is examined separately. Family-level results are not a sum-to-portfolio performance attribution.</p></div><div><h3>Execution grid</h3><p>CME reopen; 06:00/07:00 UTC; London 08:00 and 16:30; New York 09:30 and 16:00; and each product\'s settlement-window end. Clocks follow their own regional daylight-saving rules.</p><p>Holds: 30, 60, 90, 120, 180, 240, 360 and 720 minutes, plus a separate session-close timing diagnostic. A trade may not run through maintenance.</p></div></div>')
    portions.append('<p><b>Fresh / lagged-flow:</b> the last completed daily prices with previous-observation, same-contract volume/OI. <b>Older settled / held:</b> the fully settled forecast one additional model observation older, matching the daily-held information convention. The final-flow fresh reference is availability-assumed and excluded from the primary timing fits.</p>')
    portions.append('<p class="quiet">Timeline: the original daily forecast from close T is implemented at close T+1 and earns through close T+2. During session T+1, the already-held signal is therefore one forecast older than the newly available T forecast. Both are tested explicitly. Same-session information is never silently substituted for the older held position.</p>')
    portions.append('<div class="formula">Position = w(T); change = w(T) - w(T-1); reversal = -position or -change.<br>Gross P&amp;L = signed micro quantity x micro multiplier x (exit - entry).<br>Net P&amp;L = gross - $2.16 per micro round trip - modelled slippage on each side.</div>')
    portions.append('<p class="quiet">Sign-only versions are also reported. Every style has an ex-ante gross ceiling of one: large position changes are reduced, never levered up. Fractional quantities express returns per unit capital, not a whole-lot order proposal. Base slippage is one exchange tick per side except SIL, where the $0.01/oz allowance is two $0.005 minimum ticks. Standard-contract price proxies do not change the micro commission. Reversing an already-reversing family means reversing its signed vote, not automatically shorting the commodity.</p>')
    portions.append('<h2>2. Causal Timing Selection</h2><p>Timing is selected on trailing information, not by choosing the best full-history cell. The main fit is 252 sessions / 21-session refits, expanding from a 126-session minimum. Prespecified checks use 504 sessions, weekly refits and an expanding window. A complete prior session is embargoed.</p><p>Candidate fitness is the median net Sharpe across 0, +5 and +15 minute execution offsets with <b>doubled friction</b>. Both halves of the training window must have positive mean P&amp;L. Low-coverage candidates and candidates without enough trades are rejected; cash is permitted. Independent frozen 30:70, 40:60 and 50:50 train/test checks test whether timing choices survive a long subsequent period.</p>')
    portions.append(table(core,cols))
    portions.append(image(output,"causal_four.png","Timing the combined signal: four markets","Silver is excluded only in this sensitivity panel. Signal parameters, original daily selection decisions and product-risk coefficients are unchanged; these fits select execution recipes only. Flat initial calibration is prescribed cash, not in-sample profit."))
    portions.append(image(output,"causal_five.png","Timing the combined signal: all five markets","Missing active-leg outcomes remain missing. Curves compound observed outcomes and retain visible gaps; they are not a claim of a complete executable equity history where coverage is incomplete."))
    portions.append('<p>For the all-five combined signal, the conservative timing learner rejects every recipe in these replays. Its flat trace is cash after failed training/coverage admission, not proof of zero market returns. Matched input coverage and enough positive training observations are required before an execution time can be inferred.</p>')
    portions.append('<h3>Natural regional blocks: a smaller timing model</h3><p>A separate check replaces the holding-period grid with just four economic windows: CME reopen to London open; London open to US open; US open to product settlement; settlement to US close. Both ends follow their local clocks, including UK/US DST mismatch weeks. No independent holding period is optimized in this version.</p>')
    evidence_cols=[("asset","Asset","text"),("alpha","Family / timeline","text"),("grid","Timing model","text"),
        ("net_sr","252 monthly SR","number"),("stress_sr","2x cost SR","number"),("weekly_sr","252 weekly SR","number"),
        ("long_sr","504 monthly SR","number"),("30_70","30:70 SR","number"),("40_60","40:60 SR","number"),
        ("50_50","50:50 SR","number"),("ci_low","Conditional 95% low","number"),("ci_high","Conditional 95% high","number")]
    portions.append(table(evidence,evidence_cols))
    portions.append('<p class="quiet">Block-bootstrap intervals condition on the completed OOS return path; they do not correct for the historical choice of the ten-family research universe. Longer-window and expanding fits overlap heavily and are not independent confirmations. A dash may mean prescribed cash, insufficient variability or unavailable observations, not a zero Sharpe.</p>')
    portions.append(image(output,"case_curves.png","The important contrast: stable timing versus cadence-sensitive winners","All curves are causal timing replays across the full intraday span, with prescribed calibration cash shown. A strong result in only one lookback is not treated as a stable execution rule."))
    portions.append(image(output,"concentration_and_controls.png","A real signal distinction, but a concentrated payoff","Gold / older curve quality / regional timing. Actual 1.09 SR versus -0.19 always-long and +0.09 always-short controls with identical timing, absolute exposure and costs. The left panel is an ex-post missed-profit-day diagnostic, not an implementable removal rule. The dominant profit on 23 March 2026 corresponds to a same-contract overnight price decline, not a roll jump."))
    portions.append('<h2>3. Where The Family Evidence Survives</h2><p>These heatmaps show complete chronological timing replays, not the best fixed clock or holding period picked after seeing the full sample. Values include commission and slippage. A strong-looking full-history cell is insufficient if its timing choice fails out of sample.</p>')
    portions.append(image(output,"families_fresh.png","Family x asset: causal timing, fresh lagged-flow forecasts"))
    portions.append(image(output,"families_held.png","Family x asset: causal timing, older settled forecasts"))
    candidate_cols=[("asset","Asset","text"),("alpha","Information / family","text"),("clock","Latest fitted clock","text"),
        ("hold_hours","Hours","number"),("variant","Signal class","text"),("oos_sharpe","OOS SR","number"),
        ("oos_stressed_sharpe","2x cost SR","number"),("frozen_positive","Positive frozen checks / 3","int"),
        ("alternate_positive","Positive alternative fits / 3","int"),("oos_coverage","Coverage","pct"),
        ("verified_active_observations","Recent verified active observations","int")]
    portions.append('<h3>Cross-check shortlist</h3><p>This is a diagnostic screen across the completed replays, not a hindsight filter reapplied to manufacture a new equity curve. A row needs positive main and stressed OOS results and at least two positive frozen and alternative-fit checks. The latest fitted clock can differ from recipes used earlier in its OOS path.</p>')
    portions.append(table(candidates,candidate_cols,20) if count else '<p>No rows meet the screen.</p>')
    portions.append('<details><summary>All latest fitted asset/family recipes, including cash and failed checks</summary>'+table(latest,candidate_cols+[('assessment','Assessment','text')])+'</details>')
    portions.append('<h2>4. Clock, Delay And Contract Robustness</h2>')
    portions.append(image(output,"frozen_oos_delays.png","The original OOS decisions are frozen while execution and information are delayed","Gold / older curve quality / four regional windows. No recipe is reselected to accommodate the delay. At zero extra signal lag, the net Sharpe is approximately 1.09, 1.06, 1.00 and 1.02 at 0, 5, 15 and 30 minutes. One and two extra daily observation lags remain positive. Exact paired-date counts are exported."))
    portions.append(image(output,"robustness.png","Do the effects survive a different clock or an older forecast?","Illustrative latest-fitted recipes, selected for display by their asset-level causal replay. These historical diagnostics are not additional production selections. Delay comparisons use paired observable dates. A one-day signal lag tests freshness, not merely execution latency."))
    portions.append('<p>Selected fixed-duration recipes also have checks at -30, -15, +5, +15 and +30 minutes, a staggered 0/+5/+15 execution proxy and one or two extra signal observations. Regional blocks have 0/+5/+15/+30 offsets and the same information-lag checks. The complete OOS schedules are frozen for both delay studies. Separate years, weekday and largest-profit-day concentration checks are exported; fixed-recipe exact-month, micro-only and expiry/delivery exclusions are also provided.</p>')
    portions.append('<p>The fixed-hold gold curve-quality diagnostic is also positive in US summer and winter and after excluding UK/US DST mismatch weeks. It remains positive after removing its five largest profit days. However, there are no active matched observations for its latest recipe in the recent verified-month slice: a good historical curve is not a substitute for current forward validation.</p>')
    portions.append('<p><b>Trading interpretation:</b> a broad positive timing plateau is more credible than a single sharp spike. A benefit that disappears with +5 or +15 minutes belongs on a watchlist, not in an execution mandate. A signal that decays over a whole day may still be a genuine short-lived alpha; that is different from an effect that vanishes after a few minutes.</p>')
    portions.append('<h2>5. Interday Rebalancing Is Not A Round-Trip Overlay</h2><p>There are two distinct implementation questions. Moving an already-required rebalance to an earlier regional window shifts the same order; it does not create another entry and exit. A separate intraday position does create a round trip and must clear its full additional costs.</p>')
    portions.append(image(output,"capture.png","Advancing or delaying the daily rebalance: a separate opportunity-cost diagnostic","Fixed-order-size comparison against a same-contract settlement-window bar proxy, restricted to non-roll dates and matched price instruments. Equal per-lot commission and tick slippage cancel in the timing difference. This is not a rerun of the full production book, nor a new strategy return stream to add to the core."))
    portions.append(image(output,"decomposition.png","Where a held signal earns or loses during the session","Seven disjoint windows use the same contract, price instrument and fixed quantity throughout. Gross contributions telescope exactly to the full same-contract session. No fictitious commission is charged between adjacent windows. Each asset uses its own common complete-window sample."))
    portions.append('<h2>6. The Data Can And Cannot Support</h2>')
    portions.append(image(output,"coverage.png","Actual stored active-bar coverage","One-hour holds, zero requested offset. Late US-close holds run into maintenance and are intentionally invalid. Activity means positive stored volume or bar count; it is not spread/depth-confirmed execution. Bars without either indication are excluded from primary fills. Bars are sampled at five-minute spacing; the first eligible bar up to ten minutes after the request is used, never an earlier one."))
    portions.append('<p>Earlier daily delivery mappings are inferred; exact mappings form a much shorter recent subset. The selection mask uses only prior daily liquidity and never switches contracts because a later exit happens to be present. Silver minute history ends in late September. Missing price pairs are not downloaded, interpolated, reclassified as cash or silently replaced with another month.</p><p>Annualized Sharpe uses observed daily outcomes. Where outcomes are missing, cumulative return and drawdown are explicitly <b>observed-return</b> statistics, not a certified complete account path. Calendar-return forecasts and dollar capacity claims are not inferred from sparse minute samples.</p>')
    portions.append('<h2>7. Asset Execution Atlas</h2><p>These fixed-rule surfaces are descriptive full-history diagnostics. They deliberately show both a position and a reversed position-change class; all eight classes, all ten families and every horizon are in the downloadable grid. Do not replace the daily portfolio with the best-coloured cell.</p>')
    for root in ROOTS:
        portions.append(image(output,"atlas_"+root+".png",root+" / fixed-rule clock and horizon surface","Fresh lagged-flow combined signal. Net Sharpe after micro commission and tick slippage. Grey cells are unavailable or invalid, not flat performance."))
    portions.append('<h2>8. Implementation Conclusion</h2><p>Keep the interday core as the strategic book. A timing change should first be shadowed as an execution change to that book, with the same quantities and risk limits. Treat a reversed or flat-at-exit intraday position as an additional alpha sleeve, with its own cost and risk budget, not as a reason to reverse the strategic exposure.</p><p>Use regional time zones, an explicit information-ready gate and a time range or staggered schedule rather than a single hard-coded UTC minute. Contract choice remains a prior-liquidity decision. Verify the actual tradable micro spread and depth during paper trading; a parent-price proxy cannot establish micro fill quality.</p><p>Asset-specific recipes should only be retained if they improve on the simpler shared-family timing and survive subsequent paper observations. The candidate roster and research agenda already reflect prior historical work: chronological tests here reduce execution-selection lookahead but do not erase that research-selection bias.</p>')
    portions.append('<p><b>Practical priorities:</b> gold overnight curve-quality opposition is the clearest paper hypothesis, with explicit tail-risk and inactive-sample caveats; gold carry/basis and volume-spike sleeves are secondary. Copper\'s US-session volume effect needs a fixed cadence commitment because weekly retuning degrades it. Crude has suggestive London/US segmentation but unstable cross-fit evidence; natural gas should stay interday-only for now, and silver needs a reliable matched minute sample before an intraday conclusion.</p><p>The separate per-asset fitting of all ten families was also tested in the fixed-hold study, with a common gross ceiling and no hindsight pruning. It does not produce a convincing diversified intraday portfolio after costs. More personalized clocks do not, by themselves, solve alpha selection.</p><p>The existing prior-held forecasts can be shadowed from archived targets. A fresh lagged-flow forecast at CME reopen would require a separate close-ready calculation and timestamped data gate; changing the execution timer alone is insufficient. Any intraday addition must be netted against the strategic book and funded inside the same portfolio risk budget.</p>')
    portions.append('<details><summary>Friction sensitivity of the experimental ten-family combinations</summary><p>The combined-daily-signal tests are charged on their net position. The separate ten-family research combinations below conservatively pay independent-sleeve costs: simultaneous orders are not event-netted. The gross column removes all modelled costs, so it distinguishes weak gross selection from fee drag; it is not a live execution forecast.</p>'+table(friction,
        [("signal","Research combination","text"),("book","Markets","text"),("observed_days","Matched days","int"),
         ("gross_sharpe","Gross SR","number"),("net_sharpe","Net SR","number"),
         ("observed_gross_return","Observed gross return","pct"),("observed_net_return","Observed net return","pct")])+'</details>')
    portions.append('<h2>Audit And Downloads</h2><p>'+str(fee_audit["scalar_comparisons"])+
        ' independent scalar lot/P&amp;L checks; maximum error '+f'{fee_audit["maximum_absolute_error"]:.2g}'+
        '. Seven-window gross reconciliation maximum error '+f'{decomp_audit["maximum_absolute_error"]:.2g}'+'.</p>')
    files=[("full_grid.csv","Complete fixed-rule results: families, assets, books, signal classes and horizons"),
        ("walk_forward_metrics.csv","Causal timing-fit metrics and conditional block-bootstrap intervals"),
        ("walk_forward_paths.parquet","Complete daily equity inputs for every fixed-hold causal alpha/asset/fit replay"),
        ("timing_selection_ledger.csv","Every timing fit: training cutoff, chosen clock, holding period and signal class"),
        ("delay_lag_cost_robustness.csv","Timing offsets, signal delays, staggered fills and cost stress"),
        ("latest_causal_timing_candidates.csv","Latest fitted recipes and cross-check screen"),
        ("contract_expiry_and_proxy_checks.csv","Verified-month, micro-only and delivery/expiry sensitivity"),
        ("rebalance_timing_capture.csv","Fixed-order timing opportunity cost, separate from round-trip alpha"),
        ("session_return_decomposition.csv","Disjoint-window gross contribution reconciliation"),
        ("recipe_ablations.csv","Weekday and largest-profit-day removal"),
        ("regional_block_full_grid.csv","Four natural regional windows: all signal classes and families"),
        ("regional_fixed_rule_era_checks.csv","Fixed regional hypotheses by year and chronological tail; not pristine holdout claims"),
        ("regional_block_causal_metrics.csv","Causal regional-window selection with the same split and rolling protocol"),
        ("regional_block_causal_paths.parquet","Complete daily equity inputs for every regional-window causal replay"),
        ("regional_frozen_schedule_stress.csv","Regional OOS schedules frozen under execution and signal delays"),
        ("regional_direction_controls.csv","Always-long/short controls with identical OOS timing, absolute exposure and costs"),
        ("regional_oos_concentration_checks.csv","Complete OOS concentration, season and weekday checks without refitting"),
        ("dominant_profit_price_audit.csv","Neighboring same-contract minute observations around the largest gold profit"),
        ("key_evidence.csv","Key cross-fit evidence and conditional confidence intervals"),
        ("verification.json","Frozen input hashes, replay equivalence and report validation"),
        ("frozen_schedule_stress.csv","Fixed-hold OOS schedules frozen under execution and signal delays"),
        ("personalized_vs_shared_common_dates.csv","Asset-specific versus shared-family timing on identical dates"),
        ("research_ensemble_friction_sensitivity.csv","Experimental independent-sleeve combination before/after friction"),
        ("availability_assumed_settled_reference_grid.csv","Availability-assumed fresh final-flow reference; not certified opening-tradable"),
        ("study_manifest.json","Design, data treatment, model preservation and output hashes")]
    portions.append('<ul>'+''.join('<li><a href="'+name+'">'+title+'</a></li>' for name,title in files)+'</ul>')
    portions.append('<p class="quiet">Official references: <a href="https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457085528/Daily%2BSettlement%2BTime%2BDetails">CME settlement time details</a>; <a href="https://www.cmegroup.com/market-data/daily-bulletin.html">CME daily bulletin</a>; <a href="https://www.cmegroup.com/notices/clearing/2020/08/Chadv20-334.html">SIL minimum tick amendment</a>. Settlement-window bar opens are proxies, not guaranteed official settlement fills. Research assumptions are disclosed in the manifest.</p><footer class="quiet">Internal research. Modelled results, not brokerage executions. All charts embedded; live dashboard and collectors unchanged.</footer></main></body></html>')
    path=output / "Five_Currents_Execution_Timing_Report.html"
    path.write_text("\n".join(portions),encoding="utf-8")
    save_json(output / "report_summary.json",{"shortlist_cells":count,"daily_core_metrics":history["metrics"],
        "report":str(path),"source_snapshot":manifest["source_snapshot"],
        "top_screened_rows":candidates.head(15).replace({np.nan:None}).to_dict("records")})
    print("[report]",path,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    report(args.output_dir.resolve())
