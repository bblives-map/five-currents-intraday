"""Render released aggregate strategy returns; no market data or database required."""
from pathlib import Path
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[1]
    metadata=json.loads((root/"results/archive_metadata.json").read_text())
    returns=pd.read_csv(root/"results/strategy_returns.csv",index_col=0,parse_dates=True)
    output=root/"reports";output.mkdir(exist_ok=True)
    metrics=[]
    for name,series in returns.items():
        values=series.dropna();sd=values.std(ddof=1);curve=(1+values).cumprod()
        metrics.append({"series":name,"observations":len(values),
            "net_sharpe":float(values.mean()/sd*np.sqrt(252)) if sd>0 else None,
            "total_return":float(curve.iloc[-1]-1) if len(curve) else None,
            "max_drawdown":float((curve/curve.cummax()-1).min()) if len(curve) else None})
    figure,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[3,1]})
    for name in (name for name in ("Portfolio","Stressed") if name in returns):
        equity=100*(1+returns[name]).cumprod(skipna=True)
        axes[0].plot(equity.index,equity,label=name,lw=1.6)
        if name=="Portfolio":axes[1].plot(equity.index,100*(equity/equity.cummax()-1),lw=1.,color="#b4644b")
    axes[0].set_ylabel("Growth of 100");axes[1].set_ylabel("Drawdown %")
    axes[0].legend(frameon=False)
    for axis in axes:axis.grid(alpha=.2);axis.spines[["top","right"]].set_visible(False)
    figure.tight_layout();figure.savefig(output/"portfolio.png",dpi=145);plt.close(figure)
    assets=[name for name in ("CL","NG","GC","SI","HG") if name in returns]
    figure,axis=plt.subplots(figsize=(12,5))
    for name in assets:axis.plot(returns.index,100*(1+returns[name]).cumprod(skipna=True),label=name,lw=1.25)
    axis.legend(frameon=False,ncol=5);axis.set_ylabel("Growth of 100 / asset contributions")
    axis.grid(alpha=.2);axis.spines[["top","right"]].set_visible(False)
    figure.tight_layout();figure.savefig(output/"assets.png",dpi=145);plt.close(figure)
    table=pd.DataFrame(metrics)
    table.to_csv(output/"recomputed_metrics.csv",index=False)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(metadata['title'])}</title><style>body{{font:16px Georgia,serif;color:#163c39;background:#f6f5ef;max-width:1100px;margin:48px auto;padding:0 24px;line-height:1.55}}h1{{font-size:36px}}img{{width:100%;background:white;border-radius:10px}}table{{width:100%;border-collapse:collapse;background:white;font:13px monospace}}td,th{{padding:9px;text-align:right;border-bottom:1px solid #ddd}}td:first-child,th:first-child{{text-align:left}}.note{{padding:18px;border-left:4px solid #287b71;background:white}}</style>
<h1>{html.escape(metadata['title'])}</h1><p>{html.escape(metadata['description'])}</p>
<p>Archived observations: {returns.index.min().date()} to {returns.index.max().date()}. Figures are normalized research returns after the archived cost model, not brokerage P&amp;L.</p>
<div class="note">{html.escape(metadata['limitations'])} Missing observations remain gaps; they are not filled as zero-return observations. These are existing archived results, not a new live-data backtest.</div>
<h2>Portfolio and drawdown</h2><img src="portfolio.png" alt="Full archived portfolio equity and drawdown">
<h2>Asset contributions</h2><img src="assets.png" alt="Archived per-asset contribution curves">
<h2>Recomputed archive statistics</h2>{table.to_html(index=False,border=0,float_format=lambda value:f"{value:.4f}")}
<p>The complete original research, report generators and sensitivity tests are in the repository. Supply the private input manifest to replay them rather than treating this chart as independent validation.</p></html>'''
    (output/"archived_backtest.html").write_text(body,encoding="utf-8")
    print(output/"archived_backtest.html")


if __name__=="__main__":main()
