"""Sign-reversal sensitivity using saved returns only, without DB/API access."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_intraday_replay import HORIZONS, ROOTS, reverse_paths, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("five_currents_intraday_exit_20261009_matched"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    names = ("asset_daily_returns.parquet", "summary_metrics.csv", "equity_curves.parquet")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    for name in names:
        if hashes[name] != source_manifest["output_sha256"][name]:
            raise ValueError("Source checksum mismatch: " + name)
    original = pd.read_parquet(source / "asset_daily_returns.parquet")
    baseline = pd.read_csv(source / "summary_metrics.csv")
    opposite = reverse_paths(original)
    np.testing.assert_allclose(opposite.net_return + original.net_return,
                               -2 * original.cost_return, equal_nan=True, atol=1e-14)
    pd.testing.assert_frame_equal(original[["mode", "session", "root", "horizon", "ready", "active"]],
                                  opposite[["mode", "session", "root", "horizon", "ready", "active"]])
    policy = "all-horizons" if baseline["sample"].eq("all-horizons matched").all() else "per-horizon"
    metrics, curves = summarize(opposite, sample=policy)
    keys = ["mode", "book", "horizon"]
    compared = metrics.merge(baseline, on=keys, suffixes=("_opposite", "_original"), validate="one_to_one")
    assert compared.days_opposite.eq(compared.days_original).all()
    np.testing.assert_allclose(compared.gross_sharpe_opposite, -compared.gross_sharpe_original,
                               equal_nan=True, atol=1e-12)
    source_curves = pd.read_parquet(source / "equity_curves.parquet")
    date_keys = ["mode", "book", "horizon", "session"]
    pd.testing.assert_frame_equal(curves[date_keys].reset_index(drop=True),
                                  source_curves[date_keys].reset_index(drop=True))
    out = args.output_dir or Path("five_currents_intraday_opposite_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(exist_ok=False)
    opposite.to_parquet(out / "asset_daily_returns.parquet", index=False)
    metrics.to_csv(out / "summary_metrics.csv", index=False)
    compared.to_csv(out / "original_vs_opposite.csv", index=False)
    curves.to_parquet(out / "equity_curves.parquet", index=False)
    diagnostic = compared.loc[compared.book.eq("portfolio_ex_silver") & compared["mode"].eq("target")]
    columns = ["horizon", "days_opposite", "net_sharpe_original", "gross_sharpe_opposite",
               "net_sharpe_opposite", "stressed_sharpe_opposite", "total_return_opposite", "max_drawdown_opposite"]
    print("OPPOSITE NET POSITIONS / FOUR-ASSET MATCHED SAMPLE", flush=True)
    print(diagnostic[columns].to_string(index=False), flush=True)
    asset_metrics = metrics.loc[metrics["mode"].eq("target") & metrics.book.isin(ROOTS)]
    asset_table = asset_metrics.pivot(index="book", columns="horizon", values="net_sharpe").reindex(ROOTS)
    asset_table.loc["SI"] = np.nan
    print("OPPOSITE NET POSITION / ASSET NET SHARPE", flush=True)
    print(asset_table.to_string(), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for color, horizon in zip(plt.cm.viridis(np.linspace(.1, .9, len(HORIZONS))), HORIZONS):
        one = curves.loc[curves["mode"].eq("target") & curves.book.eq("portfolio_ex_silver") & curves.horizon.eq(horizon)]
        axes[0].plot(one.session, one.equity, label=f"{horizon} min", color=color)
        axes[1].plot(one.session, one.drawdown * 100., color=color)
    axes[0].set_title("Opposite net positions / CL-NG-GC-HG / after costs", loc="left")
    axes[0].set_ylabel("Growth of 100")
    axes[1].set_ylabel("Drawdown %")
    axes[0].legend(ncol=7, frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=150, bbox_inches="tight", facecolor="#faf9f4")
    plt.close(fig)
    embedded = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    display = diagnostic[columns].copy()
    for name in ("total_return_opposite", "max_drawdown_opposite"):
        display[name] = display[name].map(lambda value: f"{100 * value:.3f}%")
    asset_display = asset_metrics.copy()
    for name in ("gross_sharpe", "net_sharpe", "stressed_sharpe"):
        asset_display.loc[asset_display.days.lt(20), name] = np.nan
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Five Currents / Opposite Intraday Positions</title>
<style>body{{font:15px/1.6 Georgia,serif;color:#193c36;background:#faf9f4;margin:0}}main{{max-width:1120px;margin:auto;padding:40px}}h1{{font-size:36px}}img{{width:100%}}table{{width:100%;border-collapse:collapse;font:11px/1.5 Consolas,monospace}}th,td{{padding:8px;text-align:right;border-bottom:1px solid #d9ded1}}th{{background:#e7ede3}}.scroll{{overflow:auto}}</style></head>
<body><main><h1>Opposite Net Positions, Intraday Exit</h1>
<p>Reverse each commodity's full combined daily target, retaining exactly the original dates, size, contract, entry, exit and cost assumptions. Entry is 06:00 UTC in US summer and 07:00 UTC in winter. Gross P&amp;L changes sign; costs do not. No refitting, API calls, database reads or production changes.</p>
<h2>Four-Asset Matched Comparison</h2><div class="scroll">{display.to_html(index=False, border=0, float_format=lambda value: f'{value:.2f}')}</div><img src="{embedded}" alt="Opposite net position equity and drawdown">
<p>CL, NG, GC and HG keep their original economic weights; silver's weight is not redistributed. All seven horizons use the same 125 complete dates, from 27 September 2024 to 10 August 2026. Equity compounds matched observations only, not a continuous deployable calendar record. Missing outcomes are not treated as zero. Sharpe is annualized with 252 observations.</p>
<h2>Individual Assets / Opposite Net Positions</h2><div class="scroll">{asset_display[['book','horizon','days','start','end','gross_sharpe','net_sharpe','stressed_sharpe','total_return','max_drawdown','win_rate']].to_html(index=False, border=0, na_rep='n/a', float_format=lambda value: f'{value:.4f}')}</div>
<p>Each asset has its own all-horizon matched sample; asset rows are not the identical history used for the portfolio. Total return, drawdown and win rate in the asset table are decimal fractions. Silver has only five matched asset days and the five-asset book only four: neither supports meaningful inference.</p>
<h2>Interpretation</h2><p>This sign reversal was proposed after seeing the original outcome and is therefore an exploratory, in-sample sensitivity, not independent validation. Historical daily publication vintages remain incomplete, so signal availability by the entry time is conditional. Standard/micro fees, adverse slippage, doubled-cost stress and minute-open execution proxies are unchanged from the original study.</p>
<p>The same reversal of the target-change experiment is retained separately in the CSV/parquet audit files; headline results here refer to full net positions only.</p></main></body></html>'''
    report = out / "Five_Currents_Opposite_Intraday_Report.html"
    report.write_text(html, encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_study": str(source),
                "source_hashes": hashes, "operation": "reverse gross P&L; retain positive execution costs",
                "signal_refits": False, "new_api_calls": False, "database_access": False,
                "same_dates_and_exposures": True, "independent_validation": False,
                "output_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                  for file in out.iterdir() if file.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[report] " + str(report.resolve()), flush=True)


if __name__ == "__main__":
    main()
