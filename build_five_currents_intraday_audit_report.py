"""Build an embedded, reproducible summary of the cached intraday reconciliation."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_currents_front_intraday import stats


def run(audit, rebuild, output):
    output.mkdir(exist_ok=False)
    verification = json.loads((rebuild / "full_daily_rebuild_verification.json").read_text())
    assert verification["operational_daily_pnl_matches_snapshot"]
    checks = json.loads((audit / "daily_reproduction_verification.json").read_text())
    baseline = pd.read_csv(audit / "dashboard_daily_benchmarks.csv")
    common = pd.read_csv(audit / "common_date_12h_route_and_timing_metrics.csv")
    timing = pd.read_csv(audit / "intraday_timing_metrics.csv")
    full = pd.read_csv(audit / "standalone_12h_metrics.csv")
    decomp = pd.read_parquet(audit / "per_asset_price_path_decomposition.parquet")
    records = []
    for root, group in decomp.loc[decomp.route.eq("stored_front")].groupby("root"):
        available = group.loc[group.entry_price.notna()]
        records.append({"Market": root, "Entry prices": len(available),
                        "Same daily delivery month (%)": available.same_daily_contract.astype(bool).mean()*100.})
    alignment = pd.DataFrame(records)
    components = decomp.loc[decomp.route.eq("stored_front")
                            & decomp.status.eq("complete session decomposition")
                            & decomp.daily_price_return.notna()].copy()
    valid = components.groupby("session").root.nunique().eq(4)
    components = components.loc[components.session.isin(valid.index[valid])]
    labels = {"overnight_return": "Prior close to morning entry",
              "intraday_on_close_base": "Morning entry to 12-hour exit",
              "after_exit_on_close_base": "12-hour exit to session close"}
    means = {label: (components.held_weight*components[column]).groupby(components.session).sum().mean()*10000.
             for column, label in labels.items()}
    outside = (list(means.values())[0]+list(means.values())[2])/sum(means.values())
    paths = pd.read_parquet(audit / "standalone_12h_daily_returns.parquet")
    wide = paths.pivot(index="session", columns="case", values="return")
    dates = wide.dropna().index
    plot_cases = {"stored_front/fresh/net": "Fresh forecast / stored front",
                  "stored_front/held/net": "Daily-held position / stored front",
                  "daily_contract_control/held/net": "Daily-held position / daily contract"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#aa773b", "#647f8f", "#0c756b"]
    for (case, label), color in zip(plot_cases.items(), colors):
        values = wide.loc[dates, case]
        equity = ((1.+values).cumprod()*100.).reindex(wide.index)
        axes[0].plot(equity.index, equity, label=label, color=color, lw=1.5)
    axes[0].set_title(f"Identical {len(dates)} dates / 12-hour returns after micro commissions", loc="left")
    axes[0].set_ylabel("Growth of 100")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].barh(list(means), list(means.values()), color=["#647f8f", "#0c756b", "#aa773b"])
    axes[1].set_xlabel("Average daily gross P&L contribution, basis points")
    axes[1].set_title(f"Where session P&L accrued / identical {len(valid.index[valid])} complete dates", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    (output / "audit_comparison.png").write_bytes(buffer.getvalue())
    chart = base64.b64encode(buffer.getvalue()).decode()
    common_display = common.loc[common.case.isin(plot_cases)].copy()
    common_display["Test"] = common_display.case.map(plot_cases)
    common_display = common_display[["Test", "days", "sharpe", "total_return", "max_drawdown"]]
    common_display.columns = ["12-hour test", "Days", "Net Sharpe", "Total return", "Max drawdown"]
    for column in ("Total return", "Max drawdown"):
        common_display[column] = common_display[column].map(lambda x: f"{x:.2%}")
    timing_display = timing.loc[timing.book.eq("four") & timing.variant.eq("net_magnitude") & timing.hours.eq(12)]
    evidence = {"complete_front_session_decomposition_days": int(valid.sum()),
                "average_component_pnl_bps": means, "gross_pnl_share_outside_12h_window": outside,
                "common_route_timing_days": len(dates), "source_daily_checks": checks,
                "full_rebuild": verification}
    (output / "audit_findings.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Five Currents / Intraday Reconciliation Audit</title>
<style>body{{background:#f7f5ee;color:#173e38;font:16px/1.6 Georgia,serif;margin:0}}main{{max-width:1050px;margin:auto;padding:42px 30px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:32px}}table{{width:100%;border-collapse:collapse;font:13px/1.5 Consolas,monospace}}td,th{{text-align:right;padding:9px;border-bottom:1px solid #d8dfd4}}td:first-child,th:first-child{{text-align:left}}th{{background:#e7ece1}}img{{width:100%}}.box{{background:#e9eee2;padding:18px 24px;border-left:4px solid #0c756b}}.small{{font-size:13px;color:#52665e}}.scroll{{overflow:auto}}@media print{{body{{background:white}}main{{padding:0}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}</style></head><body><main>
<p>FIVE CURRENTS / ACCOUNTING AND TIMING AUDIT</p><h1>The daily edge and the morning trade were not the same experiment.</h1>
<div class="box">All ten daily concepts were rebuilt from cached daily price, volume and open-interest inputs. Original causal refits, risk scaling, operational weights and the dashboard's daily P&L reproduce to numerical precision. No live collection, database writes, dashboard changes or new research parameter search.</div>
<h2>What Was Different</h2>
<p><b>Timing.</b> The daily convention is forecast at close t, execution at close t+1, then marking at close t+2. The previous morning test used the fresh forecast from the immediately preceding close, not the position already held under that daily convention. For example, Wednesday morning's held position comes from Monday's forecast; Tuesday's fresh forecast is scheduled for Wednesday's close. Both are legitimate experiments, but they answer different questions.</p>
<p><b>Contract series.</b> The first stored calendar-month series is not always the daily strategy's delivery month. The control below chooses the known daily contract at entry, never a contract selected from exit availability or its eventual P&L. Each trade stays in one contract.</p>
{alignment.to_html(index=False,border=0,float_format=lambda x:f"{x:.1f}")}
<p><b>Changing coverage.</b> Silver recovery, silver exclusion and exact-contract matching changed the observation dates. Sharpe differences across those samples cannot be attributed to strategy changes alone. Below, all three curves and headline comparisons use the same {len(dates)} dates.</p>
<p><b>Different costs and price windows.</b> The daily strategy pays for changes and rolls; an intraday strategy pays a round trip every active day. Micro fees remain $2.16 per round trip, with no extra conversion factor. The 12-hour window is 06:00-18:00 UTC in US summer and 07:00-19:00 UTC in winter. It omits overnight and remaining-session returns. Daily settlement marks and last-minute session-close marks are different anchors.</p>
<h2>Daily Replay / Verified Benchmarks</h2>
<p class="small">Full history: 2020-07-08 to 2026-10-09. Intraday calendar: 2024-06-20 to 2026-10-09. Four markets are CL/NG/GC/HG; five markets also include SI. Metrics below include the original daily strategy's modeled costs.</p>
<div class="scroll">{baseline.to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}")}</div>
<h2>Same Dates, Same Weights / Controlled 12-Hour Comparison</h2>
{common_display.to_html(index=False,border=0,float_format=lambda x:f"{x:.3f}")}
<p class="small">CL, NG, GC and HG only. Original root weights are retained without reallocation; exposure sign is not substituted for magnitude. Commission-only headline results. Price marks are not proof of obtainable fills; separate modeled-slippage results remain in the linked data.</p>
<img src="data:image/png;base64,{chart}" alt="Common-date intraday equity and session P&L decomposition">
<h2>Do Not Replace the Longer Sample With the Better Subset</h2>
<div class="scroll">{timing_display.to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}")}</div>
<p>The corrected daily-held timing improves the original front-series test, but does not turn its longer-sample 12-hour result into a 1.5-Sharpe strategy. The better same-contract control has narrower coverage and must remain conditional evidence, not a replacement full-history claim.</p>
<h2>Where the Difference Went</h2>
<p>On the {len(valid.index[valid])}-day complete front-price decomposition cohort, {outside:.1%} of gross session P&L accrued outside the morning-to-12-hour window. The component P&Ls use one common previous-close denominator and add exactly to the full-session return. This is a gross attribution, not three independent tradable strategies. Its full-session anchor is 16:00 Chicago, not an assertion that the exchange settlement price occurred then.</p>
<h2>Verification</h2>
<p>Independent scalar accounting matches the vectorized daily ledger. All ten standalone concept return histories match the dashboard. An independent same-contract source-price lookup checked {checks['frozen_input_price_pair_count']:,} frozen forward-return pairs; maximum error {checks['frozen_input_price_pair_maximum_error']:.3g}. Historical final/revised OI availability at the morning entry is still an assumption, not something this price-path audit can prove.</p>
<p class="small">Inputs: publication 32, signal date 2026-10-09; stored minute history begins in June 2024. Missing active prices remain missing. No interpolation, missing-return cash substitution, revised commission scaling or hindsight contract switching. The original trials and live publications were preserved.</p>
</main></body></html>'''
    path = output / "Five_Currents_Intraday_Reconciliation_Audit.html"
    path.write_text(html, encoding="utf-8")
    inputs = [audit / "manifest.json", rebuild / "full_daily_rebuild_verification.json", Path(__file__)]
    manifest = {"input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("five_currents_intraday_audit_final_20261011"))
    parser.add_argument("--rebuild", type=Path, default=Path("five_currents_daily_full_rebuild_20261011"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.audit, args.rebuild, args.output_dir)
