"""Produce an auditable, offline summary of the corrected intraday replay."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRONT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_frontmonth_20261025"
REPORT = FRONT / "frontmonth_replay_report.html"


def table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data", escape=True,
                         float_format=lambda value: f"{value:,.2f}")


def growth_axis(ax, dates, values, label, color):
    observed = np.isfinite(values)
    line = np.cumprod(1 + np.where(observed, values, 0.0))
    line[~observed] = np.nan
    ax.plot(dates, line, lw=1.6, label=label, color=color)


def write_curve(daily, columns, destination, title):
    fig, ax = plt.subplots(figsize=(12, 4.3))
    colors = ("#145d58", "#b47d35", "#496e94", "#974a45")
    for column, color in zip(columns, colors):
        growth_axis(ax, daily.session, daily[column].to_numpy(float),
                    column.replace("_", " "), color)
    ax.axhline(1, color="#a4aca6", lw=.7)
    ax.grid(alpha=.17)
    ax.set_title(title, loc="left", fontsize=13, weight="bold")
    ax.set_ylabel("Observed-trade growth of 1")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    strict = pd.read_parquet(FRONT / "clock_coverage.parquet")
    rolled = pd.read_parquet(FRONT / "clock_coverage_liquidity.parquet")
    joined = strict.merge(rolled, on=["root", "clock", "hold_minutes", "sessions"],
                          suffixes=("_strict", "_rolled"))
    coverage = joined.loc[joined.hold_minutes.eq(60)
                          & joined.clock.isin(("asia_open", "london_open", "us_open")),
                          ["root", "clock", "matched_sessions_strict",
                           "matched_sessions_rolled", "coverage_strict", "coverage_rolled"]].copy()
    coverage["coverage_strict"] *= 100
    coverage["coverage_rolled"] *= 100
    coverage.columns = ("Asset", "Clock", "Strict priced", "Prior-volume priced",
                        "Strict coverage %", "Prior-volume coverage %")
    coverage.to_parquet(FRONT / "coverage_comparison.parquet", index=False)

    audit = pd.read_parquet(FRONT / "alpha_audit" / "individual_hindsight_winners.parquet")
    books = pd.read_parquet(FRONT / "two_books" / "book_comparison.parquet")
    legs = pd.read_parquet(FRONT / "two_books" / "leg_metrics.parquet")
    rolling = pd.read_parquet(FRONT / "rolling" / "grid_metrics.parquet")
    book_daily = pd.read_parquet(FRONT / "two_books" / "book_daily.parquet")
    roll_daily = pd.read_parquet(FRONT / "rolling" / "daily_replay.parquet")
    if not book_daily.session.equals(roll_daily.session):
        raise AssertionError("Book and rolling session calendars differ")
    selection = pd.read_parquet(FRONT / "rolling" / "family_selection_counts.parquet")
    routes = pd.read_parquet(FRONT / "route_selection_liquidity.parquet")
    for root in ("CL", "NG", "GC", "HG"):
        if not (routes.root.eq(root).sum() == len(book_daily)):
            raise AssertionError(f"Missing route decision for {root}")
    favorite_books = ("two_books_carry_oi", "two_books_equal", "four_curve")
    write_curve(book_daily, favorite_books, FRONT / "front_books_equity.png",
                "Fixed intraday books on corrected front/second-month fills")
    favorite_rolling = ("fixed_carry_oi", "family_L252_C21_K2_equal",
                        "family_L63_C5_K2_equal")
    write_curve(roll_daily, favorite_rolling, FRONT / "front_rolling_equity.png",
                "Fixed candidate set versus past-only intraday allocation")

    summary = books.loc[books.book.isin(favorite_books),
                        ["book", "net_sr", "stress_sr", "after_top5_sr",
                         "priced_trade_coverage", "planned_days", "matched_days",
                         "sr_2025", "sr_2026"]].copy()
    summary.priced_trade_coverage *= 100
    summary.columns = ("Book", "Net SR", "2x cost SR", "SR less top five wins",
                       "Priced trade %", "Planned days", "Priced days",
                       "2025 SR", "2026 SR")
    rolling_view = rolling.loc[rolling.strategy.isin(favorite_rolling),
                               ["strategy", "net_sr", "stress_sr", "coverage",
                                "planned_days", "priced_days", "sr_2025", "sr_2026"]].copy()
    rolling_view.coverage *= 100
    rolling_view.columns = ("Allocation", "Net SR", "2x cost SR", "Priced trade %",
                            "Planned days", "Priced days", "2025 SR", "2026 SR")
    asset_view = legs.loc[legs.book.str.startswith(("CL:", "NG:", "GC:", "HG:")),
                          ["root", "family", "signal_class", "clock", "hold_minutes",
                           "net_sr", "priced_trade_coverage", "sr_2026"]].copy()
    asset_view.priced_trade_coverage *= 100
    asset_view.columns = ("Asset", "Family", "Vote", "Clock", "Hold min",
                          "Net SR", "Priced trade %", "2026 SR")
    selected = audit[["root", "family", "signal_class", "clock", "hold_minutes",
                      "net_sr", "priced_trade_coverage"]].sort_values(["root", "family"])
    selected.to_parquet(FRONT / "selected_rules.parquet", index=False)
    selection_view = selection.loc[selection.strategy.eq("family_L252_C21_K2_equal"),
                                   ["root", "family", "selected_2025", "selected_2026"]]
    route_view = (routes.groupby(["root", "preferred_table"]).size()
                  .rename("sessions").reset_index())
    fixed_row = books.set_index("book").loc["two_books_carry_oi"]
    causal_row = rolling.set_index("strategy").loc["family_L252_C21_K2_equal"]
    gc_early = coverage.loc[coverage.Asset.eq("GC") & coverage.Clock.eq("asia_open")].iloc[0]
    hg_early = coverage.loc[coverage.Asset.eq("HG") & coverage.Clock.eq("asia_open")].iloc[0]
    finding = (
        f"At the 60-minute Asia-open test, gold coverage rose from "
        f"{gc_early['Strict coverage %']:.0f}% to {gc_early['Prior-volume coverage %']:.0f}%; "
        f"copper rose from {hg_early['Strict coverage %']:.0f}% to "
        f"{hg_early['Prior-volume coverage %']:.0f}%. The fixed four-asset book has "
        f"{fixed_row.net_sr:.2f} net Sharpe on only {fixed_row.priced_trade_coverage:.0%} "
        f"of intended sessions. The past-only 252-session/monthly allocation has "
        f"{causal_row.net_sr:.2f} net Sharpe and {causal_row.coverage:.0%} coverage; "
        f"its doubled-cost Sharpe is {causal_row.stress_sr:.2f}. Neither is a "
        "complete investable history. Thin copper nearby months are still unresolved."
    )

    caveat = ("Intraday prices are read-only cached 1-minute IBKR bars. The daily curves "
              "and votes are unchanged; the intraday execution contract is the nearest "
              "tradable rank one or two, chosen from the prior five completed sessions' "
              "volume, with a pre-expiry roll. Entries and exits use positive-volume "
              "minute-close price proxies within ten minutes and the same listed contract. All "
              "figures are after the existing micro-commission and slippage model. "
              "Missing active fills remain missing, not zero returns. Vote treatment, "
              "clock and hold are selected using the full sample and therefore carry "
              "hindsight bias; only the subsequent allocation refits are past-only.")
    superseded = ("Earlier intraday exhibits used a highest-ADV contract from ranks 1-6 "
                  "and a five-minute sampled price cache. Those studies should not be "
                  "used as evidence for front-month execution. The corrected raw grid, "
                  "fill ledgers and selection paths are retained alongside this report.")
    residual = ("Copper remains the binding data limitation. For example, in September "
                "2025 the cached rank-one route was mostly October copper and rank two "
                "mostly November copper; both were thin. December copper was available "
                "in rank two only late in the month. Missing active trades are left "
                "unpriced. Daily open interest is present in the archived curve panel, "
                "but most historic contract-month identities there are inferred, so "
                "this exact-micro execution roll uses observed prior-session minute "
                "volume rather than joining uncertain open-interest identities.")
    sections = [
        ("Coverage repair", table(coverage)),
        ("Contract route choices", table(route_view)),
        ("Corrected fixed books", '<img src="front_books_equity.png" alt="Fixed book equity">'
         + table(summary)),
        ("Asset-level rules", '<img src="two_books/asset_legs.png" alt="Asset-level equity">'
         + table(asset_view)),
        ("Carry, curve and momentum families",
         '<img src="two_books/carry_momentum_group_heatmap.png" alt="Family comparison">'),
        ("Past-only allocation of hindsight-selected recipes",
         '<img src="front_rolling_equity.png" alt="Rolling equity">' + table(rolling_view)),
        ("Intraday recipe selection frequency", table(selection_view)),
    ]
    body = "\n".join(f"<section><h2>{html.escape(title)}</h2>{content}</section>"
                     for title, content in sections)
    generated = pd.Timestamp.now(tz="UTC").isoformat()
    REPORT.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Five Currents | Front-Month Intraday Replay</title><style>
body{{font-family:Georgia,serif;background:#f4f1e9;color:#17332f;margin:0;line-height:1.5}}
main{{max-width:1080px;margin:auto;padding:48px 30px 80px}}
h1{{font-size:42px;margin:0 0 10px}}h2{{font-size:23px;border-top:1px solid #b9c4bb;padding-top:26px}}
.lead{{font-size:17px;max-width:930px}}.meta{{color:#59716d;font-size:13px}}
section{{margin:40px 0}}img{{max-width:100%;display:block;margin:20px 0}}
table{{border-collapse:collapse;font:12px/1.4 Arial,sans-serif;width:100%;margin:20px 0}}
th,td{{padding:8px 10px;border-bottom:1px solid #d9dfd9;text-align:left}}
th{{background:#e2ebe4;position:sticky;top:0}}tr:nth-child(even){{background:#fafbf8}}
.note{{border-left:4px solid #b47d35;padding:14px 20px;background:#fff9eb}}
</style></head><body><main><p class="meta">Research audit | {html.escape(generated)}</p>
<h1>Five Currents: Front-Month Repair</h1>
<p class="lead">Daily signals kept intact. Intraday execution rebuilt from the nearest tradable micro contract, using only the cached one-minute bars.</p>
<p class="note"><strong>Main finding.</strong> {html.escape(finding)}</p>
<p>{html.escape(caveat)}</p><p>{html.escape(residual)}</p>
<p>{html.escape(superseded)}</p>{body}
<p class="meta">Signal dates: {book_daily.session.min().date()} to {book_daily.session.max().date()}.
Silver excluded from this four-asset replay because its intraday minute coverage was not validated.</p>
</main></body></html>""", encoding="utf-8")
    (FRONT / "frontmonth_report_manifest.json").write_text(json.dumps({
        "report": str(REPORT), "created_utc": generated,
        "sources": ["manifest_liquidity.json", "alpha_audit/manifest.json",
                    "rolling/manifest.json"],
        "caveat": caveat, "finding": finding, "residual": residual,
        "superseded": superseded,
    }, indent=2), encoding="utf-8")
    print(f"[report] {REPORT}")
    print("[coverage]\n" + coverage.to_string(index=False))
    print("[books]\n" + summary.to_string(index=False))
    print("[rolling]\n" + rolling_view.to_string(index=False))


if __name__ == "__main__":
    main()
