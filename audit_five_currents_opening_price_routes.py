"""Separate price-route effects from sample-date effects in the opening experiment."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from compare_five_currents_opening_policies import PRIMARY_POLICIES
from five_currents_opening_study import stats


def paired_route_metrics(daily):
    rows = []
    selected = daily.loc[daily.signal.eq("portfolio_net") & daily.policy.isin(PRIMARY_POLICIES)]
    for key, group in selected.groupby(["policy", "book", "variant", "hours"]):
        pivot = group.pivot(index="session", columns="price_route", values="net_return").dropna()
        if set(pivot.columns) != {"front", "second_series_proxy"}:
            raise AssertionError("Missing a price route")
        metadata = dict(zip(("policy", "book", "variant", "hours"), key))
        correlation = pivot.front.corr(pivot.second_series_proxy) if len(pivot) > 1 else np.nan
        for route, subset in group.groupby("price_route"):
            subset = subset.set_index("session").loc[pivot.index]
            for basis in ("gross_return", "net_return", "slippage_net_return"):
                rows.append({**metadata, "price_route": route, "cost_basis": basis,
                             "paired_route_net_return_correlation": correlation,
                             "start": pivot.index.min(), "end": pivot.index.max(), **stats(subset[basis])})
    return pd.DataFrame(rows)


def run(source):
    daily = pd.read_parquet(source / "portfolio_daily_returns.parquet")
    paired = paired_route_metrics(daily)
    paired.to_csv(source / "same_date_price_route_metrics.csv", index=False)
    counts = (daily.loc[daily.signal.eq("portfolio_net") & daily.policy.isin(PRIMARY_POLICIES)]
              .assign(year=lambda frame: frame.session.dt.year)
              .groupby(["price_route", "policy", "book", "variant", "hours", "year"])
              .net_return.count().rename("matched_days").reset_index())
    counts.to_csv(source / "sample_dates_by_year.csv", index=False)
    view = paired.loc[paired.book.eq("four") & paired.cost_basis.eq("net_return")
                      & paired.variant.isin(("net_magnitude", "change_magnitude"))]
    table = view.pivot(index=["policy", "variant", "price_route"], columns="hours", values="sharpe")
    samples = view.loc[view.variant.eq("net_magnitude")].pivot(
        index=["policy", "price_route"], columns="hours", values="days")
    control = view.loc[view.policy.eq("lagged_volume_oi") & view.variant.eq("net_magnitude") & view.hours.eq(6)].set_index("price_route")
    broad = pd.read_csv(source / "common_date_policy_metrics.csv")
    broad = broad.loc[broad.policy.eq("lagged_volume_oi") & broad.variant.eq("net_magnitude")
                      & broad.hours.eq(6) & broad.book.eq("four") & broad.cost_basis.eq("net_return")
                      & broad.price_route.eq("second_series_proxy")].iloc[0]
    finding = (f'<h2>Interpretation</h2><p>Lagged flow inputs retain positive net-position performance '
               f'in the limited front-series cohort, but that does not establish a full-history opening edge. '
               f'On {int(control.loc["front", "days"])} identical four-market six-hour dates, commission-net Sharpe '
               f'is {control.loc["front", "sharpe"]:.2f} for the front series versus '
               f'{control.loc["second_series_proxy", "sharpe"]:.2f} for the nearby archive. '
               f'Across the nearby archive\'s larger {int(broad.days)}-date sample it is {broad.sharpe:.2f}. '
               f'The difference in available dates therefore materially changes the conclusion; '
               f'it cannot be attributed solely to volume/OI availability or contract choice.</p>')
    section = ('<h2>Same-Date Price-Route Control</h2><p>Each pair below uses the identical availability policy, '
               'signal weights, opening time, holding period and dates in both stored price archives. '
               'These conditional results separate the effect of different contract/price paths from the '
               'effect of the larger second-series archive. They do not extend the front-contract track record.</p>'
               '<div class="scroll">'+table.to_html(border=0,float_format=lambda x:f"{x:.2f}",na_rep="n/a")+
               '<p>Paired observations:</p>'+samples.to_html(border=0)+'</div>')
    report = source / "Five_Currents_Opening_Availability_Study.html"
    text = report.read_text(encoding="utf-8")
    assert "<h2>Same-Date Price-Route Control</h2>" not in text
    text = text.replace("<h2>Individual Alphas / Commission-Net Sharpe</h2>",
                        section+"<h2>Individual Alphas / Commission-Net Sharpe</h2>")
    text = text.replace("<h2>What Changes</h2>", finding+"<h2>What Changes</h2>")
    report.write_text(text, encoding="utf-8")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["same_date_price_route_control"] = True
    manifest["price_route_control_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_code_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["source_code_sha256"]["five_currents_opening_metrics.py"] = hashlib.sha256(
        Path("five_currents_opening_metrics.py").read_bytes()).hexdigest()
    for name in manifest["source_code_sha256"]:
        manifest["source_code_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    original_manifest = Path(manifest["source_study"]) / "manifest.json"
    manifest["source_study_manifest_sha256"] = hashlib.sha256(original_manifest.read_bytes()).hexdigest()
    manifest["outputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source.iterdir() if p.is_file() and p.name != "manifest.json"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[paired commission-net Sharpe / four markets]")
    print(table.round(3).to_string())
    print("[paired observations]")
    print(samples.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    run(parser.parse_args().source)
