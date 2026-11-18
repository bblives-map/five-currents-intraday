"""Uniform nearby-contract selection and frozen-signal intraday replay, cached data only."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_five_currents_cached_replay import (
    KEYS, RETURN_FIELDS, VARIANTS, compare_replay, leg_values, totals_for_book,
)
from five_currents_active_contracts import (
    ADV_OBSERVATIONS, EXPIRY_BUFFER_SESSIONS, MAX_NEARBY_RANK, MIN_OBSERVATIONS,
    daily_liquidity, delivery_month, select_active_contracts,
)
from five_currents_front_intraday import evaluate
from five_currents_opening_study import stats
from run_five_currents_cached_window_replay import (
    BOOKS, CLOCKS, SNAPSHOT, load_local_bars, price_moves, score,
)


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
PREVIOUS = ROOT / "five_currents_cached_window_replay_20261013" / "silver_copper"
OUT = ROOT / "five_currents_uniform_active_intraday_20261014"
SPLIT = pd.Timestamp("2025-07-01")


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection(predictors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_parquet(SNAPSHOT / "input_panel.parquet")
    mapping = pd.read_parquet(SNAPSHOT / "input_mapping.parquet")
    candidates = daily_liquidity(wide, mapping)
    sessions = pd.DatetimeIndex(sorted(predictors.session.unique()))
    winners, attempts = select_active_contracts(candidates, sessions,
                                                pd.DatetimeIndex(candidates.date.unique()))
    dates = predictors[["session", "root", "signal_date"]].drop_duplicates()
    targets = winners.merge(dates, on=["session", "root"], validate="one_to_one")
    if targets.volume_asof.dropna().ge(targets.loc[targets.volume_asof.notna(), "liquidity_date"]).any():
        raise AssertionError("Selection volume includes an unlagged source observation")
    return targets, attempts


def price_coverage(moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_rows, book_rows = [], []
    for (clock, hours), group in moves.groupby(["clock", "hours"], sort=False):
        for root, asset in group.groupby("root", sort=False):
            asset_rows.append({"clock": clock, "hours": int(hours), "root": root,
                               "possible_days": len(asset),
                               "matched_days": int(asset.status.eq("matched").sum()),
                               "active_bar_days": int((asset.status.eq("matched")
                                                        & asset.trade_pair_confirmed.eq(True)).sum()),
                               "verified_month_days": int((asset.status.eq("matched")
                                                            & asset.identity_basis.eq("verified_month")).sum())})
        for book, roots in BOOKS.items():
            part = group.loc[group.root.isin(roots)]
            matched = part.pivot(index="session", columns="root", values="status").reindex(columns=roots).eq("matched").all(axis=1)
            active = part.pivot(index="session", columns="root", values="trade_pair_confirmed").reindex(columns=roots).eq(True).all(axis=1)
            exact = part.pivot(index="session", columns="root", values="identity_basis").reindex(columns=roots).eq("verified_month").all(axis=1)
            book_rows.append({"clock": clock, "hours": int(hours), "book": book,
                              "possible_days": len(matched), "matched_days": int(matched.sum()),
                              "active_bar_days": int((matched & active).sum()),
                              "verified_month_days": int((matched & exact).sum())})
    return pd.DataFrame(asset_rows), pd.DataFrame(book_rows)


def asset_grid(predictors: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for clock in CLOCKS:
        chosen_moves = moves.loc[moves.clock.eq(clock)].drop(columns=["clock", "signal_date"])
        for signal, forecast in predictors.groupby("signal", sort=False):
            paths = evaluate(forecast.drop(columns="signal"), chosen_moves)
            for period, mask in (("full", np.ones(len(paths), dtype=bool)),
                                 ("train", paths.session.lt(SPLIT).to_numpy()),
                                 ("test", paths.session.ge(SPLIT).to_numpy())):
                chosen = paths.loc[mask]
                grouped = chosen.groupby(["root", "variant", "hours"], sort=True)
                aggregate = grouped.slippage_net_return.agg(days="count", mean="mean", std="std", net_contribution="sum")
                aggregate["sharpe"] = aggregate["mean"] / aggregate["std"] * np.sqrt(252)
                aggregate.loc[aggregate["std"].le(0), "sharpe"] = np.nan
                aggregate["gross_contribution"] = grouped.gross_return.sum(min_count=1)
                aggregate["commission_drag"] = grouped.commission_return.sum(min_count=1)
                aggregate["slippage_drag"] = grouped.slippage_return.sum(min_count=1)
                aggregate["possible_days"] = grouped.size()
                frames.append(aggregate.drop(columns=["mean", "std"]).reset_index()
                              .assign(clock=clock, signal=signal, period=period))
        print(f"[assets] {clock}: all families, positions and holds", flush=True)
    return pd.concat(frames, ignore_index=True)


def independent_portfolio(predictors: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    forecast = predictors.loc[predictors.signal.eq("portfolio_net")].drop(columns="signal")
    joined = moves.merge(forecast, on=["session", "root"], suffixes=("_move", "_signal"),
                         validate="many_to_one")
    if not joined.signal_date_move.eq(joined.signal_date_signal).all():
        raise AssertionError("Forecast dates differ in independent replay")
    frames = []
    for variant in VARIANTS:
        legs = leg_values(joined, variant)
        for book in BOOKS:
            frames.append(totals_for_book(legs, book).assign(variant=variant, book=book,
                                                           signal="portfolio_net"))
    return pd.concat(frames, ignore_index=True)


def common_date_comparison(daily: pd.DataFrame, predictors: pd.DataFrame,
                           old_moves: pd.DataFrame | None = None) -> pd.DataFrame:
    if old_moves is None:
        old_moves = pd.read_parquet(PREVIOUS / "price_moves.parquet")
    old = independent_portfolio(predictors, old_moves)
    paired = daily.merge(old, on=KEYS, suffixes=("_uniform", "_previous"), validate="one_to_one")
    rows = []
    for (clock, hours, variant, book), group in paired.groupby(["clock", "hours", "variant", "book"]):
        valid = group.slippage_net_return_uniform.notna() & group.slippage_net_return_previous.notna()
        common = group.loc[valid].sort_values("session")
        before = stats(common.slippage_net_return_previous)
        after = stats(common.slippage_net_return_uniform)
        rows.append({"clock": clock, "hours": int(hours), "variant": variant, "book": book,
                     "common_days": len(common), "previous_sharpe": before["sharpe"],
                     "uniform_sharpe": after["sharpe"],
                     "previous_total_return": before["total_return"],
                     "uniform_total_return": after["total_return"]})
    return pd.DataFrame(rows)


def subset_metrics(daily: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book, roots in BOOKS.items():
        for (clock, hours), group in moves.loc[moves.root.isin(roots)].groupby(["clock", "hours"]):
            status = group.pivot(index="session", columns="root", values="status").reindex(columns=roots).eq("matched").all(axis=1)
            active = group.pivot(index="session", columns="root", values="trade_pair_confirmed").reindex(columns=roots).eq(True).all(axis=1)
            exact = group.pivot(index="session", columns="root", values="identity_basis").reindex(columns=roots).eq("verified_month").all(axis=1)
            chosen = daily.loc[daily.book.eq(book) & daily.clock.eq(clock) & daily.hours.eq(hours)]
            for variant, path in chosen.groupby("variant"):
                values = path.set_index("session").slippage_net_return.sort_index()
                for label, mask in (("all matched months", status), ("active bars at both ends", status & active),
                                    ("verified contract months", status & exact), ("inferred delivery months", status & ~exact)):
                    rows.append({"book": book, "clock": clock, "hours": int(hours), "variant": variant,
                                 "subset": label, **stats(values.reindex(mask.index)[mask])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--selection-only", action="store_true")
    modes.add_argument("--prices-only", action="store_true")
    modes.add_argument("--score-only", action="store_true")
    modes.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    predictors = pd.read_parquet(PREVIOUS / "dated_predictors.parquet")
    predictors = predictors.loc[~predictors.signal.str.contains("finalized_reference")].copy()
    if args.comparison_only:
        targets = pd.read_parquet(PREVIOUS / "contract_targets.parquet")
        identities = [delivery_month(row.root, row.contract, row.signal_date)
                      for row in targets.itertuples()]
        targets["month"] = [month for month, _ in identities]
        targets["identity_basis"] = [basis for _, basis in identities]
        bars, _ = load_local_bars(pd.DatetimeIndex(sorted(predictors.session.unique())))
        baseline = price_moves(targets, bars)
        baseline.to_parquet(OUT / "previous_mask_selected_month_prices.parquet", index=False)
        daily = pd.read_parquet(OUT / "portfolio_daily_returns.parquet")
        comparison = common_date_comparison(daily, predictors, baseline)
        comparison.to_csv(OUT / "uniform_mask_comparison.csv", index=False)
        print(comparison.loc[comparison.book.eq("five") & comparison.clock.eq("us_open")
                             & comparison.variant.eq("reverse_change_magnitude")].to_string(index=False), flush=True)
        return
    predictors.to_parquet(OUT / "dated_predictors.parquet", index=False)
    if not args.score_only:
        targets, ledger = selection(predictors)
        targets.to_parquet(OUT / "contract_targets.parquet", index=False)
        ledger.to_parquet(OUT / "liquidity_candidates.parquet", index=False)
        counts = targets.groupby(["root", "rank", "identity_basis"], dropna=False).size().rename("sessions").reset_index()
        counts.to_csv(OUT / "selected_rank_counts.csv", index=False)
        print("[selection]", counts.to_string(index=False), flush=True)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "cached_only": True,
                    "broker_calls": False, "database_calls": False,
                    "source_snapshot": str(SNAPSHOT), "signal_source": str(PREVIOUS / "dated_predictors.parquet"),
                    "signals_retuned": False, "products": list(BOOKS["five"]),
                    "policy": {"nearby_candidates": MAX_NEARBY_RANK, "adv_observations": ADV_OBSERVATIONS,
                               "min_observations": MIN_OBSERVATIONS, "liquidity_observation_lag": 1,
                               "expiry_buffer_sessions": EXPIRY_BUFFER_SESSIONS,
                               "tie_break": "nearer delivery month, prior OI, rank",
                               "selected_contracts_per_asset": 1, "availability_fallback": False,
                               "execution_window_minutes": 10},
                    "mapping": "Actual month when present; existing inferred delivery calendar otherwise",
                    "inputs_sha256": {name: fingerprint(path) for name, path in {
                        "predictors": PREVIOUS / "dated_predictors.parquet", "daily_panel": SNAPSHOT / "input_panel.parquet",
                        "contract_mapping": SNAPSHOT / "input_mapping.parquet"}.items()}}
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.selection_only:
            return
        bars, provenance = load_local_bars(pd.DatetimeIndex(sorted(predictors.session.unique())))
        moves = price_moves(targets, bars)
        extra = targets[["session", "root", "liquidity_date", "volume_asof", "adv", "expiry", "sessions_to_expiry"]]
        moves = moves.merge(extra, on=["session", "root"], validate="many_to_one")
        moves.to_parquet(OUT / "price_moves.parquet", index=False)
        assets, books = price_coverage(moves)
        assets.to_csv(OUT / "asset_price_coverage.csv", index=False)
        books.to_csv(OUT / "portfolio_price_coverage.csv", index=False)
        moves.groupby(["clock", "hours", "root", "status"]).size().rename("days").reset_index().to_csv(
            OUT / "coverage_reasons.csv", index=False)
        manifest["cached_price_sources"] = provenance
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("[coverage]", books.loc[books.clock.eq("us_open")].to_string(index=False), flush=True)
        if args.prices_only:
            return
    else:
        moves = pd.read_parquet(OUT / "price_moves.parquet")
    metrics, daily = score(predictors, moves, include_five_families=True, persist_five_portfolio=True)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    daily.to_parquet(OUT / "portfolio_daily_returns.parquet", index=False)
    independent = independent_portfolio(predictors, moves)
    audit = compare_replay(independent, daily, (*RETURN_FIELDS, "gross_exposure", "ready"),
                           KEYS, "uniform portfolio daily replay")
    (OUT / "portfolio_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    subset_metrics(daily, moves).to_csv(OUT / "portfolio_subsets.csv", index=False)
    common_date_comparison(daily, predictors).to_csv(OUT / "common_date_comparison.csv", index=False)
    asset_grid(predictors, moves).to_csv(OUT / "signal_asset_metrics.csv", index=False)
    chosen = metrics.loc[metrics.book.eq("five") & metrics.signal.eq("portfolio_net")
                         & metrics.basis.eq("slippage_net_return") & metrics.period.eq("full")
                         & metrics.variant.eq("reverse_change_magnitude")]
    print("[five commodities]", chosen[["clock", "hours", "days", "sharpe", "total_return"]].to_string(index=False), flush=True)
    print("[complete]", OUT, flush=True)


if __name__ == "__main__":
    main()
