"""Independent, offline arithmetic and source audit for the cached intraday replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_five_currents_cached_window_replay import load_local_bars


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
SOURCE = ROOT / "five_currents_cached_window_replay_20261013" / "silver_copper"
OUT = SOURCE / "independent_audit"
SPLIT = pd.Timestamp("2025-07-01")
IDENTITY_BOUNDARY = pd.Timestamp("2026-07-21")
ROOTS = ("CL", "NG", "GC", "SI", "HG")
BOOKS = {"four": ("CL", "NG", "GC", "HG"), "five": ROOTS}
MULTIPLIER = {"CL": 100., "NG": 1000., "GC": 10., "SI": 1000., "HG": 2500.}
TICK_COST = {"CL": .01, "NG": .001, "GC": .1, "SI": .005, "HG": .0005}
ROUND_TRIP_COMMISSION = 2.16
BASES = ("net_sign", "net_magnitude", "change_sign", "change_magnitude")
VARIANTS = (*BASES, *("reverse_" + name for name in BASES))
KEYS = ["session", "clock", "hours", "variant", "book", "signal"]
METRIC_KEYS = ["clock", "signal", "variant", "hours", "book", "basis", "period"]
METRIC_FIELDS = ("possible_days", "days", "sharpe", "total_return", "max_drawdown",
                 "annual_vol", "win_rate", "median_gain", "median_loss")
RETURN_FIELDS = ("gross_return", "commission_return", "slippage_return", "net_return",
                 "slippage_net_return")


def independent_stats(series: pd.Series) -> dict:
    values = series.dropna().to_numpy(dtype=float)
    empty = dict(days=0, sharpe=np.nan, total_return=np.nan, max_drawdown=np.nan,
                 annual_vol=np.nan, win_rate=np.nan, median_gain=np.nan, median_loss=np.nan)
    if not len(values):
        return empty
    growth = np.cumprod(1 + values)
    volatility = np.std(values, ddof=1) if len(values) > 1 else np.nan
    active = values[values != 0]
    return dict(days=len(values),
                sharpe=np.mean(values) / volatility * np.sqrt(252) if volatility > 0 else np.nan,
                total_return=growth[-1] - 1,
                max_drawdown=np.min(growth / np.maximum(np.maximum.accumulate(growth), 1) - 1),
                annual_vol=volatility * np.sqrt(252),
                win_rate=np.mean(active > 0) if len(active) else np.nan,
                median_gain=np.median(values[values > 0]) if np.any(values > 0) else np.nan,
                median_loss=np.median(values[values < 0]) if np.any(values < 0) else np.nan)


def assert_close_columns(left: pd.DataFrame, right: pd.DataFrame, fields: tuple[str, ...],
                         label: str, tolerance: float = 1e-11) -> dict:
    result = {}
    for field in fields:
        a = left[field].to_numpy(dtype=float)
        b = right[field].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(f"{label}: {field} has different missing-value mask")
        error = np.abs(a - b)
        maximum = float(np.nanmax(error)) if np.isfinite(error).any() else 0.
        if maximum > tolerance:
            raise AssertionError(f"{label}: {field} max error {maximum}")
        result[field] = maximum
    return result


def audit_signal_lineage(predictors: pd.DataFrame, targets: pd.DataFrame) -> dict:
    if predictors.duplicated(["session", "root", "signal"]).any():
        raise AssertionError("Duplicate dated predictor")
    if targets.duplicated(["session", "root"]).any():
        raise AssertionError("Duplicate contract target")
    if not predictors.signal_date.lt(predictors.session).all():
        raise AssertionError("Non-causal signal date")
    dates = predictors.merge(targets[["session", "root", "signal_date"]],
                             on=["session", "root"], suffixes=("_signal", "_target"),
                             validate="many_to_one")
    if not dates.signal_date_signal.eq(dates.signal_date_target).all():
        raise AssertionError("Signal and contract selector use different dates")
    main = predictors.loc[predictors.signal.eq("portfolio_net"),
                          ["session", "signal_date", "root", "net_magnitude", "change_magnitude"]]
    archive = pd.read_parquet(ROOT / "five_currents_opening_complete_20261011"
                             / "lagged_volume_oi_targets.parquet")
    if archive.index.has_duplicates or archive.isna().any().any():
        raise AssertionError("Archived lagged daily targets are not unique and complete")
    positions = archive.index.get_indexer(main.signal_date)
    if (positions < 1).any():
        raise AssertionError("Intraday signal date is absent from archived targets")
    latest = archive.index[archive.index.searchsorted(main.session, side="left") - 1]
    if not pd.Index(latest).equals(pd.Index(main.signal_date)):
        raise AssertionError("Intraday replay did not use the latest completed daily target")
    archive_errors = {"net_magnitude": 0., "change_magnitude": 0.}
    for root in ROOTS:
        chosen = main.loc[main.root.eq(root)]
        offsets = archive.index.get_indexer(chosen.signal_date)
        level = archive[root].to_numpy(float)[offsets]
        change = level - archive[root].to_numpy(float)[offsets - 1]
        archive_errors["net_magnitude"] = max(archive_errors["net_magnitude"],
                                               float(np.max(np.abs(chosen.net_magnitude.to_numpy() - level))))
        archive_errors["change_magnitude"] = max(archive_errors["change_magnitude"],
                                                  float(np.max(np.abs(chosen.change_magnitude.to_numpy() - change))))
    if max(archive_errors.values()) > 1e-12:
        raise AssertionError(f"Intraday forecasts differ from archived lagged targets: {archive_errors}")
    families = predictors.loc[~predictors.signal.isin(("portfolio_net", "portfolio_finalized_reference"))
                              & ~predictors.signal.str.endswith("__finalized_reference")]
    total = families.groupby(["session", "root"], as_index=False)[
        ["net_magnitude", "change_magnitude"]].sum()
    joined = main.merge(total, on=["session", "root"], suffixes=("_portfolio", "_families"),
                        validate="one_to_one")
    errors = {}
    for field in ("net_magnitude", "change_magnitude"):
        errors[field] = float((joined[f"{field}_portfolio"] - joined[f"{field}_families"]).abs().max())
        if errors[field] > 1e-10:
            raise AssertionError(f"Family weights do not sum to portfolio: {field}")
    return {"predictor_rows": len(predictors), "asset_sessions": len(main),
            "economic_families": int(families.signal.nunique()),
            "max_family_weight_error": errors,
            "max_archived_target_error": archive_errors,
            "latest_completed_observation_used": True}


def audit_price_pairs(moves: pd.DataFrame) -> dict:
    if moves.duplicated(["session", "root", "clock", "hours"]).any():
        raise AssertionError("Duplicate attempted price pair")
    matched = moves.loc[moves.status.eq("matched")].copy()
    for field in ("entry_offset_min", "exit_offset_min"):
        if matched[field].abs().gt(10 + 1e-9).any():
            raise AssertionError(f"Price outside allowed entry/exit window: {field}")
    if not matched.entry_price.gt(0).all() or not matched.exit_price.gt(0).all():
        raise AssertionError("Invalid matched price")
    if not matched.exit_ts.gt(matched.entry_ts).all():
        raise AssertionError("Exit does not follow entry")
    if matched.crosses_maintenance.any():
        raise AssertionError("Matched hold crosses scheduled maintenance")
    verified = matched.loc[matched.identity_basis.eq("verified_month")]
    if not verified.month_matches_target.eq(True).all():
        raise AssertionError("Verified-month entry does not match selected contract")
    return_error = (matched.exit_price / matched.entry_price - 1 - matched.price_return).abs().max()
    if return_error > 1e-12:
        raise AssertionError(f"Stored price return arithmetic error: {return_error}")

    bars, provenance = load_local_bars(pd.bdate_range(moves.session.min(), moves.session.max()))
    lookup = bars[["table", "symbol", "ts", "open", "local_symbol", "volume", "bar_count"]]
    pair = matched[["source_table", "symbol", "entry_ts", "exit_ts", "entry_price",
                    "exit_price", "local_symbol", "entry_trade_confirmed",
                    "exit_trade_confirmed"]].copy()
    for side in ("entry", "exit"):
        source = lookup.rename(columns={"table": "source_table", "ts": f"{side}_ts",
                                        "open": f"{side}_cached_open",
                                        "local_symbol": f"{side}_cached_contract",
                                        "volume": f"{side}_cached_volume",
                                        "bar_count": f"{side}_cached_count"})
        pair = pair.merge(source, on=["source_table", "symbol", f"{side}_ts"], how="left",
                          validate="many_to_one")
        if pair[f"{side}_cached_open"].isna().any():
            raise AssertionError(f"Matched {side} bar missing from local cache")
        np.testing.assert_allclose(pair[f"{side}_price"], pair[f"{side}_cached_open"],
                                   rtol=0, atol=1e-12)
        if not pair.local_symbol.str.upper().eq(pair[f"{side}_cached_contract"].str.upper()).all():
            raise AssertionError(f"{side} bar changed listed contract")
        traded = pair[f"{side}_cached_volume"].gt(0) | pair[f"{side}_cached_count"].gt(0)
        if not traded.eq(pair[f"{side}_trade_confirmed"].astype(bool)).all():
            raise AssertionError(f"{side} trade confirmation disagrees with cache")
    return {"attempted_pairs": len(moves), "matched_pairs": len(matched),
            "verified_month_pairs": len(verified),
            "trade_confirmed_pairs": int((matched.entry_trade_confirmed.astype(bool)
                                          & matched.exit_trade_confirmed.astype(bool)).sum()),
            "max_stored_return_error": float(return_error),
            "max_entry_offset_minutes": float(matched.entry_offset_min.abs().max()),
            "max_exit_offset_minutes": float(matched.exit_offset_min.abs().max()),
            "local_cached_rows": provenance["valid_price_rows"]}


def leg_values(joined: pd.DataFrame, variant: str) -> pd.DataFrame:
    base = variant.removeprefix("reverse_")
    direction = -1 if variant.startswith("reverse_") else 1
    weight = direction * joined[base].to_numpy(float)
    inactive = np.abs(weight) <= 1e-12
    matched = joined.status.eq("matched").to_numpy()
    ready = matched | inactive
    entry = joined.entry_price.to_numpy(float)
    ret = joined.price_return.to_numpy(float)
    multiplier = joined.root.map(MULTIPLIER).to_numpy(float)
    tick = joined.root.map(TICK_COST).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gross = np.where(inactive, 0., weight * ret)
        commission = np.where(inactive, 0., np.abs(weight) * ROUND_TRIP_COMMISSION
                              / (entry * multiplier))
        slippage = np.where(inactive, 0., np.abs(weight) * 2 * tick / entry)
    gross = np.where(ready, gross, np.nan)
    commission = np.where(ready, commission, np.nan)
    slippage = np.where(ready, slippage, np.nan)
    legs = joined[["session", "clock", "hours", "root", "identity_basis",
                   "trade_pair_confirmed"]].copy()
    legs["variant"] = variant
    legs["weight"] = weight
    legs["ready"] = ready
    legs["gross_return"] = gross
    legs["commission_return"] = commission
    legs["slippage_return"] = slippage
    legs["net_return"] = gross - commission
    legs["slippage_net_return"] = gross - commission - slippage
    return legs


def totals_for_book(legs: pd.DataFrame, book: str) -> pd.DataFrame:
    roots = BOOKS[book]
    selected = legs.loc[legs.root.isin(roots)]
    grouped = selected.groupby(["session", "clock", "hours"], sort=True)
    ready = grouped.ready.all() & grouped.root.nunique().eq(len(roots))
    values = grouped[list(RETURN_FIELDS)].sum(min_count=len(roots)).where(ready, np.nan)
    values["gross_exposure"] = selected.weight.abs().groupby(
        [selected.session, selected.clock, selected.hours], sort=True).sum()
    values["ready"] = ready
    return values.reset_index()


def sliced_stats(values: pd.DataFrame, *, book: str, signal: str, variant: str,
                 extra: dict | None = None) -> list[dict]:
    result = []
    for (clock, hours), group in values.groupby(["clock", "hours"], sort=False):
        for period, mask in (("full", np.ones(len(group), dtype=bool)),
                             ("train", group.session.lt(SPLIT).to_numpy()),
                             ("test", group.session.ge(SPLIT).to_numpy())):
            part = group.loc[mask]
            for basis in ("net_return", "slippage_net_return"):
                row = {"clock": clock, "signal": signal, "variant": variant,
                       "hours": int(hours), "book": book, "basis": basis,
                       "period": period, "possible_days": len(part),
                       **independent_stats(part[basis])}
                if extra:
                    row.update(extra)
                result.append(row)
    return result


def audit_replay(predictors: pd.DataFrame, moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, daily_rows, asset_rows = [], [], []
    for signal, frame in predictors.groupby("signal", sort=False):
        if signal.endswith("__finalized_reference"):
            continue
        joined = moves.merge(frame.drop(columns="signal"), on=["session", "root"],
                             suffixes=("_move", "_signal"), validate="many_to_one")
        if not joined.signal_date_move.eq(joined.signal_date_signal).all():
            raise AssertionError(f"Price and forecast dates differ: {signal}")
        for variant in VARIANTS:
            legs = leg_values(joined, variant)
            for book in (("four", "five") if signal == "portfolio_net" else ("four",)):
                totals = totals_for_book(legs, book)
                metric_rows.extend(sliced_stats(totals, book=book, signal=signal,
                                                variant=variant))
                if signal == "portfolio_net" and book == "four":
                    daily_rows.append(totals.assign(signal=signal, variant=variant, book=book))
            for root, group in legs.groupby("root", sort=False):
                for (clock, hours), values in group.groupby(["clock", "hours"], sort=False):
                    ordered = values.sort_values("session")
                    row = {"signal": signal, "variant": variant, "root": root,
                           "clock": clock, "hours": int(hours),
                           "matched_days": int(ordered.ready.sum()),
                           "trade_confirmed_days": int((ordered.ready & ordered.trade_pair_confirmed.eq(True)).sum())}
                    for label, subset in (("full", ordered),
                                          ("earlier", ordered.loc[ordered.session.lt(SPLIT)]),
                                          ("later", ordered.loc[ordered.session.ge(SPLIT)])):
                        stats = independent_stats(subset.slippage_net_return)
                        row.update({f"{key}_{label}": value for key, value in stats.items()})
                    asset_rows.append(row)
        print(f"[audit] {signal}: 8 variants x 6 clocks x 6 holds", flush=True)
    return pd.DataFrame(metric_rows), pd.concat(daily_rows, ignore_index=True), pd.DataFrame(asset_rows)


def compare_replay(independent: pd.DataFrame, saved: pd.DataFrame,
                   fields: tuple[str, ...], keys: list[str], label: str) -> dict:
    if independent.duplicated(keys).any() or saved.duplicated(keys).any():
        raise AssertionError(f"{label}: duplicate keys")
    matched = independent.merge(saved, on=keys, how="outer", validate="one_to_one",
                                suffixes=("_independent", "_saved"), indicator=True)
    if not matched._merge.eq("both").all():
        raise AssertionError(f"{label}: missing or extra rows: {matched._merge.value_counts().to_dict()}")
    left = matched.rename(columns={f"{field}_independent": field for field in fields})
    right = matched.rename(columns={f"{field}_saved": field for field in fields})
    return {"rows": len(matched), "max_absolute_error": assert_close_columns(left, right, fields, label)}


def candidate_diagnostics(daily: pd.DataFrame, moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, root_rows = [], []
    for hours in (1, 2, 3, 4, 6):
        pair = moves.loc[moves.clock.eq("us_open") & moves.hours.eq(hours)
                         & moves.root.isin(BOOKS["four"])]
        status = pair.pivot(index="session", columns="root", values="status").eq("matched").all(axis=1)
        traded = pair.pivot(index="session", columns="root", values="trade_pair_confirmed").eq(True).all(axis=1)
        exact = pair.pivot(index="session", columns="root", values="entry_offset_min").eq(0).all(axis=1)
        selected = daily.loc[daily.clock.eq("us_open") & daily.hours.eq(hours)
                             & daily.variant.eq("reverse_change_magnitude")]
        values = selected.set_index("session").slippage_net_return.sort_index()
        for subset, mask in (("all matched", status),
                             ("all four trade-confirmed", status & traded),
                             ("all four exact entry", status & exact),
                             ("exact entry and trade-confirmed", status & exact & traded),
                             ("historical rank-route proxy", status & (status.index < IDENTITY_BOUNDARY)),
                             ("recent verified month", status & (status.index >= IDENTITY_BOUNDARY))):
            series = values.reindex(mask.index)[mask]
            rows.append({"hours": hours, "subset": subset, **independent_stats(series)})
        for root, group in pair.groupby("root"):
            root_rows.append({"hours": hours, "root": root,
                              "matched_days": int(group.status.eq("matched").sum()),
                              "trade_confirmed_days": int(group.trade_pair_confirmed.eq(True).sum()),
                              "verified_month_matched_days": int((group.identity_basis.eq("verified_month")
                                                                    & group.status.eq("matched")).sum())})
    return pd.DataFrame(rows), pd.DataFrame(root_rows)


def candidate_attribution(predictors: pd.DataFrame, moves: pd.DataFrame,
                          daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                        pd.DataFrame, dict]:
    asset_rows, family_rows, family_asset_rows, asset_daily_rows, checks = [], [], [], [], {}
    families = sorted(set(predictors.signal)
                      - {"portfolio_net", "portfolio_finalized_reference"}
                      - {name for name in predictors.signal if name.endswith("__finalized_reference")})
    for hours in (4, 6):
        moves_here = moves.loc[moves.clock.eq("us_open") & moves.hours.eq(hours)
                                & moves.root.isin(BOOKS["four"])]
        reference = daily.loc[daily.clock.eq("us_open") & daily.hours.eq(hours)
                              & daily.variant.eq("reverse_change_magnitude")]
        complete = reference.loc[reference.ready].set_index("session").sort_index()
        assert len(complete) == (551 if hours == 4 else 537)
        used = moves_here.loc[moves_here.session.isin(complete.index)]
        gross_components = []
        for signal in ("portfolio_net", *families):
            forecast = predictors.loc[predictors.signal.eq(signal)].drop(columns="signal")
            joined = used.merge(forecast, on=["session", "root"],
                                suffixes=("_move", "_signal"), validate="many_to_one")
            legs = leg_values(joined, "reverse_change_magnitude")
            if not legs.ready.all():
                raise AssertionError(f"Incomplete candidate legs for {signal}/{hours}h")
            if signal == "portfolio_net":
                portfolio_legs = legs
                by_day = legs.groupby("session")[list(RETURN_FIELDS)].sum()
                compare = complete[list(RETURN_FIELDS)].reindex(by_day.index)
                assert_close_columns(by_day, compare, RETURN_FIELDS,
                                     f"candidate asset sum {hours}h")
                for root, asset in legs.groupby("root"):
                    ordered = asset.sort_values("session")
                    series = ordered.slippage_net_return
                    asset_daily_rows.append(ordered[["session", "root", "slippage_net_return"]]
                                            .assign(hours=hours))
                    asset_rows.append({"hours": hours, "root": root, "days": len(series),
                                       "net_contribution": float(series.sum()),
                                       "gross_contribution": float(ordered.gross_return.sum()),
                                       "commission_drag": float(ordered.commission_return.sum()),
                                       "slippage_drag": float(ordered.slippage_return.sum()),
                                       "trade_confirmed_days": int(ordered.trade_pair_confirmed.eq(True).sum()),
                                       **independent_stats(series)})
            else:
                family_daily = legs.groupby("session")[list(RETURN_FIELDS)].sum()
                gross_components.append(family_daily.gross_return.rename(signal))
                family_rows.append({"hours": hours, "signal": signal, "days": len(family_daily),
                                    "gross_contribution": float(family_daily.gross_return.sum()),
                                    "standalone_net_contribution": float(family_daily.slippage_net_return.sum()),
                                    "standalone_cost_drag": float((family_daily.commission_return
                                                                    + family_daily.slippage_return).sum()),
                                    "gross_sharpe": independent_stats(family_daily.gross_return)["sharpe"],
                                    "standalone_net_sharpe": independent_stats(
                                        family_daily.slippage_net_return)["sharpe"]})
                for root, asset in legs.groupby("root"):
                    family_asset_rows.append({"hours": hours, "signal": signal, "root": root,
                                              "gross_contribution": float(asset.gross_return.sum())})
        combined = pd.concat(gross_components, axis=1)
        portfolio_gross = portfolio_legs.groupby("session").gross_return.sum()
        maximum = float((combined.sum(axis=1) - portfolio_gross).abs().max())
        if maximum > 1e-10:
            raise AssertionError(f"Family gross P&L does not sum to portfolio: {maximum}")
        cost = float((portfolio_legs.commission_return + portfolio_legs.slippage_return).sum())
        checks[str(hours)] = {"days": len(complete), "max_daily_gross_attribution_error": maximum,
                              "portfolio_gross_contribution": float(portfolio_gross.sum()),
                              "portfolio_cost_drag": cost,
                              "portfolio_net_contribution": float(complete.slippage_net_return.sum())}
        if not np.isclose(checks[str(hours)]["portfolio_gross_contribution"] - cost,
                          checks[str(hours)]["portfolio_net_contribution"], atol=1e-12):
            raise AssertionError("Portfolio gross minus costs does not equal net")
    return (pd.DataFrame(asset_rows), pd.DataFrame(family_rows),
            pd.DataFrame(family_asset_rows), pd.concat(asset_daily_rows, ignore_index=True), checks)


def candidate_leave_one_out(asset_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hours, group in asset_daily.groupby("hours"):
        wide = group.pivot(index="session", columns="root", values="slippage_net_return")
        if wide.isna().any().any():
            raise AssertionError("Candidate attribution is not on common portfolio dates")
        for omitted in ("none", *BOOKS["four"]):
            remaining = wide if omitted == "none" else wide.drop(columns=omitted)
            rows.append({"hours": int(hours), "omitted_asset": omitted,
                         **independent_stats(remaining.sum(axis=1))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-only", action="store_true",
                        help="Use the already reconciled independent daily ledger")
    parser.add_argument("--lineage-only", action="store_true",
                        help="Recheck dated forecasts against the archived daily targets")
    args = parser.parse_args()
    if args.attribution_only and args.lineage_only:
        parser.error("Choose only one focused audit mode")
    OUT.mkdir(exist_ok=True)
    predictors = pd.read_parquet(SOURCE / "dated_predictors.parquet")
    moves = pd.read_parquet(SOURCE / "price_moves.parquet")
    if args.lineage_only:
        targets = pd.read_parquet(SOURCE / "contract_targets.parquet")
        lineage = audit_signal_lineage(predictors, targets)
        summary_path = OUT / "audit_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["lineage"] = lineage
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(lineage, indent=2), flush=True)
        return
    if args.attribution_only:
        daily = pd.read_parquet(OUT / "independent_portfolio_daily.parquet")
        cases, roots = candidate_diagnostics(daily, moves)
        asset_attribution, family_attribution, family_asset, asset_daily, checks = candidate_attribution(
            predictors, moves, daily)
        leave_one_out = candidate_leave_one_out(asset_daily)
        cases.to_csv(OUT / "portfolio_subsets.csv", index=False)
        roots.to_csv(OUT / "asset_price_coverage.csv", index=False)
        asset_attribution.to_csv(OUT / "candidate_asset_attribution.csv", index=False)
        family_attribution.to_csv(OUT / "candidate_family_attribution.csv", index=False)
        family_asset.to_csv(OUT / "candidate_family_asset_attribution.csv", index=False)
        asset_daily.to_parquet(OUT / "candidate_asset_daily.parquet", index=False)
        leave_one_out.to_csv(OUT / "candidate_leave_one_out.csv", index=False)
        summary_path = OUT / "audit_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["candidate_attribution"] = checks
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(asset_attribution.to_string(index=False), flush=True)
        print(cases.to_string(index=False), flush=True)
        print(f"[audit] candidate attribution saved to {OUT}", flush=True)
        return
    targets = pd.read_parquet(SOURCE / "contract_targets.parquet")
    saved_daily = pd.read_parquet(SOURCE / "portfolio_daily_returns.parquet")
    saved_metrics = pd.read_csv(SOURCE / "metrics.csv")
    lineage = audit_signal_lineage(predictors, targets)
    print("[audit] forecast lineage and family sum checked", flush=True)
    price = audit_price_pairs(moves)
    print("[audit] matched prices verified against local cached bars", flush=True)
    metrics, daily, assets = audit_replay(predictors, moves)
    daily_check = compare_replay(daily, saved_daily,
                                 (*RETURN_FIELDS, "gross_exposure", "ready"), KEYS,
                                 "portfolio daily ledger")
    metric_check = compare_replay(metrics, saved_metrics, METRIC_FIELDS, METRIC_KEYS,
                                  "all saved metrics")
    cases, roots = candidate_diagnostics(daily, moves)
    asset_attribution, family_attribution, family_asset, asset_daily, attribution_check = candidate_attribution(
        predictors, moves, daily)
    leave_one_out = candidate_leave_one_out(asset_daily)
    assets.to_csv(OUT / "signal_asset_metrics.csv", index=False)
    cases.to_csv(OUT / "portfolio_subsets.csv", index=False)
    roots.to_csv(OUT / "asset_price_coverage.csv", index=False)
    asset_attribution.to_csv(OUT / "candidate_asset_attribution.csv", index=False)
    family_attribution.to_csv(OUT / "candidate_family_attribution.csv", index=False)
    family_asset.to_csv(OUT / "candidate_family_asset_attribution.csv", index=False)
    asset_daily.to_parquet(OUT / "candidate_asset_daily.parquet", index=False)
    leave_one_out.to_csv(OUT / "candidate_leave_one_out.csv", index=False)
    daily.to_parquet(OUT / "independent_portfolio_daily.parquet", index=False)
    summary = {"offline_only": True, "lineage": lineage, "price_pairs": price,
               "portfolio_daily": daily_check, "all_metrics": metric_check,
               "signal_asset_metric_rows": len(assets), "candidate_attribution": attribution_check}
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[audit] full daily and metric grid reconciled", flush=True)
    print(cases.loc[cases.hours.isin((4, 6))].to_string(index=False), flush=True)
    print(asset_attribution.to_string(index=False), flush=True)
    print("[audit] outputs:", OUT, flush=True)


if __name__ == "__main__":
    main()
