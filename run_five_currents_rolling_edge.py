"""Past-only intraday family selection and conservative asset-edge tilt.

The intraday recipe for each asset/family is the archived hindsight winner.
Only the rolling allocation and selection in this module are causal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_five_currents_two_books_report import (
    AUDIT, BOOKS, PRICE_SOURCE, ROOT, SOURCE, leg_path,
)
from five_currents_execution_timing import ROOTS
from five_currents_group_clock_core import row_stats
from run_five_currents_group_clock import bundle


OUT = ROOT / "five_currents_rolling_edge_20261024"
ASSETS = ("CL", "NG", "GC", "HG")
WARMUP = 63
FAMILY_WINDOWS = (63, 126, 252, 504)
FAMILY_CADENCES = (5, 10, 21, 63)
ASSET_WINDOWS = (63, 126, 252, 504)
ASSET_CADENCES = (5, 21, 63)
PRIMARY = (252, 21)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates():
    with np.load(AUDIT / "recomputed_votes.npz") as saved:
        raw = saved["gc4_lagged"]
    arrays, names, sessions, dates = bundle(SOURCE)
    with np.load(PRICE_SOURCE) as saved:
        prices = {key: saved[key] for key in saved.files}
    winners = pd.read_parquet(AUDIT / "individual_hindsight_winners.parquet")
    winners = (winners.loc[winners.state.eq("gc4") & winners.root.isin(ASSETS)]
               .sort_values(["root", "family"]).reset_index(drop=True))
    n, t = len(winners), len(sessions)
    gross = np.empty((t, n))
    cost = np.empty_like(gross)
    planned = np.empty((t, n), dtype=bool)
    for j, row in enumerate(winners.itertuples(index=False)):
        leg_gross, leg_cost, leg_planned = leg_path(
            row, row.root, row.family, raw, dates, sessions, prices, names)
        gross[:, j] = leg_gross[0]
        cost[:, j] = leg_cost[0]
        planned[:, j] = leg_planned
    if t != len(sessions) or n == 0 or winners.duplicated(["root", "family"]).any():
        raise AssertionError("Candidate session alignment or root/family uniqueness failed")
    return sessions, winners, gross, cost, planned


def trailing_scores(gross, cost, planned, start, end):
    net = gross[start:end] - cost[start:end]
    stress = gross[start:end] - 2 * cost[start:end]
    active = planned[start:end]
    scores = np.zeros(net.shape[1])
    midpoint = len(net) // 2
    for j in range(net.shape[1]):
        valid = np.isfinite(net[:, j])
        intended = int(active[:, j].sum())
        priced_active = int((active[:, j] & valid).sum())
        if (intended < 8 or priced_active < 8 or priced_active / intended < .75
                or int(valid.sum()) < 40):
            continue
        returns = net[valid, j]
        deviation = returns.std(ddof=1)
        if deviation <= 1e-12 or returns.mean() <= 0:
            continue
        recent = net[midpoint:, j]
        recent = recent[np.isfinite(recent)]
        stressed = stress[:, j]
        stressed = stressed[np.isfinite(stressed)]
        if not len(recent) or recent.mean() <= 0 or not len(stressed) or stressed.mean() <= 0:
            continue
        sr = returns.mean() / deviation * np.sqrt(252)
        scores[j] = min(sr, 3.0) * np.sqrt(priced_active / (priced_active + 20))
    return scores


def weighted_assets(weights, gross, cost, asset_indices):
    t = len(weights)
    result_gross = np.zeros((t, len(asset_indices)))
    result_cost = np.zeros_like(result_gross)
    activity = np.zeros_like(result_gross, dtype=bool)
    for a, indices in enumerate(asset_indices):
        selected = weights[:, indices] > 0
        activity[:, a] = selected.any(axis=1)
        missing = (selected & (~np.isfinite(gross[:, indices]) |
                               ~np.isfinite(cost[:, indices]))).any(axis=1)
        result_gross[:, a] = np.sum(
            weights[:, indices] * np.where(np.isfinite(gross[:, indices]),
                                            gross[:, indices], 0), axis=1)
        result_cost[:, a] = np.sum(
            weights[:, indices] * np.where(np.isfinite(cost[:, indices]),
                                            cost[:, indices], 0), axis=1)
        result_gross[missing, a] = np.nan
        result_cost[missing, a] = np.nan
    return result_gross, result_cost, activity


def family_replay(gross, cost, planned, winners, *, lookback, cadence,
                  top_k=2, freeze_after_first=False):
    t, n = gross.shape
    asset_indices = [np.flatnonzero(winners.root.to_numpy() == root) for root in ASSETS]
    weights = np.zeros((t, n))
    scores = np.zeros(n)
    refits = []
    for day in range(t):
        if day >= WARMUP and (day == WARMUP or
                              (not freeze_after_first and (day - WARMUP) % cadence == 0)):
            scores = trailing_scores(gross, cost, planned,
                                     max(0, day - lookback), day)
            refits.append(day)
        for indices in asset_indices:
            today = indices[planned[day, indices]]
            if not len(today):
                continue
            if day < WARMUP:
                weights[day, today] = 1 / len(today)
                continue
            ranked = today[np.argsort(-scores[today], kind="stable")]
            picked = ranked[scores[ranked] > 0][:top_k]
            if len(picked):
                weights[day, picked] = scores[picked] / scores[picked].sum()
    asset_gross, asset_cost, active = weighted_assets(weights, gross, cost, asset_indices)
    return {"candidate_weights": weights, "asset_gross": asset_gross,
            "asset_cost": asset_cost, "asset_active": active,
            "refit_days": np.array(refits, dtype=int)}


def asset_edge_weights(asset_gross, asset_cost, *, lookback, cadence):
    net = asset_gross - asset_cost
    t, n_assets = net.shape
    weights = np.full((t, n_assets), 1 / n_assets)
    current = weights[0].copy()
    for day in range(WARMUP, t):
        if day == WARMUP or (day - WARMUP) % cadence == 0:
            trail = net[max(0, day - lookback):day]
            scores = np.ones(n_assets)
            for a in range(n_assets):
                observed = trail[:, a]
                observed = observed[np.isfinite(observed)]
                if len(observed) < 40:
                    continue
                vol = observed.std(ddof=1)
                if vol <= 1e-12:
                    continue
                sr = observed.mean() / vol * np.sqrt(252)
                scores[a] = 1 + .25 * np.clip(sr, -2, 2)
                if len(observed) / len(trail) < .75:
                    scores[a] = min(scores[a], .75)
            # Half the budget remains equally allocated, so this cannot become
            # an unconstrained switch between commodities or add leverage.
            current = .5 / n_assets + .5 * scores / scores.sum()
        weights[day] = current
    return weights


def portfolio(asset_gross, asset_cost, asset_active, asset_weights):
    missing = (~np.isfinite(asset_gross) | ~np.isfinite(asset_cost)) & (asset_weights > 0)
    gross = np.sum(asset_weights * np.where(np.isfinite(asset_gross), asset_gross, 0), axis=1)
    cost = np.sum(asset_weights * np.where(np.isfinite(asset_cost), asset_cost, 0), axis=1)
    gross[missing.any(axis=1)] = np.nan
    cost[missing.any(axis=1)] = np.nan
    planned = (asset_active & (asset_weights > 0)).any(axis=1)
    return gross, cost, planned


def performance(name, gross, cost, planned, sessions, baseline_net=None):
    net = gross - cost
    observed = np.isfinite(net)
    priced_active = int((planned & observed).sum())
    row = {"strategy": name, "net_sr": row_stats(net)["sharpe"],
           "stress_sr": row_stats(gross - 2 * cost)["sharpe"],
           "annual_vol": row_stats(net)["annual_vol"],
           "observed_growth": row_stats(net)["observed_total_return"],
           "max_drawdown": row_stats(net)["observed_max_drawdown"],
           "planned_days": int(planned.sum()), "priced_days": priced_active,
           "coverage": priced_active / planned.sum() if planned.any() else np.nan,
           "sr_after_warmup": row_stats(net[WARMUP:])["sharpe"]}
    for year in (2024, 2025, 2026):
        row[f"sr_{year}"] = row_stats(net[sessions.year == year])["sharpe"]
    late = sessions >= pd.Timestamp("2026-08-01")
    row["late_sr"] = row_stats(net[late])["sharpe"]
    row["late_growth"] = row_stats(net[late])["observed_total_return"]
    row["late_priced_days"] = int((planned & observed & late).sum())
    if baseline_net is not None:
        common = observed & np.isfinite(baseline_net)
        row["common_days"] = int(common.sum())
        row["common_sr"] = row_stats(net[common])["sharpe"]
        row["baseline_common_sr"] = row_stats(baseline_net[common])["sharpe"]
        row["common_sr_delta"] = row["common_sr"] - row["baseline_common_sr"]
        common_late = common & late
        row["common_late_days"] = int(common_late.sum())
        row["common_late_growth"] = row_stats(net[common_late])["observed_total_return"]
        row["baseline_common_late_growth"] = row_stats(
            baseline_net[common_late])["observed_total_return"]
    return row


def fixed_book(members, winners, gross, cost, planned):
    indices = [int(winners.index[(winners.root == root) & (winners.family == family)][0])
               for root, family in members.items()]
    out_gross = np.mean(gross[:, indices], axis=1)
    out_cost = np.mean(cost[:, indices], axis=1)
    out_planned = planned[:, indices].any(axis=1)
    return out_gross, out_cost, out_planned


def run():
    OUT.mkdir(exist_ok=True)
    sessions, winners, gross, cost, planned = load_candidates()
    np.savez_compressed(OUT / "candidate_paths.npz", gross=gross, cost=cost,
                        planned=planned, sessions=sessions.to_numpy(dtype="datetime64[ns]"))
    winners.to_parquet(OUT / "candidate_metadata.parquet", index=False)
    incumbent = fixed_book(BOOKS["two_books_carry_oi"], winners, gross, cost, planned)
    incumbent_energy = fixed_book(BOOKS["energy_carry_oi"], winners,
                                  gross, cost, planned)
    spike_energy = fixed_book(BOOKS["energy_spike"], winners,
                              gross, cost, planned)
    spike = fixed_book(BOOKS["two_books_equal"], winners, gross, cost, planned)
    incumbent_net = incumbent[0] - incumbent[1]
    rows = [performance("fixed_carry_oi", *incumbent, sessions),
            performance("fixed_volume_spike", *spike, sessions, incumbent_net)]
    daily = pd.DataFrame({"session": sessions,
                          "fixed_carry_oi": incumbent_net,
                          "fixed_volume_spike": spike[0] - spike[1],
                          "fixed_energy": incumbent_energy[0] - incumbent_energy[1],
                          "fixed_energy_spike": spike_energy[0] - spike_energy[1]})
    family_cache = {}
    primary_replay = None
    energy_rows = [performance("fixed_energy", *incumbent_energy, sessions),
                   performance("fixed_energy_spike", *spike_energy, sessions,
                               incumbent_energy[0] - incumbent_energy[1])]
    for lookback in FAMILY_WINDOWS:
        for cadence in FAMILY_CADENCES:
            name = f"family_L{lookback}_C{cadence}_K2_equal"
            replay = family_replay(gross, cost, planned, winners,
                                   lookback=lookback, cadence=cadence)
            family_cache[(lookback, cadence)] = replay
            asset_weights = np.full((len(sessions), len(ASSETS)), 1 / len(ASSETS))
            pnl = portfolio(replay["asset_gross"], replay["asset_cost"],
                            replay["asset_active"], asset_weights)
            rows.append(performance(name, *pnl, sessions, incumbent_net))
            daily[name] = pnl[0] - pnl[1]
            energy_gross = replay["asset_gross"][:, :2].mean(axis=1)
            energy_cost = replay["asset_cost"][:, :2].mean(axis=1)
            energy_planned = replay["asset_active"][:, :2].any(axis=1)
            energy_rows.append(performance(name, energy_gross, energy_cost,
                                           energy_planned, sessions,
                                           incumbent_energy[0] - incumbent_energy[1]))
            if (lookback, cadence) in ((63, 5), PRIMARY):
                daily[name + "__energy"] = energy_gross - energy_cost
                for a, root in enumerate(ASSETS):
                    daily[name + "__" + root] = (replay["asset_gross"][:, a]
                                                   - replay["asset_cost"][:, a])
            if (lookback, cadence) == PRIMARY:
                primary_replay = replay
                primary_pnl = pnl
    assert primary_replay is not None
    frozen = family_replay(gross, cost, planned, winners, lookback=252,
                           cadence=21, freeze_after_first=True)
    top1 = family_replay(gross, cost, planned, winners, lookback=252,
                         cadence=21, top_k=1)
    for name, replay in (("one_time_fit_K2_equal", frozen),
                         ("family_L252_C21_K1_equal", top1)):
        weights = np.full((len(sessions), len(ASSETS)), 1 / len(ASSETS))
        pnl = portfolio(replay["asset_gross"], replay["asset_cost"],
                        replay["asset_active"], weights)
        rows.append(performance(name, *pnl, sessions, incumbent_net))
        daily[name] = pnl[0] - pnl[1]
    for family_label, replay in (("L252_C21", primary_replay),
                                 ("L63_C5", family_cache[(63, 5)])):
        for lookback in ASSET_WINDOWS:
            for cadence in ASSET_CADENCES:
                name = f"family_{family_label}_K2_asset_L{lookback}_C{cadence}"
                asset_weights = asset_edge_weights(replay["asset_gross"],
                                                   replay["asset_cost"],
                                                   lookback=lookback, cadence=cadence)
                pnl = portfolio(replay["asset_gross"], replay["asset_cost"],
                                replay["asset_active"], asset_weights)
                rows.append(performance(name, *pnl, sessions, incumbent_net))
                daily[name] = pnl[0] - pnl[1]
                if (lookback, cadence) == PRIMARY and family_label == "L252_C21":
                    primary_asset_weights = asset_weights
                    primary_asset_pnl = pnl
                if (lookback, cadence) == PRIMARY and family_label == "L63_C5":
                    fast_asset_weights = asset_weights
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(OUT / "grid_metrics.parquet", index=False)
    pd.DataFrame(energy_rows).to_parquet(OUT / "energy_metrics.parquet", index=False)
    daily.to_parquet(OUT / "daily_replay.parquet", index=False)
    np.savez_compressed(OUT / "primary_weights.npz",
                        family_weights=primary_replay["candidate_weights"],
                        asset_weights=primary_asset_weights,
                        asset_gross=primary_replay["asset_gross"],
                        asset_cost=primary_replay["asset_cost"],
                        asset_active=primary_replay["asset_active"],
                        primary_gross=primary_pnl[0], primary_cost=primary_pnl[1],
                        tilted_gross=primary_asset_pnl[0],
                        tilted_cost=primary_asset_pnl[1])
    np.savez_compressed(OUT / "weekly_63_weights.npz",
                        family_weights=family_cache[(63, 5)]["candidate_weights"],
                        asset_weights=fast_asset_weights,
                        asset_gross=family_cache[(63, 5)]["asset_gross"],
                        asset_cost=family_cache[(63, 5)]["asset_cost"],
                        asset_active=family_cache[(63, 5)]["asset_active"])
    selections = []
    for name, replay in (("family_L63_C5_K2_equal", family_cache[(63, 5)]),
                         ("family_L252_C21_K2_equal", primary_replay)):
        weights = replay["candidate_weights"]
        for j, row in enumerate(winners.itertuples(index=False)):
            selected = weights[:, j] > 0
            selections.append({"strategy": name, "root": row.root,
                               "family": row.family,
                               "selected_2025": int((selected &
                                   (sessions.year == 2025)).sum()),
                               "selected_2026": int((selected &
                                   (sessions.year == 2026)).sum()),
                               "selected_aug_oct_2026": int((selected &
                                   (sessions >= pd.Timestamp("2026-08-01"))).sum()),
                               "mean_weight_2026": float(weights[
                                   sessions.year == 2026, j].mean())})
    pd.DataFrame(selections).to_parquet(OUT / "family_selection_counts.parquet",
                                        index=False)
    # Perturb only future realized P&L. Historical weights must remain invariant.
    cutoff = len(sessions) - 60
    disturbed = gross.copy()
    disturbed[cutoff:] += 100 * planned[cutoff:]
    perturbed = family_replay(disturbed, cost, planned, winners,
                              lookback=PRIMARY[0], cadence=PRIMARY[1])
    np.testing.assert_allclose(primary_replay["candidate_weights"][:cutoff],
                               perturbed["candidate_weights"][:cutoff], rtol=0, atol=0)
    disturbed_assets = primary_replay["asset_gross"].copy()
    disturbed_assets[cutoff:] += 100
    perturbed_asset_weights = asset_edge_weights(
        disturbed_assets, primary_replay["asset_cost"],
        lookback=PRIMARY[0], cadence=PRIMARY[1])
    np.testing.assert_allclose(primary_asset_weights[:cutoff],
                               perturbed_asset_weights[:cutoff], rtol=0, atol=0)
    (OUT / "manifest.json").write_text(json.dumps({
        "source_sha256": {str(path): sha256(path) for path in (
            AUDIT / "recomputed_votes.npz", AUDIT / "individual_hindsight_winners.parquet",
            PRICE_SOURCE, SOURCE / "sessions.parquet")},
        "sample_start": str(sessions[0].date()),
        "sample_end": str(sessions[-1].date()), "sessions": len(sessions),
        "candidates": len(winners), "family_windows": FAMILY_WINDOWS,
        "family_cadences": FAMILY_CADENCES, "asset_windows": ASSET_WINDOWS,
        "asset_cadences": ASSET_CADENCES, "warmup_sessions": WARMUP,
        "daily_candidate_activation_known_pre_entry": True,
        "future_perturbation_pre_cutoff_weights_unchanged": True,
        "future_perturbation_pre_cutoff_asset_weights_unchanged": True,
        "database_access": False, "market_api_access": False,
        "limitation": "Only rolling allocations are causal. Each candidate's vote treatment, clock and hold were chosen on the full sample in the earlier archive.",
    }, indent=2), encoding="utf-8")
    print(metrics[["strategy", "net_sr", "stress_sr", "sr_2025", "sr_2026",
                   "late_growth", "coverage", "common_sr_delta"]].to_string(index=False))
    print("\nEnergy only:\n" + pd.DataFrame(energy_rows)[[
        "strategy", "net_sr", "sr_2025", "sr_2026", "late_growth",
        "coverage", "common_sr_delta"]].to_string(index=False))


if __name__ == "__main__":
    run()
