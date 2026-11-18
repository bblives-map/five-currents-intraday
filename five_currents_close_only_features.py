"""Frozen flow-state/updated-close counterfactual for the existing alpha roster."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from five_currents_opening_study import PRICE_CONCEPTS


def close_only_concepts(original, operational, baseline_concepts):
    from five_currents_daily.curve_policy import CURVE_RANK_LIMITS
    from five_currents_daily.engine import DEFAULT_SOURCE_DIR
    from run_us_commodity_fixed_vintage_robustness import candidate_weights_from_row
    from us_commodity_broad_technical_core import _causal_zscore
    from us_commodity_carry_tmv_expansion_core import (
        _dynamic_curve_statistics, _dynamic_curve_sum, _safe_zscore,
    )
    from us_commodity_execution_overlays import build_execution_overlays
    from us_commodity_front_roll_core import install_execution_ranks
    from us_commodity_full_signal_core import build_product_signal_data
    from us_commodity_research_core import PRODUCT_ORDER

    # Imports are kept local because this helper never needs database access.
    from us_commodity_research_core import _field_matrix

    library = pd.read_csv(DEFAULT_SOURCE_DIR / "neighborhood_library.csv")
    library = library.loc[~library.economic_family.eq("fibonacci_reversion")].copy()
    panel = install_execution_ranks(original.panel, original.combined.wide,
                                   original.combined.mapping, original.signal_ranks)
    overlays = build_execution_overlays(panel, original.combined.wide, original.combined.mapping)
    data = build_product_signal_data(original.combined.wide, original.combined.mapping, panel)
    technical_ids = sorted(library.loc[library.effective_source.eq("technical"), "signal_id"].unique())
    carry_ids = sorted(library.loc[library.economic_family.isin(("carry_oi_reversal", "aggregate_volume_carry")), "signal_id"].unique())
    current, frozen = {}, {}
    for root in PRODUCT_ORDER:
        bars = data.bars[root]
        log_close = np.log(bars.close.where(bars.close.gt(0)))
        daily_return = log_close.diff()
        volatility = daily_return.rolling(20, min_periods=12).std(ddof=0)
        volume = bars.volume.where(bars.volume.gt(0))
        oi = bars.open_interest.where(bars.open_interest.gt(0))
        for signal_id in technical_ids:
            direction = -1. if signal_id.endswith("__reversion") else 1.
            base_id = signal_id.rsplit("__", 1)[0]
            if match := re.fullmatch(r"obv_divergence_w(\d+)", base_id):
                window = int(match[1])
                minimum = max(3, int(math.ceil(.60*window)))
                signed = np.sign(daily_return).fillna(0)*volume.fillna(0)
                flow = signed.rolling(window, min_periods=minimum).sum().div(
                    volume.rolling(window, min_periods=minimum).sum().replace(0, np.nan))
                price = np.tanh(log_close.diff(window).div(
                    volatility.mul(math.sqrt(window)).replace(0, np.nan)).clip(-8, 8))
                score = flow-price
                held_score = flow.shift(1)-price
            elif match := re.fullmatch(r"volume_(spike|dryup)_v(\d+)_z([\d.]+)_h(\d+)", base_id):
                kind, window, threshold, horizon = match.groups()
                window, threshold, horizon = int(window), float(threshold), int(horizon)
                volume_z = _causal_zscore(np.log1p(volume), window)
                strength = ((volume_z if kind == "spike" else -volume_z)-threshold).clip(lower=0)
                gate = strength/(1.+strength)
                price = log_close.diff(horizon).div(
                    volatility.mul(math.sqrt(horizon)).replace(0, np.nan))
                score = price*gate
                held_score = price*gate.shift(1)
            elif match := re.fullmatch(r"oi_direction_h(\d+)", base_id):
                horizon = int(match[1])
                score = _causal_zscore(np.log(oi).diff(horizon), max(126, 4*horizon))
                held_score = score.shift(1)
            else:
                raise ValueError("Unsupported existing technical alpha: " + signal_id)
            current.setdefault(signal_id, {})[root] = direction*np.tanh(score.clip(-8, 8))
            frozen.setdefault(signal_id, {})[root] = direction*np.tanh(held_score.clip(-8, 8))
        price = _field_matrix(original.combined.wide, root, "PX_SETTLE")
        price = price.where(price.gt(0), _field_matrix(original.combined.wide, root, "PX_LAST"))
        curve_oi = _field_matrix(original.combined.wide, root, "OPEN_INT")
        curve_volume = _field_matrix(original.combined.wide, root, "PX_VOLUME")
        limit = CURVE_RANK_LIMITS[root]
        price = price.where(np.broadcast_to(price.columns <= limit, price.shape))
        curve_oi = curve_oi.where(np.broadcast_to(curve_oi.columns <= limit, curve_oi.shape))
        curve_volume = curve_volume.where(np.broadcast_to(curve_volume.columns <= limit, curve_volume.shape))

        def z(values):
            return _safe_zscore(pd.DataFrame({root: values}))[root]

        for signal_id in carry_ids:
            if match := re.fullmatch(r"carry2__slope_oi_d(\d+)_h(\d+)", signal_id):
                depth, horizon = map(int, match.groups())
                slope = z(_dynamic_curve_statistics(price, original.signal_ranks[root], depth).slope)
                total = _dynamic_curve_sum(curve_oi, original.signal_ranks[root], depth)
                flow = np.tanh(z(np.log1p(total.clip(lower=0)).diff(horizon)))
                value, held = slope*flow, slope*flow.shift(1)
            elif match := re.fullmatch(r"carry2__aggregate_volume_d(\d+)_h(\d+)", signal_id):
                depth, horizon = map(int, match.groups())
                total = _dynamic_curve_sum(curve_volume, original.signal_ranks[root], depth)
                value = z(np.log1p(total.clip(lower=0)).diff(horizon))
                held = value.shift(1)
            else:
                raise ValueError("Unsupported existing flow curve alpha: " + signal_id)
            current.setdefault(signal_id, {})[root] = value.clip(-8, 8)
            frozen.setdefault(signal_id, {})[root] = held.clip(-8, 8)
    current = {key: pd.DataFrame(value).reindex(columns=PRODUCT_ORDER) for key, value in current.items()}
    frozen = {key: pd.DataFrame(value).reindex(columns=PRODUCT_ORDER) for key, value in frozen.items()}

    def rebuild(features):
        values = baseline_concepts.copy()
        lookup = {}
        for signal_id, frame in features.items():
            source = "carry_expanded" if signal_id.startswith("carry2__") else "technical"
            key = "close_refresh__" + signal_id
            panel.product_features[key] = frame
            lookup[(source, signal_id)] = key
        regions = {}
        for family, rows in library.groupby("economic_family", sort=False):
            if family in PRICE_CONCEPTS or family in ("carry_basis_directional", "basis_momentum_carry"):
                continue
            regions[family] = np.mean([candidate_weights_from_row(panel, overlays, lookup, row)
                                      for row in rows.itertuples(index=False)], axis=0)
        regions["obv_divergence"] = .5*(regions.pop("obv_continuation")+regions.pop("obv_reversion"))
        for name, weights in regions.items():
            values[operational.concept_names.index(name)] = weights
        return values

    # Verify the copied factorization before using its held-flow counterfactual.
    control = rebuild(current)
    np.testing.assert_allclose(control, baseline_concepts, atol=1e-10, equal_nan=True)
    return rebuild(frozen)
