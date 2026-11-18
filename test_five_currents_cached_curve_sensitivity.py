import numpy as np
import pandas as pd
from types import SimpleNamespace

from run_five_currents_cached_curve_sensitivity import (
    curve_mask, exact_contract_adv, frozen_allocation, requested_curve_features,
)
from us_commodity_carry_tmv_expansion_core import build_expanded_carry_signals
from us_commodity_research_core import PRODUCT_ORDER


def test_liquidity_selection_uses_only_prior_exact_contract_volume():
    dates = pd.bdate_range("2025-01-01", periods=35)
    volume = pd.DataFrame({rank: np.full(len(dates), rank * 10.) for rank in range(1, 7)}, index=dates)
    mapping = pd.DataFrame({rank: [f"contract{rank}"] * len(dates) for rank in range(1, 7)}, index=dates)
    mask = curve_mask("liquid4", volume, mapping)
    assert mask.iloc[15][[1, 2, 5, 6]].all()
    assert not mask.iloc[15][[3, 4]].any()
    altered = volume.copy()
    altered.loc[dates[15]:, 3] = 1e9
    changed = curve_mask("liquid4", altered, mapping)
    pd.testing.assert_frame_equal(mask.loc[:dates[15]], changed.loc[:dates[15]])
    assert changed.loc[dates[16], 3]


def test_adv_tracks_physical_identity_across_rank_migration():
    dates = pd.bdate_range("2025-01-01", periods=12)
    volume = pd.DataFrame(np.nan, index=dates, columns=[1, 2])
    mapping = pd.DataFrame(None, index=dates, columns=[1, 2])
    volume.loc[dates[:6], 2], mapping.loc[dates[:6], 2] = 40., "same"
    volume.loc[dates[6]:, 1], mapping.loc[dates[6]:, 1] = 40., "same"
    adv = exact_contract_adv(volume, mapping, window=10, minimum=5)
    assert adv.loc[dates[6], 1] == 40.
    assert pd.isna(adv.loc[dates[6], 2])


def test_near_mask_does_not_renumber_or_fill_contracts():
    data = pd.DataFrame(1., index=pd.bdate_range("2025-01-01", periods=3), columns=range(1, 8))
    for count in (3, 4, 5, 6):
        mask = curve_mask("near" + str(count), data, data)
        assert mask.sum(axis=1).eq(count).all()
        assert not mask[count + 1].any()


def test_mixed_mask_changes_only_silver_and_copper():
    data = pd.DataFrame(1., index=pd.bdate_range("2025-01-01", periods=3), columns=range(1, 8))
    for root in PRODUCT_ORDER:
        count = 4 if root in ("SI", "HG") else 6
        mixed = curve_mask("mixed", data, data, root=root)
        expected = curve_mask("near" + str(count), data, data)
        pd.testing.assert_frame_equal(mixed, expected)


def test_frozen_selection_changes_on_original_refit_boundary():
    dates = pd.bdate_range("2025-01-01", periods=4)
    concepts = np.zeros((2, 4, 5))
    concepts[0, :, 0] = .02
    concepts[1, :, 1] = -.02
    selection = pd.DataFrame([
        {"refit_ordinal": ordinal, "forecast_start": dates[start], "forecast_end": dates[end],
         "economic_family": name, "selected": selected}
        for ordinal, start, end in ((1, 0, 1), (2, 2, 3))
        for name, selected in (("A", ordinal == 1), ("B", ordinal == 2))])
    coefficients = pd.DataFrame([{"refit_ordinal": ordinal, "product": root,
                                  "inverse_volatility_coefficient": 1.}
                                 for ordinal in (1, 2) for root in PRODUCT_ORDER])
    actual = frozen_allocation(concepts, ("A", "B"), dates, selection, coefficients)
    np.testing.assert_allclose(actual[:2, 0], .02)
    np.testing.assert_allclose(actual[2:, 1], -.02)
    assert not actual[:2, 1:].any()
    assert not actual[2:, 0].any()


def test_targeted_features_match_original_definitions():
    dates = pd.bdate_range("2024-01-01", periods=360)
    ranks = range(1, 7)
    time = np.arange(len(dates), dtype=float)
    wide, inputs, metadata, mappings = {}, {}, [], []
    anchors = pd.DataFrame(1., index=dates, columns=PRODUCT_ORDER)
    for position, root in enumerate(PRODUCT_ORDER):
        fields = {field: pd.DataFrame(index=dates, columns=ranks, dtype=float)
                  for field in ("price", "volume", "oi")}
        mapping = pd.DataFrame(index=dates, columns=ranks, dtype=object)
        for rank in ranks:
            price = 100 + position * 10 + rank * (1 + np.sin(time / 50)) + np.cos(time / 13)
            volume = 1000 + rank * 100 + time * 3 + 100 * np.sin(time / 19)
            oi = 5000 + rank * 200 + time * 2 + 300 * np.cos(time / 27)
            ticker = f"{root}{rank} Comdty"
            metadata.append({"root": root, "ticker": ticker, "rank": rank, "instrument": f"{root}|r{rank}"})
            mapping[rank] = f"{root}_contract_{rank}"
            mappings.extend({"date": date, "ticker": ticker, "contract": f"{root}_contract_{rank}"} for date in dates)
            for field, value in (("price", price), ("volume", volume), ("oi", oi)):
                fields[field][rank] = value
            for field, value in (("PX_SETTLE", price), ("PX_LAST", price), ("PX_VOLUME", volume), ("OPEN_INT", oi)):
                wide[(ticker, field)] = value
        inputs[root] = {**fields, "mapping": mapping}
    wide = pd.DataFrame(wide, index=dates)
    instruments = [f"{root}|r{rank}" for root in PRODUCT_ORDER for rank in ranks]
    columns = np.tile(np.arange(5) * 6, (len(dates), 1))
    panel = SimpleNamespace(index=dates, products=PRODUCT_ORDER, instruments=instruments,
                            execution_columns=columns, metadata=pd.DataFrame(metadata))
    original, _ = build_expanded_carry_signals(wide, pd.DataFrame(mappings), panel)
    wanted = ["carry2__slope_oi_d3_h5", "carry2__curve_breadth_slope_d6",
              "carry2__aggregate_volume_d6_h20", "carry2__carry_basis_agreement_o3_h63",
              "carry2__basis_momentum_o2_h5"]
    rebuilt = requested_curve_features(inputs, anchors, wanted)
    for name in wanted:
        pd.testing.assert_frame_equal(rebuilt[name], original[name])
