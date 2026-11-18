"""Check the offline family-selection repeat independently of its HTML renderer."""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from five_currents_execution_timing import summarize_matrix
from prepare_five_currents_execution_timing import digest, save_json
from run_five_currents_family_timing import OUTPUT


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(manifest["source"])
    for name, expected in manifest["source_sha256"].items():
        assert digest(source / name) == expected, name
    paths = pd.read_parquet(output / "paths.parquet")
    metrics = pd.read_csv(output / "metrics.csv")
    stress = pd.read_csv(output / "stress.csv")
    ledger = pd.read_parquet(output / "refit_ledger.parquet")
    sessions = pd.DatetimeIndex(pd.read_parquet(source / "sessions.parquet").session)
    assert not paths.duplicated(["id", "session"]).any()
    assert (paths.exposure <= 1+1e-12).all()
    assert (paths.exposure >= 0).all()
    assert (ledger.train_end_exclusive < ledger.start).all()
    assert (ledger.last_training_session < ledger.forecast_start).all()
    assert (ledger.train_end_exclusive-ledger.train_start >= 126).all()
    assert (ledger.allocation >= 0).all()
    assert ((ledger.candidate >= 0) | ledger.allocation.eq(0)).all()
    for _, part in ledger.loc[ledger.allocation > 0].groupby(["grid", "timeline", "pool", "fit", "method", "start"]):
        assert part.allocation.sum() <= 1+1e-12
        if part.method.iloc[0] == "greedy_three":
            assert len(part) <= 3
            np.testing.assert_allclose(part.allocation, 1/len(part))
    metric_checks, stress_checks = 0, 0
    for sid, frame in paths.groupby("id"):
        frame = frame.sort_values("session")
        pd.testing.assert_index_equal(pd.DatetimeIndex(frame.session), sessions, check_names=False)
        np.testing.assert_allclose(frame.net, frame.gross-frame.cost, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(frame.stressed, frame.gross-2*frame.cost, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(frame[["CL", "NG", "GC", "SI", "HG"]].sum(axis=1, min_count=5),
                                   frame.net, atol=1e-12, equal_nan=True)
        assert (frame.loc[frame.calibrating, "net"] == 0).all()
        for period, mask in (("full_window", np.ones(len(frame), dtype=bool)),
                             ("post_calibration", ~frame.calibrating.to_numpy()),
                             ("common_late_window", np.arange(len(frame)) >= int(len(frame)*.5))):
            saved = metrics.loc[metrics.id.eq(sid) & metrics.period.eq(period)].iloc[0]
            stats = summarize_matrix(frame.net.to_numpy()[mask])
            for key in stats:
                np.testing.assert_allclose(saved[key], stats[key][0], atol=1e-12, rtol=1e-10, equal_nan=True)
            metric_checks += 1
        zero = stress.loc[stress.id.eq(sid) & stress.delay_minutes.eq(0) & stress.lag.eq(0)]
        if len(zero):
            saved = metrics.loc[metrics.id.eq(sid) & metrics.period.eq("post_calibration")].iloc[0]
            np.testing.assert_allclose(zero.iloc[0].sharpe, saved.sharpe, atol=1e-12, equal_nan=True)
            stress_checks += 1
    with np.load(output / "schedules.npz") as schedules:
        for sid in paths.id.unique():
            choices, alpha = schedules[sid+"__choices"], schedules[sid+"__allocations"]
            assert choices.shape == alpha.shape == (len(sessions), 5, 10)
            assert (alpha >= 0).all() and (alpha.sum(axis=2) <= 1+1e-12).all()
            assert np.all((choices >= 0) | (alpha == 0))
    scalar = pd.read_csv(output / "independent_scalar_audit.csv")
    assert len(scalar) > 0 and scalar.absolute_error.max() < 1e-12
    attribution = pd.read_csv(output / "attribution_common_dates.csv")
    for (sid, kind), part in attribution.groupby(["id", "kind"]):
        expected = metrics.loc[metrics.id.eq(sid) & metrics.period.eq("post_calibration")].iloc[0]
        np.testing.assert_allclose(part.observed_pnl_contribution.sum(), expected.observed_total_return, atol=1e-12)
        assert (part.days == expected.days).all()
    document = (output / "Five_Currents_Family_Walkforward_Report.html").read_text(encoding="utf-8")
    images = re.findall(r'data:image/png;base64,([^"\s]+)', document)
    for encoded in images:
        assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
    for link in re.findall(r'href="([^"]+)"', document):
        assert (output / link).is_file(), link
    assert len(images) >= 6
    result = {"passed": True, "portfolios": paths.id.nunique(), "sessions_per_path": len(sessions),
        "metric_reconciliations": metric_checks, "zero_delay_reconciliations": stress_checks,
        "source_hashes_unchanged": True, "embedded_pngs": len(images),
        "maximum_scalar_pnl_error": float(scalar.absolute_error.max()),
        "common_date_attribution_checks": attribution.groupby(["id", "kind"]).ngroups,
        "new_market_api_requests": False, "production_changes": False,
        "files_sha256": {name: digest(output / name) for name in (
            "manifest.json", "paths.parquet", "metrics.csv", "refit_ledger.parquet", "schedules.npz",
            "stress.csv", "Five_Currents_Family_Walkforward_Report.html")}}
    save_json(output / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    verify(parser.parse_args().output.resolve())
