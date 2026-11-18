"""Refresh presentation from verified saved results, without refitting any model."""
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from report_five_currents_consistency import render_report
from run_five_currents_lightweight_combination import sha256


def rebuild(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    presentation = {"Five_Currents_Consistency_Objectives.html", "consistency_admission_equity.png",
                    "consistency_pruned_equity.png", "consistency_portfolio_only_equity.png",
                    "consistency_windfall_ablation.png", "consistency_by_commodity.png"}
    for filename, digest in manifest["outputs_sha256"].items():
        assert sha256(output / filename) == digest, f"Changed output before refresh: {filename}"
    for filename, digest in manifest["source_sha256"].items():
        if Path(filename).name not in ("report_five_currents_consistency.py", "verify_five_currents_consistency_study.py"):
            assert sha256(filename) == digest, f"Model source changed: {filename}"
    metrics = pd.read_csv(output / "performance_summary.csv")
    annual = pd.read_csv(output / "yearly_performance.csv")
    paths = pd.read_parquet(output / "common_daily_paths.parquet")
    products = pd.read_parquet(output / "product_daily_contributions.parquet")
    audit = pd.read_parquet(output / "selection_audit.parquet")
    fits = pd.read_csv(output / "refit_summary.csv", parse_dates=["forecast_start", "forecast_end"])
    mature = fits.loc[fits.training_sessions.eq(504), "forecast_start"].min()
    render_report(output, metrics, annual, paths, products, audit, fits, mature)
    for filename, digest in manifest["outputs_sha256"].items():
        if filename not in presentation:
            assert sha256(output / filename) == digest, f"Model output changed during report refresh: {filename}"
    for filename in ("report_five_currents_consistency.py", "verify_five_currents_consistency_study.py", __file__):
        path = Path(filename).resolve()
        manifest["source_sha256"][str(path)] = sha256(path)
    manifest["outputs_sha256"] = {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "manifest.json"}
    manifest["report_only_refresh"] = {"utc": datetime.now(timezone.utc).isoformat(),
                                       "model_sources_unchanged": True, "all_nonpresentation_outputs_unchanged": True,
                                       "no_refitting": True}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Report refreshed; all model inputs, decisions, positions and numerical results unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("five_currents_consistency_objectives_20261012_final"))
    rebuild(parser.parse_args().output_dir.resolve())
