"""Rebuild raw daily features and causal fits from local cached input frames only."""
import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from five_currents_daily.engine import build_daily_replay
from five_currents_daily.dashboard_core import dollar_quote_replay, _chart_history


def run(output):
    output.mkdir(exist_ok=False)
    cache = Path("five_currents_dashboard/private_replay_cache.pkl")
    snapshot = Path("five_currents_dashboard/snapshots/000032_2026-10-09")
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    with cache.open("rb") as handle:
        original = pickle.load(handle)
    print("[offline rebuild] Replacing database loader with the cached CombinedDailyData frame", flush=True)
    with patch("five_currents_daily.engine.load_frozen_and_database_data", return_value=original.combined):
        rebuilt = build_daily_replay(None, end=original.last_signal_date)
    pd.testing.assert_frame_equal(rebuilt.weights, original.weights, check_exact=False, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rebuilt.concept_weights, original.concept_weights, equal_nan=True, atol=1e-10)
    np.testing.assert_allclose(rebuilt.execution.target_return, original.execution.target_return, equal_nan=True, atol=1e-10)
    print("[offline rebuild] Raw technical/curve concepts, execution returns and causal fits agree", flush=True)
    operational = dollar_quote_replay(rebuilt, curve_activation=Path("five_currents_dashboard/curve_policy_activation"))
    pd.testing.assert_frame_equal(operational.weights, pd.read_parquet(snapshot / "weights.parquet"),
                                  check_exact=False, atol=1e-10, rtol=1e-10)
    actual = _chart_history(operational)
    expected = json.loads((snapshot / "admin.json").read_text())["history"]
    np.testing.assert_allclose(np.array(actual["returns"], dtype=float), np.array(expected["returns"], dtype=float),
                               equal_nan=True, atol=1e-10)
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "database_access": False,
              "market_api_calls": False, "live_state_changes": False, "raw_concept_count": len(rebuilt.concept_names),
              "raw_concepts_rebuilt_from_local_daily_inputs": True, "original_causal_refits_reproduced": True,
              "raw_weights_match_cache": True, "operational_weights_match_snapshot": True,
              "operational_daily_pnl_matches_snapshot": True,
              "research_verification": asdict(rebuilt.verification), "cache_sha256": cache_hash,
              "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "raw_concept_maximum_error": float(np.nanmax(np.abs(rebuilt.concept_weights-original.concept_weights)))}
    (output / "full_daily_rebuild_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args().output_dir)
