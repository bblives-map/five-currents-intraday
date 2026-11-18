"""Retain the original frozen-day integration check without distributing its cache."""
import os
from pathlib import Path
import pytest


def pytest_collection_modifyitems(items):
    root=Path(os.environ.get("FIVE_CURRENTS_DATA_ROOT",str(Path(__file__).parent)))
    artifact=root/"five_currents_frontmonth_20261025/monday_20261026_signal_audit.json"
    for item in items:
        if (item.originalname or item.name)=="test_monday_blocks_reconcile_to_audited_frozen_legs" and not artifact.is_file():
            item.add_marker(pytest.mark.skip(reason="Private frozen-day signal audit not redistributed"))
