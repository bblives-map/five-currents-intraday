"""Offline replay of Five Currents books on the corrected front-route minute fills."""
from __future__ import annotations

import json
from pathlib import Path

import build_five_currents_two_books_report as books
import run_five_currents_frontmonth_alpha_audit as audit


FRONT = Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "five_currents_frontmonth_20261025"
PRICE = FRONT / "front_clock_prices_liquidity.npz"


def main():
    for required in (PRICE, FRONT / "manifest_liquidity.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    audit_manifest = audit.OUT / "manifest.json"
    if audit_manifest.exists():
        recorded = json.loads(audit_manifest.read_text(encoding="utf-8"))
        if recorded["front_price_sha256"] != audit.digest(PRICE):
            raise RuntimeError("Cached alpha audit belongs to a different price cube")
        print("[audit] Reusing the matching completed alpha grid", flush=True)
    else:
        audit.run()
    books.AUDIT = audit.OUT
    books.PRICE_SOURCE = PRICE
    books.OUT = FRONT / "two_books"
    books.DATA_ONLY = True
    books.run()

    # The rolling module imports these locations by value, so import it only
    # after directing the books module to the corrected artefacts.
    import run_five_currents_rolling_edge as rolling
    rolling.AUDIT = audit.OUT
    rolling.PRICE_SOURCE = PRICE
    rolling.OUT = FRONT / "rolling"
    rolling.run()


if __name__ == "__main__":
    main()
