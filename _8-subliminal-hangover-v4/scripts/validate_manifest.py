#!/usr/bin/env python3
"""Basic manifest sanity checks for V4."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MANIFEST_CSV = PROJECT_DIR / "manifest.csv"


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    if not MANIFEST_CSV.exists():
        _die(f"missing {MANIFEST_CSV}")

    with MANIFEST_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _die("manifest.csv empty")

    required = {
        "id",
        "model",
        "condition",
        "target_id",
        "prime_id",
        "repetition",
        "seed",
        "text",
        "output_relpath",
    }
    missing_cols = required - set(rows[0].keys())
    if missing_cols:
        _die(f"missing columns: {sorted(missing_cols)}")

    ids = [r["id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        _die(f"duplicate id(s): {dup[:5]}")

    # Paired design check: for each (model,target_id,rep) there should be exactly one noun + one number.
    by_key = defaultdict(set)
    for r in rows:
        key = (r["model"], r["target_id"], r["repetition"])
        by_key[key].add(r["condition"])
    bad = [k for k, conds in by_key.items() if conds != {"noun", "number"}]
    if bad:
        _die(f"pairing broken for {len(bad)} keys (example={bad[0]})")

    placeholder = [r["id"] for r in rows if "REPLACE_ME" in r["text"]]
    if placeholder:
        _die("placeholder text present in manifest")

    print(f"ok: {len(rows)} rows; {len(by_key)} paired keys")


if __name__ == "__main__":
    main()

