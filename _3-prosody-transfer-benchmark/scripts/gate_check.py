#!/usr/bin/env python3
"""Gate: reference F0 separation between breathy and modal groups.

Gate passes if Cohen's d >= 0.5 on F0 mean OR F0 range.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import METRICS, read_csv, safe_float

GATE_METRICS = ("f0_mean", "f0_range")  # either one must pass
MIN_COHENS_D = 0.5


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    pooled = np.sqrt(
        ((len(group_a) - 1) * np.var(group_a, ddof=1)
         + (len(group_b) - 1) * np.var(group_b, ddof=1))
        / (len(group_a) + len(group_b) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def evaluate_gate(rows: list[dict]) -> tuple[bool, list[dict]]:
    references = [row for row in rows if row["type"] == "reference"]
    if not references:
        raise ValueError("features.csv contains no reference rows")

    breathy = [row for row in references if row["condition"] == "breathy"]
    modal = [row for row in references if row["condition"] == "modal"]
    if len(breathy) < 2 or len(modal) < 2:
        raise ValueError("gate needs at least 2 breathy and 2 modal reference rows")

    results = []
    for metric in GATE_METRICS:
        breathy_vals = [safe_float(row[metric]) for row in breathy]
        modal_vals = [safe_float(row[metric]) for row in modal]
        d = cohens_d(breathy_vals, modal_vals)
        abs_d = abs(d)
        direction = "breathy_higher" if d > 0 else "modal_higher"
        passed = abs_d >= MIN_COHENS_D
        results.append({
            "metric": metric,
            "breathy_values": breathy_vals,
            "modal_values": modal_vals,
            "breathy_mean": float(np.mean(breathy_vals)),
            "modal_mean": float(np.mean(modal_vals)),
            "cohens_d": d,
            "abs_cohens_d": abs_d,
            "min_abs_d": MIN_COHENS_D,
            "direction": direction,
            "passed": passed,
        })

    primary_passed = any(r["passed"] for r in results)
    return bool(primary_passed), results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate: reference F0 separation between breathy and modal"
    )
    parser.add_argument("--features-dir", default=Path("features"), type=Path)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    features_path = args.features_dir / "features.csv"
    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run extract_features.py first.", file=sys.stderr)
        return 1

    try:
        passed, results = evaluate_gate(read_csv(features_path))
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    gate_path = args.results_dir / "gate_check.json"
    gate_path.write_text(json.dumps({"passed": passed, "metrics": results}, indent=2))
    marker_path = args.results_dir / ".gate_passed"
    if passed:
        marker_path.write_text("passed\n")
    elif marker_path.exists():
        marker_path.unlink()

    print(f"Gate {'PASSED' if passed else 'FAILED'}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{status} {r['metric']}: "
            f"breathy_mean={r['breathy_mean']:.2f} modal_mean={r['modal_mean']:.2f} "
            f"d={r['cohens_d']:.4f} |d|={r['abs_cohens_d']:.4f} "
            f"({r['direction']})"
        )
    print(f"Wrote {gate_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
