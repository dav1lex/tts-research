#!/usr/bin/env python3
"""Gate the benchmark on detectable breathy-vs-neutral reference separation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import LOWER_IS_BREATHIER, METRICS, PRIMARY_METRIC, read_csv, safe_float

DEFAULT_MIN_PRIMARY_D = 0.8
DEFAULT_MIN_SUPPORTING_D = 0.3


def cohens_d(breathy: list[float], neutral: list[float]) -> float:
    if len(breathy) < 2 or len(neutral) < 2:
        return 0.0
    breathy_var = np.var(breathy, ddof=1)
    neutral_var = np.var(neutral, ddof=1)
    pooled = np.sqrt(
        ((len(breathy) - 1) * breathy_var + (len(neutral) - 1) * neutral_var)
        / (len(breathy) + len(neutral) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((np.mean(breathy) - np.mean(neutral)) / pooled)


def paired_contrasts(rows: list[dict[str, str]], metric: str) -> list[float]:
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["condition"]] = safe_float(row[metric])

    contrasts = []
    for values in by_pair.values():
        if "breathy" not in values or "neutral" not in values:
            continue
        contrasts.append(values["breathy"] - values["neutral"])
    return contrasts


def evaluate_gate(rows: list[dict[str, str]], min_primary_d: float, min_supporting_d: float) -> tuple[bool, list[dict]]:
    references = [row for row in rows if row["type"] == "reference"]
    if not references:
        raise ValueError("features.csv contains no reference rows")

    breathy = [row for row in references if row["condition"] == "breathy"]
    neutral = [row for row in references if row["condition"] == "neutral"]
    if len(breathy) < 2 or len(neutral) < 2:
        raise ValueError("gate needs at least 2 breathy and 2 neutral reference rows")

    results = []
    for metric in METRICS:
        breathy_values = [safe_float(row[metric]) for row in breathy]
        neutral_values = [safe_float(row[metric]) for row in neutral]
        d_value = cohens_d(breathy_values, neutral_values)
        direction_ok = (np.mean(breathy_values) < np.mean(neutral_values)) == LOWER_IS_BREATHIER[metric]
        min_d = min_primary_d if metric == PRIMARY_METRIC else min_supporting_d
        effect_ok = abs(d_value) >= min_d
        contrasts = paired_contrasts(references, metric)
        paired_direction_ok = all(contrast < 0 for contrast in contrasts) if LOWER_IS_BREATHIER[metric] else all(
            contrast > 0 for contrast in contrasts
        )
        passed = bool(direction_ok and effect_ok and paired_direction_ok)
        results.append({
            "metric": metric,
            "breathy_mean": float(np.mean(breathy_values)),
            "neutral_mean": float(np.mean(neutral_values)),
            "cohens_d": d_value,
            "min_abs_d": min_d,
            "direction_ok": bool(direction_ok),
            "paired_direction_ok": bool(paired_direction_ok),
            "paired_contrasts": contrasts,
            "passed": passed,
            "primary": metric == PRIMARY_METRIC,
        })

    primary_passed = next(result for result in results if result["primary"])["passed"]
    supporting_passed = sum(result["passed"] for result in results if not result["primary"]) >= 1
    return bool(primary_passed and supporting_passed), results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reference breathiness separation before model analysis")
    parser.add_argument("--features-dir", default=Path("features"), type=Path)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--min-primary-d", default=DEFAULT_MIN_PRIMARY_D, type=float)
    parser.add_argument("--min-supporting-d", default=DEFAULT_MIN_SUPPORTING_D, type=float)
    args = parser.parse_args()

    features_path = args.features_dir / "features.csv"
    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run extract_features.py first.", file=sys.stderr)
        return 1

    try:
        passed, results = evaluate_gate(read_csv(features_path), args.min_primary_d, args.min_supporting_d)
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
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status} {result['metric']}: "
            f"breathy={result['breathy_mean']:.4f} neutral={result['neutral_mean']:.4f} "
            f"d={result['cohens_d']:.4f} min|d|={result['min_abs_d']:.2f} "
            f"direction={result['direction_ok']} paired={result['paired_direction_ok']}"
        )
    print(f"Wrote {gate_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
