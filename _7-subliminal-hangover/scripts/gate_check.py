#!/usr/bin/env python3
"""Gate check for subliminal hangover benchmark.

Validates features.csv:
- No rows with f0_mean == 0 (silence / no voicing)
- No clipped/distorted audio (flag extreme f0 values)
- Prints viable n-counts per model/condition.
"""
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

FEATURES_CSV = PROJECT / "features" / "features.csv"
GATE_JSON = PROJECT / "results" / "gate_check.json"

# Thresholds
MIN_VOICED_F0 = 50.0    # f0_mean must be >= this to count as viable
MAX_SANE_F0 = 500.0     # sanity cap for f0_mean


def main():
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run extract_features.py first.")
        return 1

    with open(FEATURES_CSV) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    failed = []
    passed = []
    n_counts = {}

    for row in rows:
        f0_mean = float(row["f0_mean"])
        f0_std = float(row["f0_std"])
        model = row["model"]
        cond = row["condition"]

        issues = []

        if f0_mean < MIN_VOICED_F0:
            issues.append(f"f0_mean={f0_mean} < {MIN_VOICED_F0} (silent/unvoiced)")
        if f0_mean > MAX_SANE_F0:
            issues.append(f"f0_mean={f0_mean} > {MAX_SANE_F0} (suspicious)")
        if f0_std == 0 and f0_mean > 0:
            issues.append("f0_std=0 with f0_mean>0 (flat pitch, suspicious)")
        if float(row["target_duration_s"]) < 0.5:
            issues.append(f"target too short ({row['target_duration_s']}s)")

        if issues:
            failed.append({"filename": row["filename"], "issues": issues})
            print(f"FAIL {row['filename']}: {'; '.join(issues)}")
        else:
            passed.append(row)
            key = f"{model}_{cond}"
            n_counts[key] = n_counts.get(key, 0) + 1

    # Aggregate per model/condition
    summary = {}
    for model in sorted(set(r["model"] for r in rows)):
        for cond in sorted(set(r["condition"] for r in rows)):
            key = f"{model}_{cond}"
            n = n_counts.get(key, 0)
            total_key = sum(1 for r in rows if r["model"] == model and r["condition"] == cond)
            summary[key] = {"viable": n, "total": total_key, "passed": n == total_key}
            status = "PASS" if n == total_key else f"PARTIAL ({n}/{total_key})"
            print(f"  {key}: {status}")

    gate_result = {
        "total_rows": total,
        "passed_rows": len(passed),
        "failed_rows": len(failed),
        "fail_details": failed,
        "n_counts": n_counts,
        "summary": summary,
        "gate_passed": len(failed) == 0,
        "thresholds": {
            "min_voiced_f0": MIN_VOICED_F0,
            "max_sane_f0": MAX_SANE_F0,
            "min_target_duration_s": 0.5,
        }
    }

    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_JSON, "w") as f:
        json.dump(gate_result, f, indent=2)

    print(f"\nGate: {len(passed)}/{total} rows viable")
    print(f"Gate result: {'✅ PASS' if gate_result['gate_passed'] else '⚠️  PARTIAL FAIL'}")
    print(f"Gate JSON saved to {GATE_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
