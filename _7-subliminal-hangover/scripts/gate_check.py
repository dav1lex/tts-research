#!/usr/bin/env python3
"""Gate check for subliminal hangover benchmark V3.

Validates features.csv for new V3 columns:
- No rows with f0_mean == 0 (silence / no voicing)
- No clipped/distorted audio
- f0_cv must be > 0
- speaking_rate must be plausible (1-15 syllables/sec)
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
MIN_VOICED_F0 = 50.0
MAX_SANE_F0 = 500.0
MIN_SPEAKING_RATE = 1.0
MAX_SPEAKING_RATE = 15.0


def main():
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found.")
        return 1

    with open(FEATURES_CSV) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    failed = []
    passed = []
    n_counts = {}

    for row in rows:
        f0_mean = float(row["f0_mean"])
        f0_cv = float(row["f0_cv"])
        speaking_rate = float(row["speaking_rate"])
        model = row["model"]
        cond = row["condition"]

        issues = []

        if f0_mean < MIN_VOICED_F0:
            issues.append(f"f0_mean={f0_mean} < {MIN_VOICED_F0} (silent/unvoiced)")
        if f0_mean > MAX_SANE_F0:
            issues.append(f"f0_mean={f0_mean} > {MAX_SANE_F0} (suspicious)")
        if f0_cv <= 0 and f0_mean > 0:
            issues.append("f0_cv <= 0 with f0_mean > 0 (flat pitch)")
        if speaking_rate < MIN_SPEAKING_RATE:
            issues.append(f"speaking_rate={speaking_rate} < {MIN_SPEAKING_RATE} (too slow)")
        if speaking_rate > MAX_SPEAKING_RATE:
            issues.append(f"speaking_rate={speaking_rate} > {MAX_SPEAKING_RATE} (too fast)")
        if float(row["target_duration_s"]) < 0.3:
            issues.append(f"target too short ({row['target_duration_s']}s)")

        if issues:
            failed.append({"filename": row["filename"], "issues": issues})
            print(f"FAIL {row['filename']}: {'; '.join(issues)}")
        else:
            passed.append(row)
            key = f"{model}_{cond}"
            n_counts[key] = n_counts.get(key, 0) + 1

    # Summary per model/condition
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
            "min_speaking_rate": MIN_SPEAKING_RATE,
            "max_speaking_rate": MAX_SPEAKING_RATE,
            "min_target_duration_s": 0.3,
        }
    }

    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_JSON, "w") as f:
        json.dump(gate_result, f, indent=2)

    print(f"\nGate: {len(passed)}/{total} rows viable")
    print(f"Gate result: {'✅ PASS' if gate_result['gate_passed'] else '⚠️  PARTIAL FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
