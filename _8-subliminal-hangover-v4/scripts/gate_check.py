#!/usr/bin/env python3
"""Gate check for V4 features.csv."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_CSV = PROJECT_DIR / "features" / "features.csv"
GATE_JSON = PROJECT_DIR / "results" / "gate_check.json"

MIN_VOICED_F0 = 50.0
MAX_SANE_F0 = 500.0
MIN_SPEAKING_RATE = 1.0
MAX_SPEAKING_RATE = 15.0
MIN_TARGET_DURATION_S = 0.3


def main() -> int:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found.", file=sys.stderr)
        return 2

    with FEATURES_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    failed = []
    passed = []
    n_counts: dict[str, int] = {}

    for row in rows:
        try:
            f0_mean = float(row["f0_mean"])
            f0_cv = float(row["f0_cv"])
            speaking_rate = float(row["speaking_rate"])
            dur = float(row["target_duration_s"])
        except Exception as e:
            failed.append({"id": row.get("id", ""), "issues": [f"parse_error: {e}"]})
            continue

        model = row.get("model", "")
        cond = row.get("condition", "")

        issues = []
        if f0_mean < MIN_VOICED_F0:
            issues.append(f"f0_mean={f0_mean} < {MIN_VOICED_F0}")
        if f0_mean > MAX_SANE_F0:
            issues.append(f"f0_mean={f0_mean} > {MAX_SANE_F0}")
        if f0_cv <= 0 and f0_mean > 0:
            issues.append("f0_cv <= 0 with f0_mean > 0")
        if speaking_rate < MIN_SPEAKING_RATE:
            issues.append(f"speaking_rate={speaking_rate} < {MIN_SPEAKING_RATE}")
        if speaking_rate > MAX_SPEAKING_RATE:
            issues.append(f"speaking_rate={speaking_rate} > {MAX_SPEAKING_RATE}")
        if dur < MIN_TARGET_DURATION_S:
            issues.append(f"target_duration_s={dur} < {MIN_TARGET_DURATION_S}")

        if issues:
            failed.append({"id": row.get("id", ""), "filename": row.get("filename", ""), "issues": issues})
        else:
            passed.append(row)
            key = f"{model}_{cond}"
            n_counts[key] = n_counts.get(key, 0) + 1

    summary: dict[str, dict] = {}
    models = sorted(set(r.get("model", "") for r in rows if r.get("model")))
    conds = sorted(set(r.get("condition", "") for r in rows if r.get("condition")))
    for m in models:
        for c in conds:
            key = f"{m}_{c}"
            total_key = sum(1 for r in rows if r.get("model") == m and r.get("condition") == c)
            viable = n_counts.get(key, 0)
            summary[key] = {"viable": viable, "total": total_key, "passed": viable == total_key}

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
            "min_target_duration_s": MIN_TARGET_DURATION_S,
        },
    }

    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")

    print(f"Gate: {len(passed)}/{total} rows viable")
    print(f"Gate result: {'PASS' if gate_result['gate_passed'] else 'PARTIAL FAIL'}")
    return 0 if gate_result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
