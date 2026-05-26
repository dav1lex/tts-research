#!/usr/bin/env python3
"""Gate check: period pauses >= configured ms for >= configured fraction of items.

This is a minimum viability check. All models passing means the audio is
intelligible and produces detectable silences at sentence boundaries. It does
NOT validate prosodic appropriateness or naturalness.
"""
import json
import sys
from collections import defaultdict

from common import (
    CONFIG,
    FEATURES_CSV,
    GATE_JSON,
    load_csv,
)

PERIOD_MIN_MS = CONFIG["gate"]["period_min_ms"]
PASS_RATE = CONFIG["gate"]["pass_rate"]


def main():
    rows = list(load_csv(FEATURES_CSV))
    gate_results = []

    # Group by model
    models = defaultdict(list)
    for r in rows:
        models[r["model"]].append(r)

    for model, mrows in models.items():
        # Sentence-end + trailing period items (punct_type=period)
        period_items = [r for r in mrows if r["punct_type"] == "period"]
        num_periods = len(period_items)

        # Check best_pause_ms >= PERIOD_MIN_MS
        periods_passing = sum(
            1 for r in period_items
            if r["best_pause_ms"] and float(r["best_pause_ms"]) >= PERIOD_MIN_MS
        )
        pass_rate = periods_passing / num_periods if num_periods else 0
        passed = pass_rate >= PASS_RATE

        print(f"{model}: {periods_passing}/{num_periods} periods >= {PERIOD_MIN_MS}ms "
              f"({pass_rate:.0%}) -> {'PASS' if passed else 'FAIL'}")

        gate_results.append({
            "model": model,
            "period_items": num_periods,
            "periods_passing": periods_passing,
            "pass_rate": round(pass_rate, 3),
            "gate_passed": passed,
            "gate_description": (
                f"Period-ending pauses >= {PERIOD_MIN_MS}ms for >= {PASS_RATE:.0%} of items. "
                "This confirms audible speech at boundaries, not prosodic quality."
            ),
        })

        # Per-item details
        for r in period_items:
            pause = float(r["best_pause_ms"]) if r["best_pause_ms"] else 0
            status = "OK" if pause >= PERIOD_MIN_MS else "FAIL"
            print(f"  {r['id']}: {r['subcategory']} - {int(float(r['best_pause_ms']))}ms {status}")

    # Write gate report
    with open(GATE_JSON, "w") as f:
        json.dump(gate_results, f, indent=2)

    all_pass = all(g["gate_passed"] for g in gate_results)
    print(f"\nOverall gate: {'PASS' if all_pass else 'FAIL'}")
    print(f"Gate allows analysis to proceed: {'YES' if all_pass else 'NO'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())