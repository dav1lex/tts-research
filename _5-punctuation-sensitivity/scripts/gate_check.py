#!/usr/bin/env python3
"""Gate check: period pauses >= 150ms for >= 80% of items."""
import csv, json, sys
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/home/davilex/tts-research/_5-punctuation-sensitivity")
FEATURES = PROJECT / "results" / "features" / "pause_features.csv"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PERIOD_MIN_MS = 150
PASS_RATE = 0.80


def main():
    rows = list(csv.DictReader(open(FEATURES)))
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
              f"({pass_rate:.0%}) → {'PASS' if passed else 'FAIL'}")

        gate_results.append({
            "model": model,
            "period_items": num_periods,
            "periods_passing": periods_passing,
            "pass_rate": round(pass_rate, 3),
            "gate_passed": passed,
        })

        # Also show per-item details
        for r in period_items:
            pause = float(r["best_pause_ms"]) if r["best_pause_ms"] else 0
            status = "OK" if pause >= PERIOD_MIN_MS else "FAIL"
            print(f"  {r['id']}: {r['subcategory']} — {pause}ms {status}")

    # Write gate report
    with open(RESULTS_DIR / "gate_check.json", "w") as f:
        json.dump(gate_results, f, indent=2)

    all_pass = all(g["gate_passed"] for g in gate_results)
    print(f"\nOverall gate: {'PASS' if all_pass else 'FAIL'}")
    print(f"Gate allows analysis to proceed: {'YES' if all_pass else 'NO'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
