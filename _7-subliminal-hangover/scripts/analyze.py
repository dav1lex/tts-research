#!/usr/bin/env python3
"""Statistical analysis for subliminal hangover benchmark.

Loads features.csv, groups by model/condition, computes mean and SE for f0_std,
and runs a paired test (Wilcoxon signed-rank) comparing control vs subliminal
per model.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

FEATURES_CSV = PROJECT / "features" / "features.csv"
STATS_JSON = PROJECT / "results" / "stats.json"


def load_features() -> list[dict]:
    with open(FEATURES_CSV) as f:
        return list(csv.DictReader(f))


def group_by_model_condition(rows: list[dict]):
    """Group rows by (model, condition) and by (model) for paired testing."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["model"], r["condition"])
        groups[key].append(r)
    return groups


def compute_stats(values: list[float]):
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        "se": round(float(np.std(arr, ddof=1) / np.sqrt(len(arr))), 4) if len(arr) > 1 else 0.0,
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def main():
    rows = load_features()
    if not rows:
        print("ERROR: no feature rows found")
        return 1

    valid_rows = [r for r in rows if float(r["f0_std"]) > 0]
    if len(valid_rows) < len(rows):
        print(f"WARN: {len(rows) - len(valid_rows)} rows with f0_std=0 excluded")

    groups = group_by_model_condition(valid_rows)

    # Per-model paired data: collect (control_f0_std, subliminal_f0_std) pairs
    models = sorted(set(r["model"] for r in valid_rows))
    metrics = ["f0_std", "f0_mean", "energy_std"]

    results = {}
    paired_results = {}

    for model in models:
        model_data = {}
        for metric in metrics:
            model_data[metric] = {}
            for cond in ["control", "subliminal"]:
                cond_rows = groups.get((model, cond), [])
                vals = [float(r[metric]) for r in cond_rows]
                model_data[metric][cond] = compute_stats(vals)

        results[model] = model_data

        # Paired test: align by run number
        control_rows = sorted(
            groups.get((model, "control"), []),
            key=lambda r: int(r["run"])
        )
        subliminal_rows = sorted(
            groups.get((model, "subliminal"), []),
            key=lambda r: int(r["run"])
        )

        if len(control_rows) == len(subliminal_rows) and len(control_rows) >= 2:
            c_vals = np.array([float(r["f0_std"]) for r in control_rows])
            s_vals = np.array([float(r["f0_std"]) for r in subliminal_rows])

            # Paired difference
            diff = c_vals - s_vals
            diff_mean = float(np.mean(diff))
            diff_pct = float((diff_mean / np.mean(c_vals)) * 100) if np.mean(c_vals) > 0 else 0.0

            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = wilcoxon(c_vals, s_vals, alternative="greater")
                sig = "significant" if w_p < 0.05 else "not significant"
            except (ValueError, RuntimeError) as e:
                w_stat, w_p = None, None
                sig = f"test failed: {e}"

            paired_results[model] = {
                "metric": "f0_std",
                "control_mean": round(float(np.mean(c_vals)), 4),
                "subliminal_mean": round(float(np.mean(s_vals)), 4),
                "mean_difference": round(diff_mean, 4),
                "mean_difference_pct": round(diff_pct, 2),
                "control_values": [round(float(v), 4) for v in c_vals],
                "subliminal_values": [round(float(v), 4) for v in s_vals],
                "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
                "wilcoxon_p_value": round(float(w_p), 6) if w_p is not None else None,
                "significance": sig,
                "n_pairs": len(c_vals),
            }

            print(f"\n--- {model} ---")
            print(f"  Control f0_std:    {paired_results[model]['control_mean']:.2f} ± ...")
            print(f"  Subliminal f0_std: {paired_results[model]['subliminal_mean']:.2f} ± ...")
            print(f"  Difference:        {paired_results[model]['mean_difference']:+.2f} Hz ({paired_results[model]['mean_difference_pct']:+.1f}%)")
            print(f"  Wilcoxon:          W={w_stat}, p={w_p:.4f} ({sig})")
        else:
            print(f"\n--- {model} ---")
            print(f"  WARN: mismatched or insufficient pairs (control={len(control_rows)}, subliminal={len(subliminal_rows)})")

    stats_out = {
        "descriptive": results,
        "paired_tests": paired_results,
        "analysis_params": {
            "primary_metric": "f0_std",
            "test": "wilcoxon signed-rank (one-sided, alternative='greater': control > subliminal)",
            "metrics_tracked": metrics,
        }
    }

    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_JSON, "w") as f:
        json.dump(stats_out, f, indent=2)

    print(f"\nStats saved to {STATS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
