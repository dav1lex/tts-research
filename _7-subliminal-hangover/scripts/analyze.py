#!/usr/bin/env python3
"""Statistical analysis for subliminal hangover benchmark V3.

Primary metric: f0_cv (coefficient of variation of pitch).
Adds speaking_rate as control metric — if f0_cv drops after numbers
but speaking_rate is flat, the hangover effect is clean.

Wilcoxon signed-rank tests on f0_cv for:
  - Subliminal vs Nouns (primary — isolates robotic hangover)
  - Subliminal vs Control (original comparison)
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

CONDITION_LABELS = {
    "control": "Nature Sentence (short)",
    "noun": "Nouns (length-matched)",
    "subliminal": "Numbers (robotic)",
}

MODEL_LABELS = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro",
}

# Two primary comparisons
COMPARISONS = [
    ("nouns_vs_subliminal", "noun", "subliminal",
     "Nouns (length-matched) vs Numbers — primary: noun f0_cv > number f0_cv?"),
    ("subliminal_vs_control", "subliminal", "control",
     "Numbers vs Nature Sentence (original comparison)"),
]


def load_features() -> list[dict]:
    with open(FEATURES_CSV) as f:
        return list(csv.DictReader(f))


def group_by_model_condition(rows: list[dict]):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
    return groups


def compute_stats(values: list[float]):
    arr = np.array(values)
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        "se": round(float(np.std(arr, ddof=1) / np.sqrt(len(arr))), 4) if len(arr) > 1 else 0.0,
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def run_paired_test(a_rows, b_rows):
    """Wilcoxon signed-rank, alternative='greater': a > b (higher f0_cv in first condition)."""
    if len(a_rows) != len(b_rows) or len(a_rows) < 2:
        return None

    a_vals = np.array([float(r["f0_cv"]) for r in a_rows])
    b_vals = np.array([float(r["f0_cv"]) for r in b_rows])
    diff = a_vals - b_vals
    diff_mean = float(np.mean(diff))
    diff_pct = float((diff_mean / np.mean(a_vals)) * 100) if np.mean(a_vals) > 0 else 0.0

    try:
        w_stat, w_p = wilcoxon(a_vals, b_vals, alternative="greater")
        sig = "significant" if w_p < 0.05 else "not significant"
    except (ValueError, RuntimeError) as e:
        w_stat, w_p = None, None
        sig = f"test failed: {e}"

    return {
        "condition_a_mean": round(float(np.mean(a_vals)), 4),
        "condition_b_mean": round(float(np.mean(b_vals)), 4),
        "mean_difference": round(diff_mean, 4),
        "mean_difference_pct": round(diff_pct, 2),
        "condition_a_values": [round(float(v), 4) for v in a_vals],
        "condition_b_values": [round(float(v), 4) for v in b_vals],
        "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_value": round(float(w_p), 6) if w_p is not None else None,
        "significance": sig,
        "n_pairs": int(len(a_vals)),
        "test": "wilcoxon signed-rank (greater): f0_cv a > f0_cv b",
    }


def run_speaking_rate_test(a_rows, b_rows):
    """Check if speaking rates differ between conditions."""
    if len(a_rows) != len(b_rows) or len(a_rows) < 2:
        return None

    a_vals = np.array([float(r["speaking_rate"]) for r in a_rows])
    b_vals = np.array([float(r["speaking_rate"]) for r in b_rows])

    try:
        w_stat, w_p = wilcoxon(a_vals, b_vals, alternative="two-sided")
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)[:100]}

    return {
        "condition_a_mean": round(float(np.mean(a_vals)), 4),
        "condition_b_mean": round(float(np.mean(b_vals)), 4),
        "wilcoxon_p_value": round(float(w_p), 6),
        "significant_difference": bool(w_p < 0.05),
    }


def main():
    rows = load_features()
    if not rows:
        print("ERROR: no feature rows found")
        return 1

    valid_rows = [r for r in rows if float(r["f0_cv"]) > 0]
    if len(valid_rows) < len(rows):
        print(f"WARN: excluded {len(rows) - len(valid_rows)} rows with f0_cv <= 0")

    groups = group_by_model_condition(valid_rows)
    models = sorted(set(r["model"] for r in valid_rows))
    metrics = ["f0_cv", "f0_std", "f0_mean", "speaking_rate", "energy_std"]
    all_conditions = sorted(set(r["condition"] for r in valid_rows))

    # ── Descriptive stats ───────────────────────────────────────────
    descriptive = {}
    for model in models:
        descriptive[model] = {}
        for metric in metrics:
            descriptive[model][metric] = {}
            for cond in all_conditions:
                cond_rows = groups.get((model, cond), [])
                vals = [float(r[metric]) for r in cond_rows]
                descriptive[model][metric][cond] = compute_stats(vals)

    # ── Paired tests on f0_cv + speaking_rate checks ────────────────
    paired_results = {}

    for comp_key, cond_a, cond_b, desc in COMPARISONS:
        print(f"\n{'='*65}")
        print(f"  {desc}")
        print(f"{'='*65}")
        paired_results[comp_key] = {}

        for model in models:
            a_rows = sorted(groups.get((model, cond_a), []), key=lambda r: int(r["run"]))
            b_rows = sorted(groups.get((model, cond_b), []), key=lambda r: int(r["run"]))

            # Main test: f0_cv
            result = run_paired_test(a_rows, b_rows)
            if result is None:
                print(f"  {model}: insufficient pairs")
                continue

            # Speaking rate sanity check
            sr_result = run_speaking_rate_test(a_rows, b_rows)

            result["speaking_rate_check"] = sr_result
            paired_results[comp_key][model] = result

            a_label = CONDITION_LABELS.get(cond_a, cond_a)
            b_label = CONDITION_LABELS.get(cond_b, cond_b)
            sig = result["significance"]
            sr_note = ""
            if sr_result and not sr_result.get("error"):
                if sr_result.get("significant_difference"):
                    sr_note = f" ⚠️ speaking_rate DIFFERS (p={sr_result['wilcoxon_p_value']:.4f})"
                else:
                    sr_note = f" ✓ speaking_rate stable (p={sr_result['wilcoxon_p_value']:.4f})"

            print(f"\n  --- {model} ---")
            print(f"  {a_label} f0_cv: {result['condition_a_mean']:.4f}")
            print(f"  {b_label} f0_cv: {result['condition_b_mean']:.4f}")
            print(f"  Diff: {result['mean_difference']:+.4f} ({result['mean_difference_pct']:+.1f}%)")
            print(f"  Wilcoxon: W={result['wilcoxon_statistic']}, p={result['wilcoxon_p_value']:.4f} ({sig}){sr_note}")

    stats_out = {
        "descriptive": descriptive,
        "paired_tests": paired_results,
        "condition_labels": CONDITION_LABELS,
        "model_labels": MODEL_LABELS,
        "analysis_params": {
            "primary_metric": "f0_cv (coefficient of variation = f0_std / f0_mean)",
            "test": "wilcoxon signed-rank (one-sided, alternative='greater': condition_a f0_cv > condition_b f0_cv)",
            "primary_comparison": "nouns_vs_subliminal — tests noun f0_cv > number f0_cv (isolates robotic hangover)",
            "metrics_tracked": metrics,
            "speaking_rate_note": "If speaking_rate is flat across conditions but f0_cv drops, the effect is clean",
        }
    }

    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_JSON, "w") as f:
        json.dump(stats_out, f, indent=2)

    print(f"\nStats saved to {STATS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
