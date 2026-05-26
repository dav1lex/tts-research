#!/usr/bin/env python3
"""Analyze punctuation sensitivity from extracted features.

Adds bootstrap 95% CIs and Cohen's d effect sizes for key comparisons.
Keeps same output schema as before for backwards compatibility, with
added CI/effect-size fields.
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import (
    ANALYSIS_JSON,
    CONFIG,
    FEATURES_CSV,
    SCORES_JSON,
    load_csv,
    safe_float,
)

BOOTSTRAP_ITER = CONFIG["analysis"]["bootstrap_iterations"]


# ── Statistics helpers ───────────────────────────────────────────────────────


def bootstrap_ci(values: list[float], n_iter: int = BOOTSTRAP_ITER, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence interval for the mean of values."""
    if len(values) < 2:
        return {"mean": round(np.mean(values), 1) if values else None, "ci_low": None, "ci_high": None, "n": len(values)}
    means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_iter)
    ])
    alpha = (1 - ci) / 2
    low, high = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return {
        "mean": round(float(np.mean(values)), 1),
        "ci_low": round(float(low), 1),
        "ci_high": round(float(high), 1),
        "n": len(values),
    }


def cohens_d(group_a: list[float], group_b: list[float]) -> dict:
    """Cohen's d effect size between two independent groups."""
    if len(group_a) < 2 or len(group_b) < 2:
        return {"d": None, "interpretation": "insufficient samples"}
    n1, n2 = len(group_a), len(group_b)
    m1, m2 = np.mean(group_a), np.mean(group_b)
    v1, v2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return {"d": None, "interpretation": "zero variance"}
    d = (m1 - m2) / pooled
    # Standard interpretation
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"
    return {"d": round(float(d), 3), "interpretation": interp}


def grouped_values(data: dict, model: str, category: str, field: str, punct_filter: str = None) -> list[float]:
    """Extract numeric values from features grouped by model and category."""
    vals = []
    for r in data[model]:
        if r["category"] != category:
            continue
        if punct_filter and r["punct_type"] != punct_filter:
            continue
        v = safe_float(r[field])
        if v is not None:
            vals.append(v)
    return vals


# ── Analysis functions ───────────────────────────────────────────────────────


def sentence_end_analysis(data):
    """Analyze F0 slope differences between terminal punctuation types."""
    print("\n" + "=" * 60)
    print("1. SENTENCE-END PUNCTUATION SENSITIVITY (descriptive only)")
    print("=" * 60)
    print("Expected: period=flat/fall, exclamation=rise-fall, question=rise\n")

    results = {}
    for model, rows in data.items():
        print(f"--- {model} ---")
        model_scores = {}
        for punct in ["period", "exclamation", "question"]:
            slopes = grouped_values(data, model, "sentence_end", "terminal_f0_slope", punct)
            ci = bootstrap_ci(slopes)
            print(f"  {punct}: mean={ci['mean']} Hz/s "
                  f"(CI: [{ci['ci_low']}, {ci['ci_high']}], n={ci['n']})")
            model_scores[punct] = ci

        # Question vs period differentiation
        period_slopes = grouped_values(data, model, "sentence_end", "terminal_f0_slope", "period")
        question_slopes = grouped_values(data, model, "sentence_end", "terminal_f0_slope", "question")
        if period_slopes and question_slopes:
            diff = np.mean(question_slopes) - np.mean(period_slopes)
            d = cohens_d(question_slopes, period_slopes)
            se_diff = round(float(diff), 1)
            print(f"  question-vs-period diff: {se_diff} Hz/s (Cohen's d={d['d']}, {d['interpretation']})")
            model_scores["question_period_diff"] = se_diff
            model_scores["question_vs_period_effect_size"] = d

        results[model] = model_scores

    return results


def pause_hierarchy_analysis(data):
    """Analyze pause duration variation by internal punctuation type."""
    print("\n" + "=" * 60)
    print("2. PAUSE HIERARCHY SENSITIVITY (descriptive only)")
    print("=" * 60)
    print("Expected: comma < semicolon < em-dash < ellipsis\n")

    results = {}
    for model, rows in data.items():
        print(f"--- {model} ---")
        model_scores = {}
        ph_items = [r for r in rows if r["category"] == "pause_hierarchy"]
        by_punct = defaultdict(list)
        for r in ph_items:
            internal = json.loads(r["internal_pause_durations_ms"])
            by_punct[r["punct_type"]].extend(internal)

        means_list = []
        for punct in ["comma", "semicolon", "em_dash", "ellipsis"]:
            durations = by_punct.get(punct, [])
            mean_dur = np.mean(durations) if durations else 0
            median_dur = np.median(durations) if durations else 0
            means_list.append(mean_dur)
            ci = bootstrap_ci(durations) if durations else {"mean": 0, "ci_low": None, "ci_high": None, "n": 0}
            print(f"  {punct}: mean={mean_dur:.0f}ms, median={median_dur:.0f}ms "
                  f"(CI: [{ci['ci_low']}, {ci['ci_high']}], n={len(durations)})")
            model_scores[punct] = {
                "mean_ms": round(float(mean_dur), 1),
                "median_ms": round(float(median_dur), 1),
                "ci_low_ms": ci["ci_low"],
                "ci_high_ms": ci["ci_high"],
                "n": len(durations),
            }

        # Hierarchy score
        if means_list:
            hierarchy_correct = sum(1 for i in range(len(means_list)-1) if means_list[i] <= means_list[i+1])
            hierarchy_score = hierarchy_correct / 3
            model_scores["hierarchy_score"] = round(hierarchy_score, 2)
            print(f"  hierarchy score: {hierarchy_score:.2f} ({hierarchy_correct}/3 pairs monotonic)")

        results[model] = model_scores

    return results


def trailing_analysis(data):
    """Analyze ellipsis vs period trailing behavior."""
    print("\n" + "=" * 60)
    print("3. TRAILING PUNCTUATION SENSITIVITY (descriptive only)")
    print("=" * 60)
    print("Expected: ellipsis=trailing F0 + amplitude fade, period=sharp fall\n")

    results = {}
    for model, rows in data.items():
        print(f"--- {model} ---")
        model_scores = {}
        by_punct = defaultdict(list)
        for r in rows:
            if r["category"] != "trailing":
                continue
            slope = safe_float(r["terminal_f0_slope"])
            decay = safe_float(r["amplitude_decay_300ms"])
            pause = safe_float(r["best_pause_ms"])
            if slope is not None:
                by_punct[r["punct_type"]].append({
                    "f0_slope": slope,
                    "amp_decay": decay,
                    "pause_ms": pause,
                })

        for punct in ["ellipsis", "period"]:
            items = by_punct.get(punct, [])
            if items:
                slopes = [x["f0_slope"] for x in items]
                decays = [x["amp_decay"] for x in items if x["amp_decay"] is not None]
                pauses_arr = [x["pause_ms"] for x in items if x["pause_ms"] is not None]
                f0_ci = bootstrap_ci(slopes)
                print(f"  {punct}: F0 slope={f0_ci['mean']} Hz/s "
                      f"(CI: [{f0_ci['ci_low']}, {f0_ci['ci_high']}], n={f0_ci['n']})")
                model_scores[punct] = {
                    "f0_slope_mean": f0_ci["mean"],
                    "f0_slope_ci_low": f0_ci["ci_low"],
                    "f0_slope_ci_high": f0_ci["ci_high"],
                    "n_f0": f0_ci["n"],
                    "amp_decay_mean": round(float(np.mean(decays)), 10) if decays else None,
                    "pause_mean": round(float(np.mean(pauses_arr)), 1) if pauses_arr else None,
                }

        # Trailing distinction effect size
        ell = by_punct.get("ellipsis", [])
        per = by_punct.get("period", [])
        if ell and per:
            ell_f0 = np.mean([x["f0_slope"] for x in ell])
            per_f0 = np.mean([x["f0_slope"] for x in per])
            f0_diff = ell_f0 - per_f0
            ell_vals = [x["f0_slope"] for x in ell]
            per_vals = [x["f0_slope"] for x in per]
            d = cohens_d(ell_vals, per_vals)
            print(f"  ellipsis-vs-period F0 diff: {f0_diff:.1f} Hz/s (Cohen's d={d['d']}, {d['interpretation']})")
            model_scores["trailing_f0_diff"] = round(float(f0_diff), 1)
            model_scores["trailing_effect_size"] = d

        results[model] = model_scores

    return results


def capitalization_analysis(data):
    """Analyze ALL-CAPS vs normal emphasis. Single-item per condition — purely anecdotal."""
    print("\n" + "=" * 60)
    print("4. CAPITALIZATION SENSITIVITY")
    print("=" * 60)
    print("WARNING: 1 item per condition. These are anecdotes, not evidence.\n")

    results = {}
    for model, rows in data.items():
        c_items = [r for r in rows if r["category"] == "capitalization"]
        print(f"--- {model} ---")
        for r in sorted(c_items, key=lambda x: x["subcategory"]):
            rms = safe_float(r["rms_mean"])
            f0_range = safe_float(r["f0_range"])
            f0_mean = safe_float(r["f0_mean"])
            print(f"  {r['subcategory']:12s}: RMS={rms}, F0_mean={f0_mean}Hz, F0_range={f0_range}Hz")

        caps_vals = {r["subcategory"]: r for r in c_items}
        if "all_caps" in caps_vals and "title_case" in caps_vals:
            caps_rms = safe_float(caps_vals["all_caps"]["rms_mean"])
            title_rms = safe_float(caps_vals["title_case"]["rms_mean"])
            caps_f0range = safe_float(caps_vals["all_caps"]["f0_range"])
            title_f0range = safe_float(caps_vals["title_case"]["f0_range"])
            has_rms_boost = caps_rms and title_rms and caps_rms > title_rms
            has_f0_boost = caps_f0range and title_f0range and caps_f0range > title_f0range
            print(f"  RMS boost: {'YES' if has_rms_boost else 'NO'}, F0-range boost: {'YES' if has_f0_boost else 'NO'}")
            results[model] = {"rms_boost": has_rms_boost, "f0_boost": has_f0_boost}

    return results


def quotation_analysis(data):
    """Analyze quoted vs reported speech prosody shift."""
    print("\n" + "=" * 60)
    print("5. QUOTATION SENSITIVITY")
    print("=" * 60)
    print("Expected: quoted speech = distinct F0 range/mean shift\n")

    results = {}
    for model, rows in data.items():
        print(f"--- {model} ---")
        by_type = defaultdict(list)
        for r in rows:
            if r["category"] != "quotation":
                continue
            f0_mean = safe_float(r["f0_mean"])
            f0_range = safe_float(r["f0_range"])
            if f0_mean is not None:
                by_type[r["punct_type"]].append({"f0_mean": f0_mean, "f0_range": f0_range})

        for ptype in ["quoted", "reported"]:
            items = by_type.get(ptype, [])
            if items:
                f0_means = [x["f0_mean"] for x in items]
                f0_ranges = [x["f0_range"] for x in items]
                print(f"  {ptype}: F0_mean={np.mean(f0_means):.1f}Hz, F0_range={np.mean(f0_ranges):.1f}Hz")

        quoted = by_type.get("quoted", [])
        reported = by_type.get("reported", [])
        if quoted and reported:
            q_f0m = np.mean([x["f0_mean"] for x in quoted])
            r_f0m = np.mean([x["f0_mean"] for x in reported])
            q_f0r = np.mean([x["f0_range"] for x in quoted])
            r_f0r = np.mean([x["f0_range"] for x in reported])
            f0_mean_shift = abs(q_f0m - r_f0m)
            f0_range_shift = abs(q_f0r - r_f0r)
            threshold = CONFIG["quotation"]["range_shift_threshold_hz"]
            has_shift = f0_range_shift > threshold

            # Effect size on F0 range
            q_ranges = [x["f0_range"] for x in quoted]
            r_ranges = [x["f0_range"] for x in reported]
            d = cohens_d(q_ranges, r_ranges)

            print(f"  F0 mean shift: {f0_mean_shift:.1f}Hz, F0 range shift: {f0_range_shift:.1f}Hz")
            print(f"  Cohen's d (range): {d['d']}, {d['interpretation']}")
            print(f"  Quotation shift (>={threshold}Hz): {'YES' if has_shift else 'NO'}")
            results[model] = {
                "f0_mean_shift": round(float(f0_mean_shift), 1),
                "f0_range_shift": round(float(f0_range_shift), 1),
                "shift_threshold_hz": threshold,
                "shift_detected": bool(has_shift),
                "range_effect_size": d,
            }

    return results


def overall_score(data):
    """Overall sensitivity scores."""
    print("\n" + "=" * 60)
    print("6. OVERALL SENSITIVITY SCORES (descriptive statistics)")
    print("=" * 60)
    print("WARNING: n is small (28 items). These are not robust rankings.\n")

    scores = {}
    for model in ["chatterbox", "xtts", "kokoro"]:
        if model not in data:
            continue

        # Question vs period diff
        q_slopes = grouped_values(data, model, "sentence_end", "terminal_f0_slope", "question")
        p_slopes = grouped_values(data, model, "sentence_end", "terminal_f0_slope", "period")
        qp_diff = np.mean(q_slopes) - np.mean(p_slopes) if q_slopes and p_slopes else 0

        # Pause hierarchy diff
        ph_rows = [r for r in data[model] if r["category"] == "pause_hierarchy"]
        by_punct = defaultdict(list)
        for r in ph_rows:
            internal = json.loads(r["internal_pause_durations_ms"])
            by_punct[r["punct_type"]].extend(internal)
        comma_mean = np.mean(by_punct.get("comma", [0]))
        ellipsis_mean = np.mean(by_punct.get("ellipsis", [0]))
        hierarchy_diff = ellipsis_mean - comma_mean if comma_mean > 0 else 0

        score = {
            "model": model,
            "question_vs_period_f0_diff_hz": round(float(qp_diff), 1),
            "comma_to_ellipsis_pause_diff_ms": round(float(hierarchy_diff), 1),
        }
        print(f"{model}: question-f0-diff={qp_diff:.1f}Hz, comma-ellipsis-pause={hierarchy_diff:.0f}ms")
        scores[model] = score

    with open(SCORES_JSON, "w") as f:
        json.dump(scores, f, indent=2, default=str)

    return scores


def main():
    data = {}
    for r in load_csv(FEATURES_CSV):
        data.setdefault(r["model"], []).append(r)

    print(f"Loaded data for {len(data)} models, {sum(len(v) for v in data.values())} utterances\n")

    se = sentence_end_analysis(data)
    ph = pause_hierarchy_analysis(data)
    tr = trailing_analysis(data)
    ca = capitalization_analysis(data)
    qu = quotation_analysis(data)
    ov = overall_score(data)

    # Save full analysis
    analysis = {
        "sentence_end": se,
        "pause_hierarchy": ph,
        "trailing": tr,
        "capitalization": ca,
        "quotation": qu,
        "overall": ov,
        "metadata": {
            "description": "Descriptive statistics from a 28-item punctuation-prosody probe. "
                           "Not a validated benchmark. CIs are bootstrap 95% percentile intervals.",
            "n_items_per_model": 28,
            "bootstrap_iterations": BOOTSTRAP_ITER,
            "gate_threshold_ms": CONFIG["gate"]["period_min_ms"],
            "quotation_range_threshold_hz": CONFIG["quotation"]["range_shift_threshold_hz"],
        },
    }
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nFull analysis saved to {ANALYSIS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())