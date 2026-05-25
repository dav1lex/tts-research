#!/usr/bin/env python3
"""Analyze punctuation sensitivity from extracted features."""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT = Path("/home/davilex/tts-research/_5-punctuation-sensitivity")
FEATURES = PROJECT / "results" / "features" / "pause_features.csv"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    rows = list(csv.DictReader(open(FEATURES)))
    data = defaultdict(list)
    for r in rows:
        data[r["model"]].append(r)
    return data


def safe_float(val):
    try:
        return float(val) if val and val != "None" else None
    except (ValueError, TypeError):
        return None


def sentence_end_analysis(data):
    """Analyze F0 slope differences between terminal punctuation types."""
    print("\n" + "="*60)
    print("1. SENTENCE-END PUNCTUATION SENSITIVITY")
    print("="*60)
    print("Expected: period=flat/fall, exclamation=rise-fall, question=rise\n")

    results = {}
    for model, rows in data.items():
        se_items = [r for r in rows if r["category"] == "sentence_end"]
        by_punct = defaultdict(list)
        for r in se_items:
            slope = safe_float(r["terminal_f0_slope"])
            if slope is not None:
                by_punct[r["punct_type"]].append(slope)

        print(f"--- {model} ---")
        model_scores = {}
        for punct in ["period", "exclamation", "question"]:
            slopes = by_punct.get(punct, [])
            mean_slope = np.mean(slopes) if slopes else None
            print(f"  {punct}: F0 slope mean={mean_slope:.1f} Hz/s (n={len(slopes)})")
            model_scores[punct] = {"mean_slope": round(mean_slope, 1) if mean_slope else None, "n": len(slopes)}

        # Question vs period differentiation score
        period_slopes = by_punct.get("period", [])
        question_slopes = by_punct.get("question", [])
        if period_slopes and question_slopes:
            p_mean = np.mean(period_slopes)
            q_mean = np.mean(question_slopes)
            diff = q_mean - p_mean
            # Positive diff = question rises more than period (good)
            se_diff = round(float(diff), 1)
            print(f"  question-vs-period diff: {se_diff} Hz/s (positive=rise, negative=fall)")
            model_scores["question_period_diff"] = se_diff

        results[model] = model_scores

    return results


def pause_hierarchy_analysis(data):
    """Analyze pause duration variation by internal punctuation type."""
    print("\n" + "="*60)
    print("2. PAUSE HIERARCHY SENSITIVITY")
    print("="*60)
    print("Expected: comma < semicolon < em-dash < ellipsis\n")

    results = {}
    for model, rows in data.items():
        ph_items = [r for r in rows if r["category"] == "pause_hierarchy"]
        by_punct = defaultdict(list)
        for r in ph_items:
            internal = json.loads(r["internal_pause_durations_ms"])
            by_punct[r["punct_type"]].extend(internal)

        print(f"--- {model} ---")
        model_scores = {}
        for punct in ["comma", "semicolon", "em_dash", "ellipsis"]:
            durations = by_punct.get(punct, [])
            mean_dur = np.mean(durations) if durations else 0
            median_dur = np.median(durations) if durations else 0
            print(f"  {punct}: mean={mean_dur:.0f}ms, median={median_dur:.0f}ms (n={len(durations)})")
            model_scores[punct] = {
                "mean_ms": round(float(mean_dur), 1),
                "median_ms": round(float(median_dur), 1),
                "n": len(durations),
            }

        # Hierarchy score: is there a monotonic ordering?
        means = [model_scores[p]["mean_ms"] for p in ["comma", "semicolon", "em_dash", "ellipsis"]]
        hierarchy_correct = sum(1 for i in range(len(means)-1) if means[i] <= means[i+1])
        hierarchy_score = hierarchy_correct / 3  # 0-1, 1=perfect hierarchy
        model_scores["hierarchy_score"] = round(hierarchy_score, 2)
        print(f"  hierarchy score: {hierarchy_score:.2f} (3 pairs, {hierarchy_correct}/3 monotonic)")

        results[model] = model_scores

    return results


def trailing_analysis(data):
    """Analyze ellipsis vs period trailing behavior."""
    print("\n" + "="*60)
    print("3. TRAILING PUNCTUATION SENSITIVITY")
    print("="*60)
    print("Expected: ellipsis=trailing F0 + amplitude fade, period=sharp fall\n")

    results = {}
    for model, rows in data.items():
        t_items = [r for r in rows if r["category"] == "trailing"]
        by_punct = defaultdict(list)
        for r in t_items:
            slope = safe_float(r["terminal_f0_slope"])
            decay = safe_float(r["amplitude_decay_300ms"])
            pause = safe_float(r["best_pause_ms"])
            if slope is not None:
                by_punct[r["punct_type"]].append({
                    "f0_slope": slope,
                    "amp_decay": decay,
                    "pause_ms": pause,
                })

        print(f"--- {model} ---")
        model_scores = {}
        for punct in ["ellipsis", "period"]:
            items = by_punct.get(punct, [])
            if items:
                slopes = [x["f0_slope"] for x in items]
                decays = [x["amp_decay"] for x in items if x["amp_decay"] is not None]
                pauses = [x["pause_ms"] for x in items if x["pause_ms"] is not None]
                print(f"  {punct}: F0 slope={np.mean(slopes):.1f} Hz/s, "
                      f"amp_decay={np.mean(decays) if decays else 'N/A'}, "
                      f"pause={np.mean(pauses):.0f}ms")
                model_scores[punct] = {
                    "f0_slope_mean": round(float(np.mean(slopes)), 1),
                    "amp_decay_mean": round(float(np.mean(decays)), 10) if decays else None,
                    "pause_mean": round(float(np.mean(pauses)), 1) if pauses else None,
                }

        # Trailing distinction: ellipsis should have more negative F0 slope (trailing off)
        ell = by_punct.get("ellipsis", [])
        per = by_punct.get("period", [])
        if ell and per:
            ell_f0 = np.mean([x["f0_slope"] for x in ell])
            per_f0 = np.mean([x["f0_slope"] for x in per])
            f0_diff = ell_f0 - per_f0
            print(f"  ellipsis-vs-period F0 diff: {f0_diff:.1f} Hz/s (neg=trailing, pos=rising)")
            model_scores["trailing_f0_diff"] = round(float(f0_diff), 1)

        results[model] = model_scores

    return results


def capitalization_analysis(data):
    """Analyze ALL-CAPS vs normal emphasis."""
    print("\n" + "="*60)
    print("4. CAPITALIZATION SENSITIVITY")
    print("="*60)
    print("Expected: ALL-CAPS = higher RMS + more extreme F0\n")

    results = {}
    for model, rows in data.items():
        c_items = [r for r in rows if r["category"] == "capitalization"]
        print(f"--- {model} ---")
        for r in sorted(c_items, key=lambda x: x["subcategory"]):
            rms = safe_float(r["rms_mean"])
            f0_range = safe_float(r["f0_range"])
            f0_mean = safe_float(r["f0_mean"])
            print(f"  {r['subcategory']:12s}: RMS={rms}, F0_mean={f0_mean}Hz, F0_range={f0_range}Hz")

        # ALL_CAPS vs title_case comparison
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
    print("\n" + "="*60)
    print("5. QUOTATION SENSITIVITY")
    print("="*60)
    print("Expected: quoted speech = distinct F0 range/mean shift\n")

    results = {}
    for model, rows in data.items():
        q_items = [r for r in rows if r["category"] == "quotation"]
        by_type = defaultdict(list)
        for r in q_items:
            f0_mean = safe_float(r["f0_mean"])
            f0_range = safe_float(r["f0_range"])
            rms = safe_float(r["rms_mean"])
            if f0_mean is not None:
                by_type[r["punct_type"]].append({
                    "f0_mean": f0_mean,
                    "f0_range": f0_range,
                    "rms": rms,
                })

        print(f"--- {model} ---")
        for ptype in ["quoted", "reported"]:
            items = by_type.get(ptype, [])
            if items:
                f0_means = [x["f0_mean"] for x in items]
                f0_ranges = [x["f0_range"] for x in items]
                print(f"  {ptype}: F0_mean={np.mean(f0_means):.1f}Hz, "
                      f"F0_range={np.mean(f0_ranges):.1f}Hz")

        # Prosody shift detection
        quoted = by_type.get("quoted", [])
        reported = by_type.get("reported", [])
        if quoted and reported:
            q_f0m = np.mean([x["f0_mean"] for x in quoted])
            r_f0m = np.mean([x["f0_mean"] for x in reported])
            q_f0r = np.mean([x["f0_range"] for x in quoted])
            r_f0r = np.mean([x["f0_range"] for x in reported])

            f0_mean_shift = abs(q_f0m - r_f0m)
            f0_range_shift = abs(q_f0r - r_f0r)
            has_shift = f0_range_shift > 10  # >10Hz range difference = meaningful
            print(f"  F0 mean shift: {f0_mean_shift:.1f}Hz, F0 range shift: {f0_range_shift:.1f}Hz")
            print(f"  Prosody shift detected: {'YES' if has_shift else 'NO'}")
            results[model] = {
                "f0_mean_shift": round(float(f0_mean_shift), 1),
                "f0_range_shift": round(float(f0_range_shift), 1),
                "shift_detected": has_shift,
            }

    return results


def overall_score(data):
    """Composite sensitivity score."""
    print("\n" + "="*60)
    print("6. OVERALL PUNCTUATION SENSITIVITY SCORE")
    print("="*60)

    scores = {}
    for model, rows in data.items():
        # Count how many utterance types the model differentiates
        se_items = [r for r in rows if r["category"] == "sentence_end"]
        periods = [safe_float(r["terminal_f0_slope"]) for r in se_items if r["punct_type"] == "period" and safe_float(r["terminal_f0_slope"]) is not None]
        questions = [safe_float(r["terminal_f0_slope"]) for r in se_items if r["punct_type"] == "question" and safe_float(r["terminal_f0_slope"]) is not None]

        # Pause hierarchy
        ph_items = [r for r in rows if r["category"] == "pause_hierarchy"]
        by_punct = defaultdict(list)
        for r in ph_items:
            internal = json.loads(r["internal_pause_durations_ms"])
            by_punct[r["punct_type"]].extend(internal)
        comma_mean = np.mean(by_punct.get("comma", [0]))
        ellipsis_mean = np.mean(by_punct.get("ellipsis", [0]))
        hierarchy_diff = ellipsis_mean - comma_mean if comma_mean > 0 else 0

        # Sentence-end differentiation
        qp_diff = np.mean(questions) - np.mean(periods) if periods and questions else 0

        score = {
            "model": model,
            "question_vs_period_f0_diff_hz": round(float(qp_diff), 1),
            "comma_to_ellipsis_pause_diff_ms": round(float(hierarchy_diff), 1),
        }
        print(f"{model}: question-f0-diff={qp_diff:.1f}Hz, comma→ellipsis-pause={hierarchy_diff:.0f}ms")
        scores[model] = score

    with open(RESULTS_DIR / "sensitivity_scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)

    return scores


def main():
    data = load_data()
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
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nFull analysis saved to {RESULTS_DIR}/analysis.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
