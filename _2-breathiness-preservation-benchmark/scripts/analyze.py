#!/usr/bin/env python3
"""Analyze reference-to-output breathiness preservation after gate approval."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import METRICS, read_csv, safe_float, write_csv


def robust_scale(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    std = float(np.std(values, ddof=1))
    scale = iqr / 1.349 if iqr > 0 else std
    return scale if scale > 1e-6 else 1.0


def require_gate(results_dir: Path) -> None:
    gate_json = results_dir / "gate_check.json"
    marker = results_dir / ".gate_passed"
    if not gate_json.exists() or not marker.exists():
        raise ValueError("gate has not passed; run gate_check.py and fix references before analysis")
    gate = json.loads(gate_json.read_text())
    if not gate.get("passed"):
        raise ValueError("gate_check.json says gate failed; analysis is blocked")


def index_rows(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    references = {}
    outputs = []
    for row in rows:
        key = (row["sample_id"], row["condition"])
        if row["type"] == "reference":
            references[key] = row
        elif row["type"] == "output":
            outputs.append(row)
    return references, outputs


def preservation_rows(rows: list[dict[str, str]]) -> list[dict]:
    references, outputs = index_rows(rows)
    reference_scales = {
        metric: robust_scale([safe_float(row[metric]) for row in references.values()])
        for metric in METRICS
    }

    output_rows = []
    for output in outputs:
        reference = references.get((output["sample_id"], output["condition"]))
        if not reference:
            continue

        row = {
            "sample_id": output["sample_id"],
            "pair_id": output["pair_id"],
            "condition": output["condition"],
            "model": output["model"],
        }
        distances = []
        for metric in METRICS:
            ref_value = safe_float(reference[metric])
            out_value = safe_float(output[metric])
            delta = out_value - ref_value
            scaled_abs_delta = abs(delta) / reference_scales[metric]
            row[f"{metric}_ref"] = ref_value
            row[f"{metric}_output"] = out_value
            row[f"{metric}_delta"] = delta
            row[f"{metric}_scaled_abs_delta"] = scaled_abs_delta
            distances.append(scaled_abs_delta)
        row["mean_scaled_abs_delta"] = float(np.mean(distances))
        output_rows.append(row)
    return output_rows


def contrast_rows(per_sample: list[dict]) -> list[dict]:
    by_model_pair: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in per_sample:
        by_model_pair[(row["model"], row["pair_id"])][row["condition"]] = row

    rows = []
    for (model, pair_id), conditions in by_model_pair.items():
        if "breathy" not in conditions or "neutral" not in conditions:
            continue
        breathy = conditions["breathy"]
        neutral = conditions["neutral"]
        row = {"model": model, "pair_id": pair_id}
        retained = []
        for metric in METRICS:
            ref_contrast = safe_float(breathy[f"{metric}_ref"]) - safe_float(neutral[f"{metric}_ref"])
            out_contrast = safe_float(breathy[f"{metric}_output"]) - safe_float(neutral[f"{metric}_output"])
            ratio = out_contrast / ref_contrast if abs(ref_contrast) > 1e-6 else np.nan
            error = abs(out_contrast - ref_contrast)
            row[f"{metric}_reference_contrast"] = ref_contrast
            row[f"{metric}_output_contrast"] = out_contrast
            row[f"{metric}_contrast_ratio"] = ratio
            row[f"{metric}_contrast_error"] = error
            if np.isfinite(ratio):
                retained.append(max(0.0, min(1.0, ratio)))
        row["mean_contrast_retention"] = float(np.mean(retained)) if retained else 0.0
        rows.append(row)
    return rows


def ranking_rows(per_sample: list[dict], contrasts: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    by_model_contrast = defaultdict(list)
    for row in per_sample:
        by_model[row["model"]].append(row)
    for row in contrasts:
        by_model_contrast[row["model"]].append(row)

    rankings = []
    for model, samples in by_model.items():
        contrast_samples = by_model_contrast.get(model, [])
        row = {
            "model": model,
            "n_outputs": len(samples),
            "n_pairs": len(contrast_samples),
            "mean_scaled_abs_delta": float(np.mean([sample["mean_scaled_abs_delta"] for sample in samples])),
            "median_scaled_abs_delta": float(np.median([sample["mean_scaled_abs_delta"] for sample in samples])),
            "mean_contrast_retention": float(np.mean([sample["mean_contrast_retention"] for sample in contrast_samples]))
            if contrast_samples else 0.0,
        }
        for metric in METRICS:
            values = [sample[f"{metric}_scaled_abs_delta"] for sample in samples]
            row[f"{metric}_mean_scaled_abs_delta"] = float(np.mean(values))
        row["score"] = row["mean_scaled_abs_delta"] - row["mean_contrast_retention"]
        rankings.append(row)

    rankings.sort(key=lambda row: (row["score"], row["mean_scaled_abs_delta"]))
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze breathiness preservation")
    parser.add_argument("--features-dir", default=Path("features"), type=Path)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--allow-failed-gate", action="store_true")
    args = parser.parse_args()

    try:
        if not args.allow_failed_gate:
            require_gate(args.results_dir)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    features_path = args.features_dir / "features.csv"
    if not features_path.exists():
        print(f"ERROR: {features_path} not found", file=sys.stderr)
        return 1

    rows = read_csv(features_path)
    per_sample = preservation_rows(rows)
    if not per_sample:
        print("ERROR: no output-reference pairs found", file=sys.stderr)
        return 1

    contrasts = contrast_rows(per_sample)
    rankings = ranking_rows(per_sample, contrasts)

    per_sample_fields = ["sample_id", "pair_id", "condition", "model"]
    for metric in METRICS:
        per_sample_fields.extend([
            f"{metric}_ref",
            f"{metric}_output",
            f"{metric}_delta",
            f"{metric}_scaled_abs_delta",
        ])
    per_sample_fields.append("mean_scaled_abs_delta")

    contrast_fields = ["model", "pair_id"]
    for metric in METRICS:
        contrast_fields.extend([
            f"{metric}_reference_contrast",
            f"{metric}_output_contrast",
            f"{metric}_contrast_ratio",
            f"{metric}_contrast_error",
        ])
    contrast_fields.append("mean_contrast_retention")

    ranking_fields = [
        "model",
        "n_outputs",
        "n_pairs",
        "mean_scaled_abs_delta",
        "median_scaled_abs_delta",
        "mean_contrast_retention",
        *[f"{metric}_mean_scaled_abs_delta" for metric in METRICS],
        "score",
    ]

    write_csv(args.results_dir / "per_sample_preservation.csv", per_sample, per_sample_fields)
    write_csv(args.results_dir / "paired_contrast_preservation.csv", contrasts, contrast_fields)
    write_csv(args.results_dir / "model_rankings.csv", rankings, ranking_fields)
    print(f"Wrote {args.results_dir / 'per_sample_preservation.csv'}")
    print(f"Wrote {args.results_dir / 'paired_contrast_preservation.csv'}")
    print(f"Wrote {args.results_dir / 'model_rankings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
