#!/usr/bin/env python3
"""Analyze prosody preservation: DTW distance, stat preservation, model rankings.

Scoring (same as _2):
  score = mean_scaled_abs_delta - mean_contrast_retention
  Lower is better.

DTW distance computed between reference and output F0 contours.
"""

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
        raise ValueError("gate has not passed; run gate_check.py first")
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


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute normalized DTW distance between two 1D sequences."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(seq_a[i - 1]) - float(seq_b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


def load_f0_contour(contour_path: Path) -> np.ndarray | None:
    """Load F0 contour from .npy file. Returns voiced F0 values only."""
    try:
        data = np.load(str(contour_path))
        # data is (N, 2): col0=time, col1=f0
        f0_vals = data[:, 1]
        # Return only voiced frames (f0 > 0)
        voiced = f0_vals[f0_vals > 0]
        return voiced if len(voiced) > 5 else None
    except Exception:
        return None


def preservation_rows(rows: list[dict], contours_dir: Path) -> list[dict]:
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

        row["stat_scaled_abs_delta"] = float(np.mean(distances))

        # DTW distance between F0 contours
        ref_contour_path = _contour_path(rows, output, contours_dir, is_reference=True)
        out_contour_path = _contour_path(rows, output, contours_dir, is_reference=False)
        dtw = 0.0
        if ref_contour_path and out_contour_path:
            ref_f0 = load_f0_contour(ref_contour_path)
            out_f0 = load_f0_contour(out_contour_path)
            if ref_f0 is not None and out_f0 is not None:
                dtw = dtw_distance(ref_f0, out_f0)
        row["dtw_distance"] = round(dtw, 6)

        output_rows.append(row)
    return output_rows


def _contour_path(features_rows: list[dict], output_row: dict,
                  contours_dir: Path, is_reference: bool) -> Path | None:
    """Find the F0 contour .npy path for a given reference or output."""
    if is_reference:
        # Find the reference row matching this sample_id + condition
        for r in features_rows:
            if r["type"] == "reference" and r["sample_id"] == output_row["sample_id"] \
               and r["condition"] == output_row["condition"]:
                return _path_to_contour(r["path"], contours_dir)
    else:
        return _path_to_contour(output_row.get("path", ""), contours_dir)
    return None


def _path_to_contour(audio_path: str | Path, contours_dir: Path) -> Path:
    audio_path = str(audio_path)
    stem = audio_path.replace("/", "_").replace(".wav", "")
    return contours_dir / f"{stem}_f0_contour.npy"


def contrast_rows(per_sample: list[dict]) -> list[dict]:
    by_model_pair: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in per_sample:
        by_model_pair[(row["model"], row["pair_id"])][row["condition"]] = row

    rows = []
    for (model, pair_id), conditions in by_model_pair.items():
        if "breathy" not in conditions or "modal" not in conditions:
            continue
        breathy = conditions["breathy"]
        modal = conditions["modal"]
        row_row = {"model": model, "pair_id": pair_id}
        retained = []
        for metric in METRICS:
            ref_contrast = safe_float(breathy[f"{metric}_ref"]) - safe_float(modal[f"{metric}_ref"])
            out_contrast = safe_float(breathy[f"{metric}_output"]) - safe_float(modal[f"{metric}_output"])
            ratio = out_contrast / ref_contrast if abs(ref_contrast) > 1e-6 else np.nan
            error = abs(out_contrast - ref_contrast)
            row_row[f"{metric}_reference_contrast"] = ref_contrast
            row_row[f"{metric}_output_contrast"] = out_contrast
            row_row[f"{metric}_contrast_ratio"] = ratio
            row_row[f"{metric}_contrast_error"] = error
            if np.isfinite(ratio):
                retained.append(max(0.0, min(1.0, ratio)))
        row_row["mean_contrast_retention"] = float(np.mean(retained)) if retained else 0.0
        rows.append(row_row)
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
            "stat_scaled_abs_delta": float(np.mean([s["stat_scaled_abs_delta"] for s in samples])),
            "dtw_distance": float(np.mean([s["dtw_distance"] for s in samples])),
            "mean_contrast_retention": float(
                np.mean([s["mean_contrast_retention"] for s in contrast_samples])
            ) if contrast_samples else 0.0,
        }

        for metric in METRICS:
            values = [s[f"{metric}_scaled_abs_delta"] for s in samples]
            row[f"{metric}_scaled_abs_delta"] = float(np.mean(values))

        # Score = scaled abs delta - contrast retention (lower = better)
        # Combine stat and DTW distances
        combined_distance = (row["stat_scaled_abs_delta"] + row["dtw_distance"]) / 2.0
        row["score"] = combined_distance - row["mean_contrast_retention"]
        rankings.append(row)

    rankings.sort(key=lambda r: (r["score"], r["stat_scaled_abs_delta"]))
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze prosody preservation")
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

    contours_dir = args.features_dir / "contours"

    rows = read_csv(features_path)
    per_sample = preservation_rows(rows, contours_dir)
    if not per_sample:
        print("ERROR: no output-reference pairs found", file=sys.stderr)
        return 1

    contrasts = contrast_rows(per_sample)
    rankings = ranking_rows(per_sample, contrasts)

    per_sample_fields = ["sample_id", "pair_id", "condition", "model"]
    for metric in METRICS:
        per_sample_fields.extend([
            f"{metric}_ref", f"{metric}_output",
            f"{metric}_delta", f"{metric}_scaled_abs_delta",
        ])
    per_sample_fields.extend(["stat_scaled_abs_delta", "dtw_distance"])

    contrast_fields = ["model", "pair_id"]
    for metric in METRICS:
        contrast_fields.extend([
            f"{metric}_reference_contrast", f"{metric}_output_contrast",
            f"{metric}_contrast_ratio", f"{metric}_contrast_error",
        ])
    contrast_fields.append("mean_contrast_retention")

    ranking_fields = [
        "model", "n_outputs", "n_pairs",
        "stat_scaled_abs_delta", "dtw_distance", "mean_contrast_retention",
    ]
    for metric in METRICS:
        ranking_fields.append(f"{metric}_scaled_abs_delta")
    ranking_fields.append("score")

    write_csv(args.results_dir / "per_sample_preservation.csv", per_sample, per_sample_fields)
    write_csv(args.results_dir / "paired_contrast_preservation.csv", contrasts, contrast_fields)
    write_csv(args.results_dir / "model_rankings.csv", rankings, ranking_fields)
    print(f"Wrote {args.results_dir / 'per_sample_preservation.csv'}")
    print(f"Wrote {args.results_dir / 'paired_contrast_preservation.csv'}")
    print(f"Wrote {args.results_dir / 'model_rankings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
