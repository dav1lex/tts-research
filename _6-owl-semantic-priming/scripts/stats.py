#!/usr/bin/env python3
"""Run exploratory statistics on target-sentence feature rows."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

FEATURES = (
    "f0_mean",
    "f0_std",
    "f0_range",
    "speech_rate",
    "pause_count",
    "pause_duration_mean",
    "rms_energy",
    "spectral_centroid",
)
CONDITIONS = ("cold", "primed_neutral", "primed_owl", "primed_death")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def eta_squared(groups: list[list[float]]) -> float:
    values = [value for group in groups for value in group]
    grand = float(np.mean(values))
    ss_between = sum(len(group) * (float(np.mean(group)) - grand) ** 2 for group in groups)
    ss_total = sum((value - grand) ** 2 for value in values)
    return ss_between / ss_total if ss_total else 0.0


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    return (float(np.mean(a)) - float(np.mean(b))) / pooled if pooled else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze priming feature statistics")
    parser.add_argument("--features-csv", default=Path("results/features.csv"), type=Path)
    parser.add_argument("--out-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    rows = load_rows(args.features_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by[(row["model"], row["condition"])].append(row)

    models = sorted({row["model"] for row in rows})
    summary_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for model in models:
        for feature in FEATURES:
            groups = []
            complete = True
            for condition in CONDITIONS:
                values = [float(row[feature]) for row in by[(model, condition)]]
                if not values:
                    complete = False
                groups.append(values)
                mean, sd = mean_sd(values) if values else (math.nan, math.nan)
                summary_rows.append({
                    "model": model,
                    "feature": feature,
                    "condition": condition,
                    "n": len(values),
                    "mean": round(mean, 6),
                    "sd": round(sd, 6),
                })

            if not complete:
                continue

            f_stat, p_value = stats.f_oneway(*groups)
            eta = eta_squared(groups)
            summary_rows.append({
                "model": model,
                "feature": feature,
                "condition": "ANOVA",
                "n": sum(len(group) for group in groups),
                "mean": round(float(f_stat), 6),
                "sd": round(float(p_value), 10),
                "eta_squared": round(float(eta), 6),
            })

            for cond_a, cond_b in itertools.combinations(CONDITIONS, 2):
                vals_a = [float(row[feature]) for row in by[(model, cond_a)]]
                vals_b = [float(row[feature]) for row in by[(model, cond_b)]]
                t_stat, pair_p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
                pairwise_rows.append({
                    "model": model,
                    "feature": feature,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "mean_a": round(float(np.mean(vals_a)), 6),
                    "mean_b": round(float(np.mean(vals_b)), 6),
                    "t": round(float(t_stat), 6) if math.isfinite(float(t_stat)) else float(t_stat),
                    "p": round(float(pair_p), 10) if math.isfinite(float(pair_p)) else float(pair_p),
                    "p_bonferroni_6": round(min(float(pair_p) * 6, 1.0), 10)
                    if math.isfinite(float(pair_p)) else float(pair_p),
                    "cohens_d": round(cohens_d(vals_a, vals_b), 6),
                })

    summary_path = args.out_dir / "stats_summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = ["model", "feature", "condition", "n", "mean", "sd", "eta_squared"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    pairwise_path = args.out_dir / "stats_pairwise.csv"
    with pairwise_path.open("w", newline="") as f:
        fieldnames = [
            "model", "feature", "condition_a", "condition_b",
            "mean_a", "mean_b", "t", "p", "p_bonferroni_6", "cohens_d",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {pairwise_path}")

    for model in models:
        print(f"\n{model}")
        for feature in FEATURES:
            anova = [
                row for row in summary_rows
                if row["model"] == model and row["feature"] == feature and row["condition"] == "ANOVA"
            ]
            if not anova:
                continue
            p_value = float(anova[0]["sd"])
            eta = float(anova[0]["eta_squared"])
            marker = "*" if p_value < 0.05 else " "
            print(f"{marker} {feature:20s} p={p_value:.4g} eta2={eta:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
