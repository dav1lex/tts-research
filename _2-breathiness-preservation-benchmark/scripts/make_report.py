#!/usr/bin/env python3
"""Build a concise Markdown report from gate and analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_csv


def fmt(value: str | float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create benchmark Markdown report")
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    gate_path = args.results_dir / "gate_check.json"
    rankings_path = args.results_dir / "model_rankings.csv"
    contrast_path = args.results_dir / "paired_contrast_preservation.csv"
    output_path = args.output or args.results_dir / "report.md"

    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"passed": False, "metrics": []}
    rankings = read_csv(rankings_path) if rankings_path.exists() else []
    contrasts = read_csv(contrast_path) if contrast_path.exists() else []

    lines = [
        "# Breathiness Preservation Benchmark Report",
        "",
        f"Gate status: **{'PASSED' if gate.get('passed') else 'FAILED'}**",
        "",
        "## Gate Metrics",
        "",
    ]

    gate_rows = [
        {
            "metric": metric["metric"],
            "breathy_mean": metric["breathy_mean"],
            "neutral_mean": metric["neutral_mean"],
            "cohens_d": metric["cohens_d"],
            "passed": metric["passed"],
        }
        for metric in gate.get("metrics", [])
    ]
    lines.append(markdown_table(gate_rows, ["metric", "breathy_mean", "neutral_mean", "cohens_d", "passed"]))

    lines.extend([
        "",
        "## Model Rankings",
        "",
        "Lower scaled distance is better. Higher contrast retention is better. Lower score is better.",
        "",
        markdown_table(
            rankings,
            [
                "model",
                "n_outputs",
                "n_pairs",
                "mean_scaled_abs_delta",
                "mean_contrast_retention",
                "score",
            ],
        ),
        "",
        "## Paired Contrast Preservation",
        "",
        markdown_table(
            contrasts,
            [
                "model",
                "pair_id",
                "cpp_mean_contrast_ratio",
                "hnr_mean_contrast_ratio",
                "spectral_tilt_mean_contrast_ratio",
                "mean_contrast_retention",
            ],
        ),
        "",
        "## Method",
        "",
        "- Primary metric: Praat CPPS/CPP over voiced intervals.",
        "- Supporting metrics: Praat harmonicity/HNR and voiced-frame spectral tilt.",
        "- Gate: reference clips must separate breathy from neutral before analysis is allowed.",
        "- Scoring: output-reference metric distance uses robust dataset-level reference scales.",
        "- Contrast retention: paired breathy-neutral differences test whether the model preserves the breathiness contrast.",
        "",
        "## Caveats",
        "",
        "- Objective breathiness proxies are sensitive to microphone, noise, loudness, and phonetic content.",
        "- Use matched text and recording conditions wherever possible.",
        "- Treat rankings as invalid unless the gate passes.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
