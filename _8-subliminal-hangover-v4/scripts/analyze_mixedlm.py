#!/usr/bin/env python3
"""
Placeholder for V4 analysis.

Expectation:
  - input: features/features.csv produced by extract_features.py
  - output: results/stats.json and per-model mixed model summaries

This file is scaffolded now so the workflow shape is fixed early.
"""

from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def main() -> None:
    features_csv = PROJECT_DIR / "features" / "features.csv"
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        raise SystemExit(f"missing {features_csv} (run extract_features.py first)")

    raise SystemExit(
        "analyze_mixedlm.py scaffold only. Implement MixedLM once V4 prompts/manifest are locked."
    )


if __name__ == "__main__":
    main()

