#!/usr/bin/env python3
"""
Placeholder report generator for V4.

V4 report should:
  - cite PROTOCOL.md (prereg)
  - summarize design + alignment success
  - show effect sizes + CIs (mixed model)
  - include robustness checks (leave-one-target-out)
  - clearly separate primary vs exploratory analyses
"""

from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit("make_report.py scaffold only.")


if __name__ == "__main__":
    main()

