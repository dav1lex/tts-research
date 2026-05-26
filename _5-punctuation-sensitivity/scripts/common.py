#!/usr/bin/env python3
"""Shared utilities for the punctuation sensitivity benchmark.

All scripts under scripts/ should import from here instead of hardcoding
paths, constants, or repeating utility functions.
"""
import csv
import json
import os
import yaml
from pathlib import Path
from typing import Any, Optional


# ── Project root detection ──────────────────────────────────────────────────
# All paths derived from this, so scripts are relocatable.

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent  # scripts/../  == _5-punctuation-sensitivity/


# ── Config ───────────────────────────────────────────────────────────────────

_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


CONFIG: dict[str, Any] = load_config()


# ── Derived paths ────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
FIG_DIR = RESULTS_DIR / "figures"

CORPUS = DATA_DIR / "test_corpus.csv"
FEATURES_CSV = FEATURES_DIR / "pause_features.csv"
ANALYSIS_JSON = RESULTS_DIR / "analysis.json"
GATE_JSON = RESULTS_DIR / "gate_check.json"
SCORES_JSON = RESULTS_DIR / "sensitivity_scores.json"
TEXT_NORM_LOG = RESULTS_DIR / "text_normalization_log.json"

# Ensure result directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Model constants ──────────────────────────────────────────────────────────

MODEL_ORDER = ["chatterbox", "xtts", "kokoro"]

MODEL_LABELS: dict[str, str] = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro (no-adaptation baseline)",
}

MODEL_COLORS: dict[str, str] = {
    "chatterbox": "#2196F3",
    "xtts": "#FF9800",
    "kokoro": "#9E9E9E",
}

# Punctuation category constants
PUNCT_ORDER = ["period", "exclamation", "question"]
PUNCT_LABELS: dict[str, str] = {
    "period": "Period (.)",
    "exclamation": "Exclamation (!)",
    "question": "Question (?)",
}

HIERARCHY_ORDER = ["comma", "semicolon", "em_dash", "ellipsis"]
HIERARCHY_LABELS: dict[str, str] = {
    "comma": "Comma (,)",
    "semicolon": "Semicolon (;)",
    "em_dash": "Em-dash (—)",
    "ellipsis": "Ellipsis (...)",
}

CATEGORIES = ["sentence_end", "pause_hierarchy", "trailing", "quotation", "capitalization"]
CATEGORY_LABELS: dict[str, str] = {
    "sentence_end": "Sentence End",
    "pause_hierarchy": "Pause Hierarchy",
    "trailing": "Trailing",
    "quotation": "Quotation",
    "capitalization": "Capitalization",
}


# ── Utilities ────────────────────────────────────────────────────────────────


def safe_float(val: Any) -> Optional[float]:
    """Convert a string or number to float, returning None for missing/invalid."""
    try:
        return float(val) if val is not None and val != "" and val != "None" else None
    except (ValueError, TypeError):
        return None


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_features() -> dict[str, list[dict[str, str]]]:
    """Load features CSV and group rows by model."""
    from collections import defaultdict

    rows = load_csv(FEATURES_CSV)
    data: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        data[r["model"]].append(r)
    return data


def fmt(val: Any, decimals: int = 1) -> str:
    """Format a number for display, handling None and small values."""
    if val is None:
        return "\u2014"  # em-dash
    try:
        v = float(val)
        if abs(v) < 0.01 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)