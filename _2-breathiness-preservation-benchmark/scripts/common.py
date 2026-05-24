#!/usr/bin/env python3
"""Shared utilities for the breathiness preservation benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

PRIMARY_METRIC = "cpp_mean"
METRICS = ("cpp_mean", "hnr_mean", "spectral_tilt_mean")
LOWER_IS_BREATHIER = {
    "cpp_mean": True,
    "hnr_mean": True,
    "spectral_tilt_mean": True,
}
REQUIRED_COLUMNS = {
    "sample_id",
    "pair_id",
    "text",
    "condition",
    "reference_path",
    "output_path",
    "model",
    "seed",
    "notes",
}
VALID_CONDITIONS = {"breathy", "neutral"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metadata_base_dir(metadata_path: Path) -> Path:
    return metadata_path.resolve().parent


def resolve_audio_path(base_dir: Path, value: str) -> Path | None:
    value = value.strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def validate_metadata(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("metadata is empty")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"metadata missing required columns: {', '.join(sorted(missing))}")

    seen = set()
    pair_conditions: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        pair_id = row["pair_id"].strip()
        condition = row["condition"].strip().lower()
        model = row["model"].strip()
        key = (sample_id, model, row["output_path"].strip())

        if not sample_id:
            raise ValueError(f"line {index}: sample_id is required")
        if not pair_id:
            raise ValueError(f"line {index}: pair_id is required")
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"line {index}: condition must be one of {sorted(VALID_CONDITIONS)}")
        if key in seen:
            raise ValueError(f"line {index}: duplicate sample/model/output row")
        seen.add(key)
        pair_conditions.setdefault(pair_id, set()).add(condition)

    incomplete = [pair_id for pair_id, conditions in pair_conditions.items() if conditions != VALID_CONDITIONS]
    if incomplete:
        raise ValueError(
            "each pair_id must contain both breathy and neutral references; incomplete: "
            + ", ".join(sorted(incomplete))
        )


def safe_float(value: str | float | int, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
