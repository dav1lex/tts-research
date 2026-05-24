#!/usr/bin/env python3
"""Extract prosody features (F0 mean, std, range, speaking rate) from WAVs.

Voiced-frame filtering matches _2-breathiness-preservation-benchmark exactly:
  - librosa.pyin fmin=60, fmax=400
  - Energy threshold: max(percentile(rms,20)*0.5, max(rms)*0.01, 1e-8)
  - Voiced probability >= 0.50
  - Frame length 0.04s, hop length 0.01s
  - Fallback to librosa.yin if voiced ratio < 0.10

Also saves F0 contours (time, f0) as .npy for DTW analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from common import metadata_base_dir, read_csv, resolve_audio_path, validate_metadata, write_csv

TARGET_SR = 16000
F0_MIN = 60.0
F0_MAX = 400.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01
MIN_VOICED_RATIO = 0.10
MIN_VOICED_SECONDS = 0.20


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("empty audio")
    if sample_rate != TARGET_SR:
        audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=TARGET_SR)
        sample_rate = TARGET_SR
    return audio.astype(float), sample_rate


def voiced_mask(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute voiced-frame mask using the same logic as _2 breathiness benchmark."""
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_threshold = max(
        float(np.percentile(rms, 20)) * 0.5,
        float(np.max(rms)) * 0.01,
        1e-8,
    )

    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sample_rate,
        frame_length=frame_samples,
        hop_length=hop_samples,
        center=False,
    )
    frame_count = min(len(rms), len(voiced_flag), len(voiced_probability))
    mask = (
        (rms[:frame_count] >= energy_threshold)
        & voiced_flag[:frame_count].astype(bool)
        & (voiced_probability[:frame_count] >= 0.50)
    )

    if np.mean(mask) < MIN_VOICED_RATIO:
        fallback_f0 = librosa.yin(
            audio,
            fmin=F0_MIN,
            fmax=F0_MAX,
            sr=sample_rate,
            frame_length=frame_samples,
            hop_length=hop_samples,
            center=False,
        )
        fallback_count = min(len(rms), len(fallback_f0))
        fallback_f0 = np.nan_to_num(fallback_f0[:fallback_count], nan=0.0, posinf=0.0, neginf=0.0)
        fallback_mask = (
            (rms[:fallback_count] >= energy_threshold)
            & (fallback_f0 >= F0_MIN)
            & (fallback_f0 <= F0_MAX)
        )
        if np.mean(fallback_mask) > np.mean(mask):
            return fallback_mask, fallback_f0, rms[:fallback_count]

    return mask, np.nan_to_num(f0[:frame_count], nan=0.0), rms[:frame_count]


def speaking_rate(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate speaking rate as syllable nuclei per second.

    Uses RMS energy envelope peak detection — a standard phonetic proxy.
    """
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))

    # Smooth RMS with moving average
    window = int(round(0.05 / HOP_LENGTH))  # ~50 ms smoothing
    if window > 1 and len(rms) > window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rms, kernel, mode="same")
    else:
        smoothed = rms

    # Detect peaks: local maxima above median RMS
    threshold = float(np.median(smoothed)) * 1.2
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and smoothed[i] > threshold:
            peaks.append(i)

    duration = len(audio) / sample_rate
    if duration <= 0:
        return 0.0
    return len(peaks) / duration


def f0_contour_path(path: Path, contours_dir: Path) -> Path:
    """Derive .npy contour filename from a WAV path."""
    # Use a sanitized name based on the original path
    stem = str(path).replace("/", "_").replace(".wav", "")
    return contours_dir / f"{stem}_f0_contour.npy"


def extract_one(path: Path, contours_dir: Path | None = None) -> dict[str, object]:
    audio, sample_rate = load_mono(path)
    mask, f0_vals, rms_vals = voiced_mask(audio, sample_rate)

    duration = len(audio) / sample_rate
    voiced_seconds = float(np.sum(mask) * HOP_LENGTH)
    voiced_ratio = float(np.mean(mask)) if len(mask) else 0.0

    if voiced_ratio < MIN_VOICED_RATIO or voiced_seconds < MIN_VOICED_SECONDS:
        raise ValueError(
            f"too little voiced material: ratio={voiced_ratio:.4f}, seconds={voiced_seconds:.4f}"
        )

    # F0 statistics over voiced frames only
    voiced_f0 = f0_vals[mask]
    if len(voiced_f0) == 0:
        raise ValueError("no voiced frames after masking")

    f0_mean = float(np.mean(voiced_f0))
    f0_std = float(np.std(voiced_f0, ddof=1))
    f0_range = float(np.max(voiced_f0) - np.min(voiced_f0))

    # Speaking rate
    rate = speaking_rate(audio, sample_rate)

    # Save F0 contour for DTW (time, f0 pairs — zero for unvoiced)
    if contours_dir is not None:
        contour_path = f0_contour_path(path, contours_dir)
        contour_path.parent.mkdir(parents=True, exist_ok=True)
        # Time axis for each frame
        n_frames = len(f0_vals)
        times = np.arange(n_frames) * HOP_LENGTH + FRAME_LENGTH / 2
        contour = np.column_stack([times, f0_vals])
        np.save(str(contour_path), contour)

    return {
        "duration_sec": round(duration, 4),
        "voiced_seconds": round(voiced_seconds, 4),
        "voiced_frame_ratio": round(voiced_ratio, 4),
        "f0_mean": round(f0_mean, 6),
        "f0_std": round(f0_std, 6),
        "f0_range": round(f0_range, 6),
        "speaking_rate": round(rate, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract prosody features from references and model outputs")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--features-dir", default=Path("features"), type=Path)
    args = parser.parse_args()

    rows = read_csv(args.metadata)
    try:
        validate_metadata(rows)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    base_dir = metadata_base_dir(args.metadata)
    features_dir = args.features_dir
    contours_dir = features_dir / "contours"
    features_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    failures = []
    seen_references = set()

    for row in rows:
        # Extract reference features (once per unique reference)
        reference_path = resolve_audio_path(base_dir, row["reference_path"])
        if reference_path and reference_path not in seen_references:
            seen_references.add(reference_path)
            try:
                features = extract_one(reference_path, contours_dir)
            except Exception as error:
                failures.append(f"reference {reference_path}: {error}")
            else:
                feature_rows.append({
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "condition": row["condition"].strip().lower(),
                    "type": "reference",
                    "model": "",
                    "path": str(reference_path),
                    **features,
                })

        # Extract output features
        output_path = resolve_audio_path(base_dir, row["output_path"])
        if output_path:
            try:
                features = extract_one(output_path, contours_dir)
            except Exception as error:
                failures.append(f"output {output_path}: {error}")
            else:
                feature_rows.append({
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "condition": row["condition"].strip().lower(),
                    "type": "output",
                    "model": row["model"].strip(),
                    "path": str(output_path),
                    **features,
                })

    if failures:
        for failure in failures:
            print(f"WARN: {failure}", file=sys.stderr)

    if not feature_rows:
        print("ERROR: no features extracted", file=sys.stderr)
        return 1

    fieldnames = [
        "sample_id", "pair_id", "condition", "type", "model", "path",
        "duration_sec", "voiced_seconds", "voiced_frame_ratio",
        "f0_mean", "f0_std", "f0_range", "speaking_rate",
    ]
    write_csv(features_dir / "features.csv", feature_rows, fieldnames)
    print(f"Wrote {len(feature_rows)} rows to {features_dir / 'features.csv'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
