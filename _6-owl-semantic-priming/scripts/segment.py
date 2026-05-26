#!/usr/bin/env python3
"""Extract target-sentence audio from generated priming clips.

The experiment tests whether the final neutral sentence changes after a
semantic prime. Full primed clips include the prime paragraph and are invalid
for feature extraction. This script cuts each clip down to the target sentence.

For cold clips, the whole audio is copied. For primed clips, the target is
always last, so the segmenter finds a long silence whose remaining audio
duration is compatible with the expected target duration and cuts after it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import CONDITIONS, MODELS, N_REPS, TARGET_SR  # noqa: E402

FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01
MIN_BOUNDARY_SILENCE = 0.18
START_PAD = 0.08

EXPECTED_TARGET_DURATION = 3.6
SOFT_MIN_DURATION = 2.5
SOFT_MAX_DURATION = 5.0
HARD_MIN_DURATION = 2.0
HARD_MAX_DURATION = 6.0
MIN_RMS = 0.005
MIN_VOICED_RATIO = 0.20


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


def frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    if len(audio) < frame_samples:
        return np.array([float(np.sqrt(np.mean(audio * audio)))])
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    return np.sqrt(np.mean(frames * frames, axis=1))


def silence_threshold(rms: np.ndarray) -> float:
    positive = rms[rms > 0]
    if len(positive) == 0:
        return MIN_RMS
    return max(
        float(np.percentile(positive, 15)) * 1.75,
        float(np.max(positive)) * 0.025,
        1e-5,
    )


def find_silences(audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    rms = frame_rms(audio, sample_rate)
    is_silent = rms < silence_threshold(rms)

    silences: list[tuple[float, float]] = []
    start: int | None = None
    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            duration = (i - start) * HOP_LENGTH
            if duration >= MIN_BOUNDARY_SILENCE:
                silences.append((start * HOP_LENGTH, i * HOP_LENGTH))
            start = None

    if start is not None:
        duration = (len(is_silent) - start) * HOP_LENGTH
        if duration >= MIN_BOUNDARY_SILENCE:
            silences.append((start * HOP_LENGTH, len(is_silent) * HOP_LENGTH))

    return silences


def choose_boundary(audio: np.ndarray, sample_rate: int) -> tuple[float, str]:
    total_duration = len(audio) / sample_rate
    candidates = []
    fallback = []

    for start, end in find_silences(audio, sample_rate):
        remaining = total_duration - end + START_PAD
        if HARD_MIN_DURATION <= remaining <= HARD_MAX_DURATION:
            candidates.append((abs(remaining - EXPECTED_TARGET_DURATION), end, start, remaining))
        elif remaining >= HARD_MIN_DURATION:
            fallback.append((remaining, end, start))

    if candidates:
        _, end, _, _ = min(candidates, key=lambda row: (row[0], -row[1]))
        return max(0.0, end - START_PAD), "vad"

    if fallback:
        _, end, _ = min(fallback, key=lambda row: row[0])
        return max(0.0, end - START_PAD), "vad_fallback"

    raise ValueError("no usable boundary pause found")


def voiced_ratio(audio: np.ndarray, sample_rate: int) -> float:
    rms = frame_rms(audio, sample_rate)
    threshold = max(float(np.percentile(rms, 20)) * 0.5, float(np.max(rms)) * 0.01, 1e-8)
    voiced = rms >= threshold
    return float(np.mean(voiced)) if len(voiced) else 0.0


def qc_flags(duration: float, rms: float, ratio: float, method: str) -> list[str]:
    flags = []
    if duration < HARD_MIN_DURATION:
        flags.append("duration_too_short")
    elif duration > HARD_MAX_DURATION:
        flags.append("duration_too_long")
    elif duration < SOFT_MIN_DURATION or duration > SOFT_MAX_DURATION:
        flags.append("duration_review")
    if rms < MIN_RMS:
        flags.append("low_rms")
    if ratio < MIN_VOICED_RATIO:
        flags.append("low_voiced_ratio")
    if method == "vad_fallback":
        flags.append("boundary_fallback")
    return flags


def audio_hash(audio: np.ndarray) -> str:
    normalized = np.ascontiguousarray(audio.astype(np.float32))
    return hashlib.sha256(normalized.tobytes()).hexdigest()[:16]


def write_segment(audio: np.ndarray, sample_rate: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)


def segment_one(in_path: Path, out_path: Path, condition: str) -> dict[str, object]:
    audio, sample_rate = load_mono(in_path)
    total_duration = len(audio) / sample_rate

    if condition == "cold":
        start_time = 0.0
        method = "cold_full_clip"
    else:
        start_time, method = choose_boundary(audio, sample_rate)

    start_sample = min(len(audio), int(round(start_time * sample_rate)))
    segment = audio[start_sample:]
    duration = len(segment) / sample_rate
    rms = float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.0
    ratio = voiced_ratio(segment, sample_rate) if len(segment) else 0.0
    flags = qc_flags(duration, rms, ratio, method)

    write_segment(segment, sample_rate, out_path)

    return {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "method": method,
        "source_duration": round(total_duration, 4),
        "start_time": round(start_time, 4),
        "segment_duration": round(duration, 4),
        "rms_energy": round(rms, 6),
        "voiced_ratio": round(ratio, 4),
        "segment_sha256": audio_hash(segment),
        "flags": "|".join(flags),
    }


def flag_duplicate_segments(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[object, object], dict[object, list[dict[str, object]]]] = {}
    for row in rows:
        key = (row["model"], row["condition"])
        groups.setdefault(key, {}).setdefault(row["segment_sha256"], []).append(row)

    for hashes in groups.values():
        for duplicate_rows in hashes.values():
            if len(duplicate_rows) < 2:
                continue
            for row in duplicate_rows:
                flags = str(row["flags"])
                row["flags"] = "duplicate_segment_hash" if not flags else f"{flags}|duplicate_segment_hash"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract target-only WAVs from priming outputs")
    parser.add_argument("--input-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--output-dir", default=Path("segments"), type=Path)
    parser.add_argument("--qc-csv", default=Path("results/segment_qc.csv"), type=Path)
    parser.add_argument("--clean", action="store_true", help="Remove old segment directory first")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    total = len(MODELS) * len(CONDITIONS) * N_REPS
    count = 0

    for model in MODELS:
        for condition in CONDITIONS:
            for rep in range(1, N_REPS + 1):
                in_path = args.input_dir / model / condition / f"r{rep}.wav"
                out_path = args.output_dir / model / condition / f"r{rep}.wav"
                if not in_path.exists():
                    failures.append(f"MISSING: {in_path}")
                    continue

                try:
                    row = segment_one(in_path, out_path, condition)
                    row.update({"model": model, "condition": condition, "rep": rep})
                    rows.append(row)
                    count += 1
                    flag_text = f" flags={row['flags']}" if row["flags"] else ""
                    print(
                        f"[{count}/{total}] {model}/{condition}/r{rep} "
                        f"start={row['start_time']}s dur={row['segment_duration']}s{flag_text}"
                    )
                except Exception as e:
                    failures.append(f"ERROR {in_path}: {e}")

    flag_duplicate_segments(rows)

    args.qc_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "condition", "rep", "input_path", "output_path", "method",
        "source_duration", "start_time", "segment_duration", "rms_energy",
        "voiced_ratio", "segment_sha256", "flags",
    ]
    with args.qc_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        for failure in failures:
            print(f"WARN: {failure}", file=sys.stderr)

    flagged = sum(1 for row in rows if row["flags"])
    print(f"\nWrote {len(rows)} segments to {args.output_dir}")
    print(f"Wrote QC to {args.qc_csv} ({flagged} flagged)")
    return 0 if not failures and flagged == 0 else 2 if failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
