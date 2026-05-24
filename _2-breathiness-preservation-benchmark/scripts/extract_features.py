#!/usr/bin/env python3
"""Extract breathiness metrics with Praat/parselmouth and voiced-frame filtering."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

from common import metadata_base_dir, read_csv, resolve_audio_path, validate_metadata, write_csv

TARGET_SR = 16000
F0_MIN = 60.0
F0_MAX = 400.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01
MIN_VOICED_RATIO = 0.10
MIN_VOICED_SECONDS = 0.20


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("empty audio")
    if sample_rate != TARGET_SR:
        audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=TARGET_SR)
        sample_rate = TARGET_SR
    return audio.astype(float), sample_rate


def voiced_mask(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_threshold = max(float(np.percentile(rms, 20)) * 0.5, float(np.max(rms)) * 0.01, 1e-8)

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


def voiced_intervals(mask: np.ndarray) -> list[tuple[float, float]]:
    intervals = []
    start = None
    for index, is_voiced in enumerate(mask):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            intervals.append((start * HOP_LENGTH, index * HOP_LENGTH + FRAME_LENGTH))
            start = None
    if start is not None:
        intervals.append((start * HOP_LENGTH, len(mask) * HOP_LENGTH + FRAME_LENGTH))
    return intervals


def praat_pitch_values(sound: parselmouth.Sound, intervals: list[tuple[float, float]]) -> list[float]:
    values = []
    for start, end in intervals:
        if end - start < 0.03:
            continue
        part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
        pitch = part.to_pitch(time_step=HOP_LENGTH, pitch_floor=max(F0_MIN, 75.0), pitch_ceiling=F0_MAX)
        selected = pitch.selected_array["frequency"]
        values.extend(float(value) for value in selected if value > 0)
    return values


def praat_cpp(sound: parselmouth.Sound, intervals: list[tuple[float, float]]) -> float:
    values = []
    for start, end in intervals:
        if end - start < 0.10:
            continue
        part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
        try:
            power_cepstrogram = call(part, "To PowerCepstrogram", max(F0_MIN, 75.0), HOP_LENGTH, 5000, 50)
            cpp = call(
                power_cepstrogram,
                "Get CPPS",
                True,
                0.02,
                0.0005,
                max(F0_MIN, 75.0),
                330,
                0.05,
                "Parabolic",
                0.001,
                0.0,
                "Exponential decay",
                "Robust",
            )
        except Exception:
            continue
        if math.isfinite(cpp):
            values.append(float(cpp))
    if not values:
        raise ValueError("Praat CPPS failed on voiced intervals")
    return float(np.mean(values))


def praat_hnr(sound: parselmouth.Sound, intervals: list[tuple[float, float]]) -> float:
    values = []
    for start, end in intervals:
        if end - start < 0.10:
            continue
        part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
        try:
            harmonicity = part.to_harmonicity_cc(time_step=HOP_LENGTH, minimum_pitch=max(F0_MIN, 75.0))
            hnr = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            continue
        if math.isfinite(hnr):
            values.append(float(hnr))
    if not values:
        raise ValueError("Praat harmonicity failed on voiced intervals")
    return float(np.mean(values))


def spectral_tilt(audio: np.ndarray, sample_rate: int, mask: np.ndarray) -> float:
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    usable_count = min(len(frames), len(mask))
    voiced_frames = frames[:usable_count][mask[:usable_count]]
    if len(voiced_frames) == 0:
        raise ValueError("no voiced frames for spectral tilt")

    slopes = []
    frequencies = np.fft.rfftfreq(frame_samples, 1 / sample_rate)
    band = (frequencies >= 100) & (frequencies <= 5000)
    log_frequency = np.log2(frequencies[band])
    design = np.vstack([log_frequency, np.ones(len(log_frequency))]).T
    window = np.hanning(frame_samples)

    for frame in voiced_frames:
        spectrum = np.abs(np.fft.rfft(frame * window))
        db = 20 * np.log10(np.maximum(spectrum[band], 1e-10))
        slope, _ = np.linalg.lstsq(design, db, rcond=None)[0]
        slopes.append(float(slope))
    return float(np.mean(slopes))


def quality_flags(audio: np.ndarray, mask: np.ndarray, sample_rate: int) -> dict[str, object]:
    duration = len(audio) / sample_rate
    voiced_seconds = float(np.sum(mask) * HOP_LENGTH)
    voiced_ratio = float(np.mean(mask)) if len(mask) else 0.0
    return {
        "duration_sec": round(duration, 4),
        "voiced_seconds": round(voiced_seconds, 4),
        "voiced_frame_ratio": round(voiced_ratio, 4),
        "has_clipping": bool(np.max(np.abs(audio)) >= 0.99),
        "too_little_voicing": bool(voiced_ratio < MIN_VOICED_RATIO or voiced_seconds < MIN_VOICED_SECONDS),
    }


def extract_one(path: Path) -> dict[str, object]:
    audio, sample_rate = load_mono(path)
    mask, f0, _rms = voiced_mask(audio, sample_rate)
    flags = quality_flags(audio, mask, sample_rate)
    if flags["too_little_voicing"]:
        raise ValueError(
            f"too little voiced material: ratio={flags['voiced_frame_ratio']}, seconds={flags['voiced_seconds']}"
        )

    intervals = voiced_intervals(mask)
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch_values = praat_pitch_values(sound, intervals)
    voiced_f0 = [value for value in pitch_values if value > 0]

    return {
        **flags,
        "cpp_mean": round(praat_cpp(sound, intervals), 6),
        "hnr_mean": round(praat_hnr(sound, intervals), 6),
        "spectral_tilt_mean": round(spectral_tilt(audio, sample_rate, mask), 6),
        "f0_mean": round(float(np.mean(voiced_f0)) if voiced_f0 else 0.0, 6),
        "voiced_intervals": len(intervals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract breathiness metrics from references and model outputs")
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
    feature_rows = []
    failures = []
    seen_references = set()

    for row in rows:
        reference_path = resolve_audio_path(base_dir, row["reference_path"])
        if reference_path and reference_path not in seen_references:
            seen_references.add(reference_path)
            try:
                features = extract_one(reference_path)
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

        output_path = resolve_audio_path(base_dir, row["output_path"])
        if output_path:
            try:
                features = extract_one(output_path)
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
        "sample_id",
        "pair_id",
        "condition",
        "type",
        "model",
        "path",
        "duration_sec",
        "voiced_seconds",
        "voiced_frame_ratio",
        "voiced_intervals",
        "cpp_mean",
        "hnr_mean",
        "spectral_tilt_mean",
        "f0_mean",
        "has_clipping",
        "too_little_voicing",
    ]
    write_csv(args.features_dir / "features.csv", feature_rows, fieldnames)
    print(f"Wrote {len(feature_rows)} rows to {args.features_dir / 'features.csv'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
