#!/usr/bin/env python3
"""Extract acoustic features from target-only segments.

Loads full WAV (prime + target), finds the boundary pause between prime
and target using RMS-based VAD, then extracts f0_mean, f0_std, energy_std
from the target segment only.
"""
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

TEXTS_PATH = PROJECT / "prompts" / "texts.json"
OUTPUTS_DIR = PROJECT / "outputs"
FEATURES_DIR = PROJECT / "features"
FEATURES_CSV = FEATURES_DIR / "features.csv"

# VAD parameters (matching _5-punctuation-sensitivity style)
ENERGY_THRESHOLD_FACTOR = 0.05
MIN_PAUSE_MS = 150       # only consider pauses this long as boundaries
FRAME_MS = 20
HOP_MS = 10

# F0 extraction
F0_MIN = 50
F0_MAX = 600


def rms_energy(y: np.ndarray, sr: int):
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    rms = np.array([
        np.sqrt(np.mean(y[i:i+frame_len]**2))
        for i in range(0, len(y) - frame_len, hop_len)
    ])
    return rms, hop_len


def find_boundary_pause(y: np.ndarray, sr: int) -> int:
    """Find the sample index of the last major pause (prime-target boundary).

    Returns the sample index where the target segment starts (after the
    last pause that's >= MIN_PAUSE_MS and starts before the final 30% of audio).
    Falls back to splitting at the 40% duration mark if no VAD pause found.
    """
    rms, hop_len = rms_energy(y, sr)
    threshold = np.max(rms) * ENERGY_THRESHOLD_FACTOR
    is_silence = rms < threshold

    # Find all pauses
    pauses = []
    in_pause = False
    pause_start = 0
    for i, silent in enumerate(is_silence):
        if silent and not in_pause:
            in_pause = True
            pause_start = i
        elif not silent and in_pause:
            in_pause = False
            duration_ms = (i - pause_start) * hop_len / sr * 1000
            if duration_ms >= MIN_PAUSE_MS:
                pauses.append({
                    "start_sample": pause_start * hop_len,
                    "end_sample": i * hop_len,
                    "duration_ms": round(duration_ms, 1),
                })
    if in_pause:
        duration_ms = (len(rms) - pause_start) * hop_len / sr * 1000
        if duration_ms >= MIN_PAUSE_MS:
            pauses.append({
                "start_sample": pause_start * hop_len,
                "end_sample": len(y),
                "duration_ms": round(duration_ms, 1),
            })

    # Target is always last. We want the LAST major pause that starts in
    # the first 70% of audio (to avoid picking up final trailing silence).
    total_len = len(y)
    boundary_candidates = [
        p for p in pauses
        if p["start_sample"] < total_len * 0.70
    ]

    if boundary_candidates:
        # Pick the last such pause's end as the target start
        boundary = max(boundary_candidates, key=lambda p: p["end_sample"])
        target_start = boundary["end_sample"]
        print(f"  VAD boundary pause: {boundary['duration_ms']}ms at "
              f"{boundary['start_sample']/sr:.2f}s -> target at {target_start/sr:.2f}s")
        return target_start

    # Fallback: assume prime is first 40% of audio
    fallback = int(total_len * 0.40)
    print(f"  WARN: no VAD boundary found, using fallback split at 40%={fallback/sr:.2f}s")
    return fallback


def extract_features_segment(y_seg: np.ndarray, sr: int):
    """Extract f0_mean, f0_std, energy_std from a target audio segment."""
    if len(y_seg) == 0:
        return {"f0_mean": 0.0, "f0_std": 0.0, "energy_std": 0.0}

    # F0 via parselmouth (Praat)
    snd = parselmouth.Sound(y_seg, sampling_frequency=sr)
    pitch = snd.to_pitch(
        time_step=0.01,
        pitch_floor=F0_MIN,
        pitch_ceiling=F0_MAX,
    )
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]  # remove unvoiced (0)

    if len(f0_values) == 0:
        f0_mean = 0.0
        f0_std = 0.0
    else:
        f0_mean = float(np.mean(f0_values))
        f0_std = float(np.std(f0_values))

    # RMS energy std (dynamic range proxy)
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    rms_frames = np.array([
        np.sqrt(np.mean(y_seg[i:i+frame_len]**2))
        for i in range(0, len(y_seg) - frame_len, hop_len)
    ])
    energy_std = float(np.std(rms_frames)) if len(rms_frames) > 1 else 0.0

    return {
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
        "energy_std": round(energy_std, 6),
    }


def parse_name_to_meta(filename: str):
    """Parse 'chatterbox_control_run1.wav' -> (model, condition, run)."""
    stem = Path(filename).stem  # without extension
    parts = stem.split("_")
    if len(parts) == 4:
        return parts[0], parts[1], int(parts[3])
    raise ValueError(f"Cannot parse filename: {filename}")


def main():
    with open(TEXTS_PATH) as f:
        texts = json.load(f)

    models = texts["model_order"]
    conditions = texts["conditions"]
    n_reps = texts["n_reps"]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    all_features = []

    for model in models:
        model_dir = OUTPUTS_DIR / model
        if not model_dir.exists():
            print(f"SKIP {model}: no output dir")
            continue

        print(f"\n=== MODEL: {model} ===")
        for condition in conditions:
            for run in range(1, n_reps + 1):
                filename = f"{model}_{condition}_run{run}.wav"
                wav_path = model_dir / filename
                if not wav_path.exists():
                    print(f"  MISSING: {filename}")
                    continue

                y, sr = sf.read(str(wav_path))
                if y.ndim > 1:
                    y = y.mean(axis=1)

                # Find prime-target boundary and extract target segment
                target_start = find_boundary_pause(y, sr)
                y_target = y[target_start:]

                if len(y_target) < sr * 0.1:  # less than 100ms
                    print(f"  WARN: {filename} target too short ({len(y_target)/sr:.2f}s), skipping")
                    continue

                feats = extract_features_segment(y_target, sr)
                feats["model"] = model
                feats["condition"] = condition
                feats["run"] = run
                feats["filename"] = filename
                feats["target_duration_s"] = round(len(y_target) / sr, 3)

                all_features.append(feats)
                print(f"  [{condition} run{run}] f0_mean={feats['f0_mean']}, "
                      f"f0_std={feats['f0_std']}, energy_std={feats['energy_std']}, "
                      f"dur={feats['target_duration_s']}s")

    # Write CSV
    fieldnames = ["model", "condition", "run", "filename",
                  "f0_mean", "f0_std", "energy_std", "target_duration_s"]
    with open(FEATURES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)

    print(f"\nSaved {len(all_features)} rows to {FEATURES_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
