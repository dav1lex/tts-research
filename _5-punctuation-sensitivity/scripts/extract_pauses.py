#!/usr/bin/env python3
"""Extract pause metrics and F0 contours from punctuation test audio."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch

from common import (
    CONFIG,
    CORPUS,
    FEATURES_DIR,
    FEATURES_CSV,
    OUTPUTS,
    TEXT_NORM_LOG,
    safe_float,
)

ENERGY_THRESHOLD_FACTOR = CONFIG["vad"]["energy_threshold_factor"]
MIN_PAUSE_MS = CONFIG["vad"]["min_pause_ms"]
FRAME_MS = CONFIG["vad"]["frame_ms"]
HOP_MS = CONFIG["vad"]["hop_ms"]
F0_FMIN = CONFIG["f0"]["fmin"]
F0_FMAX = CONFIG["f0"]["fmax"]
F0_FRAME_LENGTH = CONFIG["f0"]["frame_length"]
F0_WIN_LENGTH = CONFIG["f0"]["win_length"]
F0_HOP_LENGTH = CONFIG["f0"]["hop_length"]


def load_audio(path: Path):
    """Load audio, handling both WAV and PyTorch .pt files."""
    try:
        y, sr = sf.read(str(path))
        return y, sr
    except Exception:
        data = torch.load(str(path), weights_only=True)
        y = data.numpy()
        sr = 24000
        return y, sr


def rms_energy(y: np.ndarray, sr: int):
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    rms = np.array([
        np.sqrt(np.mean(y[i:i+frame_len]**2))
        for i in range(0, len(y) - frame_len, hop_len)
    ])
    return rms, hop_len


def find_pauses(rms: np.ndarray, hop_len: int, sr: int):
    threshold = np.max(rms) * ENERGY_THRESHOLD_FACTOR
    is_silence = rms < threshold
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
                    "start_ms": round(pause_start * hop_len / sr * 1000, 1),
                    "duration_ms": round(duration_ms, 1),
                })
    if in_pause:
        duration_ms = (len(rms) - pause_start) * hop_len / sr * 1000
        if duration_ms >= MIN_PAUSE_MS:
            pauses.append({
                "start_ms": round(pause_start * hop_len / sr * 1000, 1),
                "duration_ms": round(duration_ms, 1),
            })
    return pauses


def f0_slope(y: np.ndarray, sr: int, start_sample: int, end_sample: int):
    segment = y[start_sample:end_sample]
    if len(segment) < sr * 0.05:
        return None, 0
    f0, voiced_flag, _ = librosa.pyin(
        segment,
        fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
        frame_length=F0_FRAME_LENGTH, win_length=F0_WIN_LENGTH, hop_length=F0_HOP_LENGTH,
    )
    voiced = f0[voiced_flag > 0.5]
    if len(voiced) < 3:
        return None, 0
    t = np.arange(len(voiced)) * F0_HOP_LENGTH / sr
    slope, _ = np.polyfit(t, voiced, 1)
    return round(float(slope), 3), len(voiced)


def f0_mean_range(y: np.ndarray, sr: int):
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
        frame_length=F0_FRAME_LENGTH, win_length=F0_WIN_LENGTH, hop_length=F0_HOP_LENGTH,
    )
    voiced = f0[voiced_flag > 0.5]
    if len(voiced) < 3:
        return None, None, 0
    return (
        round(float(np.mean(voiced)), 1),
        round(float(np.max(voiced) - np.min(voiced)), 1),
        len(voiced),
    )


def rms_mean(y: np.ndarray):
    return round(float(np.sqrt(np.mean(y**2))), 6)


def amplitude_decay(y: np.ndarray, sr: int, last_ms: int = 300):
    n_samples = int(sr * last_ms / 1000)
    if len(y) < n_samples:
        return None
    segment = y[-n_samples:]
    rms, hop = rms_energy(segment, sr)
    if len(rms) < 3:
        return None
    t = np.arange(len(rms)) * hop / sr * 1000
    slope, _ = np.polyfit(t, rms, 1)
    return round(float(slope), 10)


def main():
    with open(CORPUS) as f:
        rows = list(csv.DictReader(f))

    models = ["chatterbox", "kokoro", "xtts"]
    all_features = []
    text_norm_log = {}  # id -> {expected_text, model_text_sent}

    for model in models:
        model_dir = OUTPUTS / model
        if not model_dir.exists():
            continue

        print(f"\n--- MODEL: {model} ---")
        for i, row in enumerate(rows):
            wav_path = model_dir / f"{row['id']}.wav"
            if not wav_path.exists():
                continue

            # Log what text we're sending vs what's in the corpus
            expected_text = row["text"]
            text_norm_log[f"{model}/{row['id']}"] = {
                "model": model,
                "id": row["id"],
                "expected_text": expected_text,
                "text_sent_to_model": expected_text,  # No normalization applied by us
            }

            y, sr = load_audio(wav_path)
            if y.ndim > 1:
                y = y.mean(axis=1)

            rms, hop_len = rms_energy(y, sr)
            pauses = find_pauses(rms, hop_len, sr)
            total_duration_ms = len(y) / sr * 1000

            # Find last speech sample
            is_silence = rms < (np.max(rms) * ENERGY_THRESHOLD_FACTOR)
            last_speech_end = len(y)
            for j in range(len(is_silence) - 1, -1, -1):
                if not is_silence[j]:
                    last_speech_end = int((j + 1) * hop_len)
                    break

            terminal_window_ms = CONFIG["analysis"]["terminal_window_ms"]
            terminal_start = max(0, last_speech_end - int(sr * terminal_window_ms / 1000))
            terminal_f0_slope, terminal_voiced = f0_slope(y, sr, terminal_start, last_speech_end)
            f0_mean_val, f0_range_val, f0_voiced = f0_mean_range(y, sr)
            amp_decay = amplitude_decay(y, sr, last_ms=300)

            best_pause_ms = max([p["duration_ms"] for p in pauses]) if pauses else 0.0
            all_pause_ms = [p["duration_ms"] for p in pauses]
            num_pauses = len(pauses)

            internal_cutoff = CONFIG["analysis"]["internal_cutoff_fraction"]
            internal_pauses = [
                p for p in pauses
                if p["start_ms"] + p["duration_ms"] < total_duration_ms * internal_cutoff
            ]
            internal_durations = [p["duration_ms"] for p in internal_pauses]

            feat = {
                "model": model,
                "id": row["id"],
                "category": row["category"],
                "subcategory": row.get("subcategory", ""),
                "punct_type": row["punct_type"],
                "duration_s": round(total_duration_ms / 1000, 3),
                "rms_mean": rms_mean(y),
                "f0_mean": f0_mean_val,
                "f0_range": f0_range_val,
                "f0_voiced_frames": f0_voiced,
                "terminal_f0_slope": terminal_f0_slope,
                "terminal_voiced": terminal_voiced,
                "amplitude_decay_300ms": amp_decay,
                "best_pause_ms": best_pause_ms,
                "num_pauses": num_pauses,
                "pause_durations_ms": json.dumps(all_pause_ms),
                "internal_pause_durations_ms": json.dumps(internal_durations),
            }
            all_features.append(feat)

            print(
                f"  [{i+1}/{len(rows)}] {row['id']}: "
                f"pauses={num_pauses}, best={best_pause_ms}ms, "
                f"f0_slope={terminal_f0_slope}"
            )

    # Write features CSV
    fieldnames = [
        "model", "id", "category", "subcategory", "punct_type",
        "duration_s", "rms_mean", "f0_mean", "f0_range",
        "f0_voiced_frames", "terminal_f0_slope", "terminal_voiced",
        "amplitude_decay_300ms", "best_pause_ms", "num_pauses",
        "pause_durations_ms", "internal_pause_durations_ms",
    ]
    with open(FEATURES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
    print(f"\nSaved {len(all_features)} rows to {FEATURES_CSV}")

    # Write text normalization log
    with open(TEXT_NORM_LOG, "w") as f:
        json.dump(text_norm_log, f, indent=2)
    print(f"Saved text normalization log to {TEXT_NORM_LOG}")

    return 0


if __name__ == "__main__":
    sys.exit(main())