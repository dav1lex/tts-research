#!/usr/bin/env python3
"""Extract pause metrics and F0 contours from punctuation test audio."""
import csv, sys, json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch

PROJECT = Path("/home/davilex/tts-research/_5-punctuation-sensitivity")
CORPUS = PROJECT / "data" / "test_corpus.csv"
OUTPUTS = PROJECT / "outputs"
FEATURES_DIR = PROJECT / "results" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_THRESHOLD_FACTOR = 0.05
MIN_PAUSE_MS = 30
FRAME_MS = 20
HOP_MS = 10


def load_audio(path):
    """Load audio, handling both WAV and PyTorch .pt files."""
    try:
        y, sr = sf.read(str(path))
        return y, sr
    except Exception:
        data = torch.load(str(path), weights_only=True)
        y = data.numpy()
        sr = 24000
        return y, sr


def rms_energy(y, sr):
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    rms = np.array([
        np.sqrt(np.mean(y[i:i+frame_len]**2))
        for i in range(0, len(y) - frame_len, hop_len)
    ])
    return rms, hop_len


def find_pauses(rms, hop_len, sr):
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


def f0_slope(y, sr, start_sample, end_sample):
    segment = y[start_sample:end_sample]
    if len(segment) < sr * 0.05:
        return None, 0
    f0, voiced_flag, _ = librosa.pyin(
        segment, fmin=50, fmax=600, sr=sr,
        frame_length=2048, win_length=1024, hop_length=256
    )
    voiced = f0[voiced_flag > 0.5]
    if len(voiced) < 3:
        return None, 0
    t = np.arange(len(voiced)) * 256 / sr
    slope, _ = np.polyfit(t, voiced, 1)
    return round(float(slope), 3), len(voiced)


def f0_mean_range(y, sr):
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=600, sr=sr,
        frame_length=2048, win_length=1024, hop_length=256
    )
    voiced = f0[voiced_flag > 0.5]
    if len(voiced) < 3:
        return None, None, 0
    return round(float(np.mean(voiced)), 1), round(float(np.max(voiced) - np.min(voiced)), 1), len(voiced)


def rms_mean(y):
    return round(float(np.sqrt(np.mean(y**2))), 6)


def amplitude_decay(y, sr, last_ms=300):
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

    for model in models:
        model_dir = OUTPUTS / model
        if not model_dir.exists():
            continue

        print(f"\n--- MODEL: {model} ---")
        for i, row in enumerate(rows):
            wav_path = model_dir / f"{row['id']}.wav"
            if not wav_path.exists():
                continue

            y, sr = load_audio(wav_path)
            if y.ndim > 1:
                y = y.mean(axis=1)

            rms, hop_len = rms_energy(y, sr)
            pauses = find_pauses(rms, hop_len, sr)
            total_duration_ms = len(y) / sr * 1000

            is_silence = rms < (np.max(rms) * ENERGY_THRESHOLD_FACTOR)
            last_speech_end = len(y)
            for j in range(len(is_silence) - 1, -1, -1):
                if not is_silence[j]:
                    last_speech_end = int((j + 1) * hop_len)
                    break

            terminal_start = max(0, last_speech_end - int(sr * 0.4))
            terminal_f0_slope, terminal_voiced = f0_slope(y, sr, terminal_start, last_speech_end)
            f0_mean, f0_range, f0_voiced = f0_mean_range(y, sr)
            amp_decay = amplitude_decay(y, sr, last_ms=300)

            best_pause_ms = max([p["duration_ms"] for p in pauses]) if pauses else 0.0
            all_pause_ms = [p["duration_ms"] for p in pauses]
            num_pauses = len(pauses)

            internal_pauses = [p for p in pauses if p["start_ms"] + p["duration_ms"] < total_duration_ms * 0.95]
            internal_durations = [p["duration_ms"] for p in internal_pauses]

            feat = {
                "model": model,
                "id": row["id"],
                "category": row["category"],
                "subcategory": row.get("subcategory", ""),
                "punct_type": row["punct_type"],
                "duration_s": round(total_duration_ms / 1000, 3),
                "rms_mean": rms_mean(y),
                "f0_mean": f0_mean,
                "f0_range": f0_range,
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

            print(f"  [{i+1}/{len(rows)}] {row['id']}: "
                  f"pauses={num_pauses}, best={best_pause_ms}ms, "
                  f"f0_slope={terminal_f0_slope}")

    out_path = FEATURES_DIR / "pause_features.csv"
    fieldnames = [
        "model", "id", "category", "subcategory", "punct_type",
        "duration_s", "rms_mean", "f0_mean", "f0_range",
        "f0_voiced_frames", "terminal_f0_slope", "terminal_voiced",
        "amplitude_decay_300ms", "best_pause_ms", "num_pauses",
        "pause_durations_ms", "internal_pause_durations_ms",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
    print(f"\nSaved {len(all_features)} rows to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
