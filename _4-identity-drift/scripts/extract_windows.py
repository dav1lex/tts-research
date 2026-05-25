#!/usr/bin/env python3
"""Slice long-form audio into 15-second windows and extract acoustic features.

Key differences from Paper 2:
- F0_MIN=50, F0_MAX=500 (wider range for long windows)
- Non-overlapping windows
- Per-window quality flags
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from parselmouth.praat import call

# --- Paths ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FEATURES_DIR = RESULTS / "features"
LONGFORM_DIR = DATA / "long_form"
REFERENCE_PATH = DATA / "reference" / "reference.wav"

# --- Parameters ---
WINDOW_SIZE = 15.0
HOP_SIZE = 15.0
TARGET_SR = 16000
F0_MIN = 50.0
F0_MAX = 500.0
F0_MIN_PRAAT = 60.0
F0_MAX_PRAAT = 400.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01

MODELS = ["chatterbox", "xtts", "kokoro"]


def load_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("empty audio")
    if sample_rate != target_sr:
        audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    return audio.astype(float), sample_rate


def voiced_mask(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy of Paper 2 voiced_mask with adjusted F0 range."""
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
    if np.mean(mask) < 0.10:
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


def voiced_intervals(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Copy of Paper 2 voiced_intervals."""
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


def praat_cpp(sound: parselmouth.Sound, intervals: List[Tuple[float, float]]) -> float:
    """Copy of Paper 2 praat_cpp — identical parameters."""
    values = []
    for start, end in intervals:
        if end - start < 0.10:
            continue
        part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
        try:
            power_cepstrogram = call(part, "To PowerCepstrogram", max(F0_MIN_PRAAT, 75.0), HOP_LENGTH, 5000, 50)
            cpp = call(
                power_cepstrogram, "Get CPPS",
                True, 0.02, 0.0005,
                max(F0_MIN_PRAAT, 75.0), 330, 0.05,
                "Parabolic", 0.001, 0.0,
                "Exponential decay", "Robust",
            )
        except Exception:
            continue
        if math.isfinite(cpp):
            values.append(float(cpp))
    if not values:
        raise ValueError("Praat CPPS failed on voiced intervals")
    return float(np.mean(values))


def extract_window_features(
    audio: np.ndarray, sample_rate: int, start_sample: int, end_sample: int
) -> dict:
    """Extract all features from one window."""
    window_audio = audio[start_sample:end_sample]
    if len(window_audio) == 0:
        raise ValueError("empty window")

    # Voiced mask for this window
    mask, f0, rms = voiced_mask(window_audio, sample_rate)

    # F0 features (voiced frames only)
    voiced_f0 = f0[mask]
    f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else float("nan")
    f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else float("nan")

    # CPP via Praat
    intervals = voiced_intervals(mask)
    sounds = parselmouth.Sound(window_audio, sampling_frequency=sample_rate)
    try:
        cpp_mean = praat_cpp(sounds, intervals)
    except ValueError:
        cpp_mean = float("nan")

    # Spectral flatness
    try:
        spec_flat = librosa.feature.spectral_flatness(y=window_audio)
        spectral_flatness = float(np.mean(spec_flat))
    except Exception:
        spectral_flatness = float("nan")

    # Spectral tilt ratio (low <1000Hz / high >=1000Hz)
    try:
        S = np.abs(librosa.stft(window_audio, n_fft=1024, hop_length=160))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        low_energy = np.sum(S[freqs < 1000])
        high_energy = np.sum(S[freqs >= 1000])
        spectral_tilt_ratio = float(low_energy / high_energy) if high_energy > 1e-8 else float("nan")
    except Exception:
        spectral_tilt_ratio = float("nan")

    # MFCC (coefficients 1-12, skipping energy/coeff 0)
    try:
        mfcc = librosa.feature.mfcc(y=window_audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc[1:13]))  # Skip coefficient 0 (energy)
    except Exception:
        mfcc_mean = float("nan")

    # RMS
    try:
        rms_feat = librosa.feature.rms(y=window_audio)
        rms_mean = float(np.mean(rms_feat))
    except Exception:
        rms_mean = float("nan")

    # Spectral centroid
    try:
        centroid = librosa.feature.spectral_centroid(y=window_audio, sr=sample_rate)
        spectral_centroid = float(np.mean(centroid))
    except Exception:
        spectral_centroid = float("nan")

    # Quality flags
    voiced_frame_ratio = float(np.mean(mask)) if len(mask) > 0 else 0.0
    voiced_seconds = float(np.sum(mask) * HOP_LENGTH)
    too_little_voicing = bool(voiced_frame_ratio < 0.10 or voiced_seconds < 0.20)
    has_clipping = bool(np.max(np.abs(window_audio)) >= 0.99)

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "cpp_mean": cpp_mean,
        "spectral_flatness": spectral_flatness,
        "spectral_tilt_ratio": spectral_tilt_ratio,
        "mfcc_mean": mfcc_mean,
        "rms_mean": rms_mean,
        "spectral_centroid": spectral_centroid,
        "voiced_frame_ratio": voiced_frame_ratio,
        "too_little_voicing": too_little_voicing,
        "has_clipping": has_clipping,
    }


def main() -> int:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failed_windows = []

    for model_name in MODELS:
        audio_path = LONGFORM_DIR / model_name / "full.wav"
        if not audio_path.exists():
            print(f"WARNING: {audio_path} not found — skipping {model_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {model_name}")
        print(f"{'=' * 60}")
        audio, sr = load_mono(audio_path)

        window_samples = int(WINDOW_SIZE * sr)
        hop_samples = int(HOP_SIZE * sr)
        total_duration = len(audio) / sr
        num_windows = (len(audio) - window_samples) // hop_samples + 1

        print(f"  Audio: {total_duration:.1f}s, {sr}Hz, {num_windows} windows")

        for win_idx in range(num_windows):
            start_sample = win_idx * hop_samples
            end_sample = start_sample + window_samples
            window_audio = audio[start_sample:end_sample]

            time_start = round(start_sample / sr, 3)
            time_end = round(end_sample / sr, 3)

            try:
                feats = extract_window_features(audio, sr, start_sample, end_sample)
            except Exception as e:
                print(f"  WARN: window {win_idx} [{time_start}s-{time_end}s] failed: {e}")
                continue

            row = {
                "model": model_name,
                "window_idx": win_idx,
                "time_start": time_start,
                "time_end": time_end,
                **feats,
            }
            all_rows.append(row)

            if feats["too_little_voicing"] or feats["has_clipping"]:
                failed_windows.append(
                    f"{model_name},{win_idx},{time_start},{time_end},"
                    f"too_little_voicing={feats['too_little_voicing']},"
                    f"has_clipping={feats['has_clipping']}"
                )

            if (win_idx + 1) % 4 == 0 or win_idx == num_windows - 1:
                print(f"  [{win_idx + 1}/{num_windows}] window {win_idx}: "
                      f"f0={feats['f0_mean']:.1f}Hz, "
                      f"voiced_ratio={feats['voiced_frame_ratio']:.3f}, "
                      f"clipping={feats['has_clipping']}")

    # Save window features CSV
    df = pd.DataFrame(all_rows)
    cols = [
        "model", "window_idx", "time_start", "time_end",
        "f0_mean", "f0_std", "cpp_mean", "spectral_flatness",
        "spectral_tilt_ratio", "mfcc_mean", "rms_mean", "spectral_centroid",
        "voiced_frame_ratio", "too_little_voicing", "has_clipping",
    ]
    df = df[cols]
    csv_path = FEATURES_DIR / "window_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} windows to {csv_path}")

    # Save failed windows report
    if failed_windows:
        failed_path = FEATURES_DIR / "failed_windows.txt"
        failed_path.write_text("model,window_idx,time_start,time_end,flags\n" + "\n".join(failed_windows))
        print(f"Wrote {len(failed_windows)} failed windows to {failed_path}")

    # Per-model summary
    for model_name in MODELS:
        model_df = df[df["model"] == model_name]
        if len(model_df) == 0:
            continue
        flagged = model_df[model_df["too_little_voicing"] | model_df["has_clipping"]]
        print(f"\n  {model_name}: {len(model_df)} windows, {len(flagged)} flagged")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
