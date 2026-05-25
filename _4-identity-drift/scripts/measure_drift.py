#!/usr/bin/env python3
"""Measure per-window drift from reference across the 5-minute monologue.

Normalization: IQR-based robust_scale from all valid windows (pooled) per feature.
Kokoro baseline: raw feature variance computed separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats as scipy_stats

# --- Paths ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FEATURES_DIR = RESULTS / "features"
REFERENCE_PATH = DATA / "reference" / "reference.wav"

# --- Parameters ---
TARGET_SR = 16000
F0_MIN = 50.0
F0_MAX = 500.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01
F0_MIN_PRAAT = 60.0
F0_MAX_PRAAT = 400.0

FEATURE_COLS = [
    "f0_mean", "f0_std", "cpp_mean", "spectral_flatness",
    "spectral_tilt_ratio", "mfcc_mean", "rms_mean", "spectral_centroid",
]

MODELS = ["chatterbox", "xtts", "kokoro"]


def robust_scale(values: list[float]) -> float:
    """IQR-based robust scale (copy from Paper 2/3)."""
    if len(values) < 2:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    std = float(np.std(values, ddof=1))
    scale = iqr / 1.349 if iqr > 0 else std
    return scale if scale > 1e-6 else 1.0


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
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_threshold = max(float(np.percentile(rms, 20)) * 0.5, float(np.max(rms)) * 0.01, 1e-8)

    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio, fmin=F0_MIN, fmax=F0_MAX, sr=sample_rate,
        frame_length=frame_samples, hop_length=hop_samples, center=False,
    )
    frame_count = min(len(rms), len(voiced_flag), len(voiced_probability))
    mask = (
        (rms[:frame_count] >= energy_threshold)
        & voiced_flag[:frame_count].astype(bool)
        & (voiced_probability[:frame_count] >= 0.50)
    )
    if np.mean(mask) < 0.10:
        fallback_f0 = librosa.yin(
            audio, fmin=F0_MIN, fmax=F0_MAX, sr=sample_rate,
            frame_length=frame_samples, hop_length=hop_samples, center=False,
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


def extract_reference_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Extract the same feature set from the reference audio."""
    window_audio = audio  # entire reference clip is short
    mask, f0, rms = voiced_mask(window_audio, sr)

    # F0
    voiced_f0 = f0[mask]
    f0_mean_val = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else float("nan")
    f0_std_val = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else float("nan")

    # CPP via Praat
    import parselmouth
    from parselmouth.praat import call
    import math

    def _praat_cpp(sound, intervals):
        values = []
        for start, end in intervals:
            if end - start < 0.10:
                continue
            part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
            try:
                power_cepstrogram = call(part, "To PowerCepstrogram", max(F0_MIN_PRAAT, 75.0), HOP_LENGTH, 5000, 50)
                cpp_val = call(
                    power_cepstrogram, "Get CPPS",
                    True, 0.02, 0.0005,
                    max(F0_MIN_PRAAT, 75.0), 330, 0.05,
                    "Parabolic", 0.001, 0.0,
                    "Exponential decay", "Robust",
                )
            except Exception:
                continue
            if math.isfinite(cpp_val):
                values.append(float(cpp_val))
        if not values:
            raise ValueError("Praat CPPS failed on voiced intervals")
        return float(np.mean(values))

    # Voiced intervals
    intervals = []
    start = None
    for idx, v in enumerate(mask):
        if v and start is None:
            start = idx
        elif not v and start is not None:
            intervals.append((start * HOP_LENGTH, idx * HOP_LENGTH + FRAME_LENGTH))
            start = None
    if start is not None:
        intervals.append((start * HOP_LENGTH, len(mask) * HOP_LENGTH + FRAME_LENGTH))

    sounds = parselmouth.Sound(window_audio, sampling_frequency=sr)
    try:
        cpp_mean_val = _praat_cpp(sounds, intervals)
    except ValueError:
        cpp_mean_val = float("nan")

    # Spectral flatness
    spec_flat = librosa.feature.spectral_flatness(y=window_audio)
    spectral_flatness_val = float(np.mean(spec_flat))

    # Spectral tilt ratio
    S = np.abs(librosa.stft(window_audio, n_fft=1024, hop_length=160))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low_energy = np.sum(S[freqs < 1000])
    high_energy = np.sum(S[freqs >= 1000])
    spectral_tilt_ratio_val = float(low_energy / high_energy) if high_energy > 1e-8 else float("nan")

    # MFCC
    mfcc = librosa.feature.mfcc(y=window_audio, sr=sr, n_mfcc=13)
    mfcc_mean_val = float(np.mean(mfcc[1:13]))

    # RMS
    rms_feat = librosa.feature.rms(y=window_audio)
    rms_mean_val = float(np.mean(rms_feat))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=window_audio, sr=sr)
    spectral_centroid_val = float(np.mean(centroid))

    return {
        "f0_mean": f0_mean_val,
        "f0_std": f0_std_val,
        "cpp_mean": cpp_mean_val,
        "spectral_flatness": spectral_flatness_val,
        "spectral_tilt_ratio": spectral_tilt_ratio_val,
        "mfcc_mean": mfcc_mean_val,
        "rms_mean": rms_mean_val,
        "spectral_centroid": spectral_centroid_val,
    }


def main() -> int:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Load window features
    csv_path = FEATURES_DIR / "window_features.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run extract_windows.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} windows from {csv_path}")

    # Extract reference features
    print(f"\nExtracting reference features from {REFERENCE_PATH}...")
    ref_audio, ref_sr = load_mono(REFERENCE_PATH)
    ref_features = extract_reference_features(ref_audio, ref_sr)
    ref_path = FEATURES_DIR / "reference_features.json"
    ref_path.write_text(json.dumps(ref_features, indent=2))
    print(f"  Reference features saved to {ref_path}")
    for col in FEATURE_COLS:
        print(f"    {col}: {ref_features.get(col, 'nan')}")

    # Filter out windows with too_little_voicing
    valid_mask = ~df["too_little_voicing"].astype(bool)
    valid_df = df[valid_mask].copy()
    total_windows = len(df)
    filtered = total_windows - len(valid_df)
    print(f"\nFiltered {filtered}/{total_windows} windows with too little voicing")

    for model_name in MODELS:
        model_total = int((df["model"] == model_name).sum())
        model_filtered = int(model_total - (valid_df["model"] == model_name).sum())
        print(f"  {model_name}: {model_filtered}/{model_total} filtered")

    if len(valid_df) == 0:
        print("ERROR: no valid windows remaining after filtering", file=sys.stderr)
        return 1

    # Compute robust_scale per feature from all valid windows (pooled)
    scales = {}
    for col in FEATURE_COLS:
        values = valid_df[col].dropna().tolist()
        scales[col] = robust_scale(values)
        print(f"  scale[{col}] = {scales[col]:.6f} (from {len(values)} values)")

    # Compute per-window drift
    drift_rows = []
    for _, row in valid_df.iterrows():
        row_dict = {
            "model": row["model"],
            "window_idx": int(row["window_idx"]),
            "time_start": float(row["time_start"]),
            "time_end": float(row["time_end"]),
        }

        per_feature_deltas = []
        for col in FEATURE_COLS:
            window_val = row[col]
            ref_val = ref_features.get(col, float("nan"))
            if pd.isna(window_val) or pd.isna(ref_val) or scales[col] <= 1e-6:
                row_dict[f"drift_{col}"] = float("nan")
            else:
                drift = abs(window_val - ref_val) / scales[col]
                row_dict[f"drift_{col}"] = float(drift)
                per_feature_deltas.append(drift)

        row_dict["drift_from_reference"] = (
            float(np.mean(per_feature_deltas)) if per_feature_deltas else float("nan")
        )
        drift_rows.append(row_dict)

    drift_df = pd.DataFrame(drift_rows)

    # Save per-window drift
    drift_cols = ["model", "window_idx", "time_start", "time_end", "drift_from_reference"]
    for col in FEATURE_COLS:
        drift_cols.append(f"drift_{col}")
    drift_df = drift_df[drift_cols]
    drift_path = RESULTS / "drift_by_window.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"\nSaved {len(drift_df)} drift rows to {drift_path}")

    # Compute per-model summaries
    summary_rows = []
    for model_name in MODELS:
        model_drift = drift_df[drift_df["model"] == model_name]
        if len(model_drift) == 0:
            continue

        n_windows = len(model_drift)
        n_filtered_val = int(
            (df["model"] == model_name).sum()
            - ((df["model"] == model_name) & valid_mask).sum()
        )

        drifts = model_drift["drift_from_reference"].dropna().values
        if len(drifts) == 0:
            continue

        drift_mean_val = float(np.mean(drifts))
        drift_std_val = float(np.std(drifts, ddof=1)) if len(drifts) > 1 else 0.0
        drift_min_val = float(np.min(drifts))
        drift_max_val = float(np.max(drifts))

        # Early vs late drift
        early_count = min(5, max(1, n_windows // 4))
        late_count = min(5, max(1, n_windows // 4))
        drift_early = float(np.mean(drifts[:early_count]))
        drift_late = float(np.mean(drifts[-late_count:]))
        drift_increase = drift_late - drift_early

        # Drift slope via linear regression
        if len(drifts) >= 3:
            x = np.arange(len(drifts))
            slope, _, _, _, _ = scipy_stats.linregress(x, drifts)
            drift_slope = float(slope)
        else:
            drift_slope = float("nan")

        # Kokoro raw variance (for baseline)
        if model_name == "kokoro":
            kokoro_model_df = valid_df[valid_df["model"] == "kokoro"]
            raw_variances = {}
            for col in FEATURE_COLS:
                col_vals = kokoro_model_df[col].dropna()
                raw_variances[col] = float(np.std(col_vals, ddof=1)) if len(col_vals) > 1 else 0.0
            kokoro_raw_variance = float(np.mean(list(raw_variances.values())))
        else:
            kokoro_raw_variance = float("nan")

        summary_rows.append({
            "model": model_name,
            "drift_mean": round(drift_mean_val, 6),
            "drift_std": round(drift_std_val, 6),
            "drift_min": round(drift_min_val, 6),
            "drift_max": round(drift_max_val, 6),
            "drift_early": round(drift_early, 6),
            "drift_late": round(drift_late, 6),
            "drift_increase": round(drift_increase, 6),
            "drift_slope": round(drift_slope, 10),
            "n_windows": n_windows,
            "n_filtered": n_filtered_val,
            "kokoro_raw_variance": round(kokoro_raw_variance, 6) if not np.isnan(kokoro_raw_variance) else float("nan"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS / "drift_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
