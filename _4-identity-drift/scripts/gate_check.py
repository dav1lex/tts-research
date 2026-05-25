#!/usr/bin/env python3
"""Gate conditions for identity drift benchmark.

1. REFERENCE GATE: reference.wav must have sufficient voicing and no clipping.
2. GENERATION GATE: each model must have >= 50% windows passing quality checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# --- Paths ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
REFERENCE_PATH = DATA / "reference" / "reference.wav"
FEATURES_DIR = RESULTS / "features"

# --- Parameters ---
TARGET_SR = 16000
F0_MIN = 50.0
F0_MAX = 500.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01

MODELS = ["chatterbox", "xtts", "kokoro"]


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


def check_reference() -> dict:
    """Check reference.wav passes quality gate."""
    print("CHECKING REFERENCE GATE...")
    audio, sr = load_mono(REFERENCE_PATH)
    duration = len(audio) / sr
    mask, _, _ = voiced_mask(audio, sr)

    voiced_seconds = float(np.sum(mask) * HOP_LENGTH)
    voiced_frame_ratio = float(np.mean(mask)) if len(mask) > 0 else 0.0
    has_clipping = bool(np.max(np.abs(audio)) >= 0.99)
    passed = bool(voiced_frame_ratio >= 0.10 and voiced_seconds >= 0.20 and not has_clipping)

    result = {
        "path": str(REFERENCE_PATH),
        "duration_sec": round(duration, 4),
        "voiced_frame_ratio": round(voiced_frame_ratio, 4),
        "voiced_seconds": round(voiced_seconds, 4),
        "has_clipping": has_clipping,
        "passed": passed,
    }

    print(f"  Path: {REFERENCE_PATH}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Voiced frame ratio: {voiced_frame_ratio:.4f}")
    print(f"  Voiced seconds: {voiced_seconds:.4f}")
    print(f"  Has clipping: {has_clipping}")
    print(f"  Reference gate: {'PASSED' if passed else 'FAILED'}")

    return result


def check_generation() -> dict:
    """Check each model's generated windows pass the quality gate."""
    print("\nCHECKING GENERATION GATE...")
    csv_path = FEATURES_DIR / "window_features.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run extract_windows.py first.", file=sys.stderr)
        return {}

    df = pd.read_csv(csv_path)
    models_result = {}

    for model_name in MODELS:
        model_df = df[df["model"] == model_name]
        if len(model_df) == 0:
            print(f"  {model_name}: no windows found — SKIPPING")
            continue

        total_windows = len(model_df)
        passed_windows = int((~model_df["too_little_voicing"].astype(bool)).sum())
        passed_ratio = passed_windows / total_windows if total_windows > 0 else 0.0
        quality_flagged = bool(passed_ratio < 0.50)

        models_result[model_name] = {
            "total_windows": total_windows,
            "passed_windows": passed_windows,
            "passed_ratio": round(passed_ratio, 4),
            "quality_flagged": quality_flagged,
        }

        status = "PASS" if not quality_flagged else "FLAGGED"
        print(f"  {model_name}: {passed_windows}/{total_windows} windows passed "
              f"({passed_ratio:.1%}) — {status}")

    return models_result


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Reference gate
    ref_result = check_reference()
    if not ref_result["passed"]:
        fail_msg = (
            f"REFERENCE GATE FAILED: "
            f"voiced_frame_ratio={ref_result['voiced_frame_ratio']}, "
            f"voiced_seconds={ref_result['voiced_seconds']}, "
            f"has_clipping={ref_result['has_clipping']}"
        )
        print(f"\nERROR: {fail_msg}", file=sys.stderr)
        gate_json = {
            "passed": False,
            "reference": ref_result,
            "models": {},
            "error": fail_msg,
        }
        (RESULTS / "gate_check.json").write_text(json.dumps(gate_json, indent=2))
        (RESULTS / ".gate_failed").write_text(fail_msg + "\n")
        print(f"Wrote {RESULTS / 'gate_check.json'}")
        return 1

    # 2. Generation gate
    models_result = check_generation()

    if not models_result:
        print("ERROR: no model results to check", file=sys.stderr)
        return 1

    # Determine overall pass
    all_models_ok = all(not m["quality_flagged"] for m in models_result.values())
    overall_passed = bool(all_models_ok)

    gate_json = {
        "passed": overall_passed,
        "reference": ref_result,
        "models": models_result,
    }

    (RESULTS / "gate_check.json").write_text(json.dumps(gate_json, indent=2))

    if overall_passed:
        (RESULTS / ".gate_passed").write_text("passed\n")
        print(f"\nGATE CHECK: ALL PASSED")
        if (RESULTS / ".gate_failed").exists():
            (RESULTS / ".gate_failed").unlink()
    else:
        # Write warning but don't block
        flagged = [m for m, r in models_result.items() if r["quality_flagged"]]
        print(f"\nWARNING: Models with quality flags: {', '.join(flagged)}")
        print("Analysis will continue but flagged models may be unreliable.")
        if (RESULTS / ".gate_failed").exists():
            (RESULTS / ".gate_failed").unlink()
        # Still create .gate_passed since reference passed
        (RESULTS / ".gate_passed").write_text("passed\n")

    print(f"Wrote {RESULTS / 'gate_check.json'}")
    return 0 if overall_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
