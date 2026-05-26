#!/usr/bin/env python3
"""Extract target-sentence prosodic features for the priming experiment.

Measures 8 features per audio clip:
  - F0 mean, std, range (parselmouth/Praat)
  - Speech rate (librosa + voiced-frame detection)
  - Pause count, pause duration mean (energy-based VAD)
  - RMS energy (librosa)
  - Spectral centroid (librosa)

Input is expected to be target-only WAVs from scripts/segment.py.

Output: results/features.csv with columns:
  model, condition, rep, file_path, f0_mean, f0_std, f0_range,
  speech_rate, pause_count, pause_duration_mean, rms_energy,
  spectral_centroid
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import CONDITIONS, MODELS, N_REPS, TARGET_SR, TARGET_WORD_COUNT  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────
F0_MIN = 60.0
F0_MAX = 400.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01
SILENCE_THRESHOLD = 0.02   # RMS energy threshold for silence
MIN_SILENCE_DUR = 0.15     # minimum silence to count as a pause (seconds)
MIN_VOICED_RATIO = 0.10    # minimum fraction of voiced frames for F0 extraction


# ── Audio loading ────────────────────────────────────────────────────

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


# ── Voiced frame detection ───────────────────────────────────────────

def voiced_frames(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return boolean mask of voiced frames using energy + F0 detection."""
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))

    energy_threshold = max(
        float(np.percentile(rms, 20)) * 0.5,
        float(np.max(rms)) * 0.01,
        1e-8,
    )

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sample_rate,
        frame_length=frame_samples,
        hop_length=hop_samples,
        center=False,
    )

    n = min(len(rms), len(voiced_flag), len(voiced_prob))
    mask = (
        (rms[:n] >= energy_threshold)
        & voiced_flag[:n].astype(bool)
        & (voiced_prob[:n] >= 0.50)
    )

    if np.mean(mask) < MIN_VOICED_RATIO:
        # Fallback: use YIN for more permissive F0 detection
        fallback_f0 = librosa.yin(
            audio,
            fmin=F0_MIN,
            fmax=F0_MAX,
            sr=sample_rate,
            frame_length=frame_samples,
            hop_length=hop_samples,
            center=False,
        )
        n2 = min(len(rms), len(fallback_f0))
        mask2 = (
            (rms[:n2] >= energy_threshold)
            & (fallback_f0[:n2] >= F0_MIN)
            & (fallback_f0[:n2] <= F0_MAX)
        )
        if np.mean(mask2) > np.mean(mask):
            return mask2

    return mask


# ── F0 extraction via Praat ──────────────────────────────────────────

def extract_f0(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Extract F0 mean, std, range using parselmouth (Praat)."""
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(
        time_step=HOP_LENGTH,
        pitch_floor=F0_MIN,
        pitch_ceiling=F0_MAX,
    )
    values = pitch.selected_array["frequency"]
    values = values[values > 0]

    if len(values) == 0:
        return {"f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0}

    return {
        "f0_mean": round(float(np.mean(values)), 4),
        "f0_std": round(float(np.std(values, ddof=1)), 4),
        "f0_range": round(float(np.max(values) - np.min(values)), 4),
    }


# ── Pause detection ──────────────────────────────────────────────────

def detect_pauses(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Detect silence segments using RMS energy threshold."""
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))

    # Absolute energy threshold: silence = below this RMS
    is_silent = rms < SILENCE_THRESHOLD

    # Find contiguous silence segments
    pauses: list[float] = []
    in_pause = False
    pause_start = 0
    for i, silent in enumerate(is_silent):
        if silent and not in_pause:
            in_pause = True
            pause_start = i * HOP_LENGTH
        elif not silent and in_pause:
            in_pause = False
            dur = i * HOP_LENGTH - pause_start
            if dur >= MIN_SILENCE_DUR:
                pauses.append(dur)
    if in_pause:
        dur = len(is_silent) * HOP_LENGTH - pause_start
        if dur >= MIN_SILENCE_DUR:
            pauses.append(dur)

    if not pauses:
        return {"pause_count": 0, "pause_duration_mean": 0.0}

    return {
        "pause_count": len(pauses),
        "pause_duration_mean": round(float(np.mean(pauses)), 4),
    }


# ── Speech rate ───────────────────────────────────────────────────────

def compute_speech_rate(mask: np.ndarray) -> float:
    """Words per second in voiced segments only."""
    voiced_duration = np.sum(mask) * HOP_LENGTH
    if voiced_duration < 0.05:
        return 0.0

    return round(TARGET_WORD_COUNT / voiced_duration, 4)


# ── Spectral features ────────────────────────────────────────────────

def extract_spectral(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Extract RMS energy and spectral centroid."""
    rms = librosa.feature.rms(y=audio, frame_length=int(round(FRAME_LENGTH * sample_rate)),
                              hop_length=int(round(HOP_LENGTH * sample_rate)))
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate,
        n_fft=2048,
        hop_length=int(round(HOP_LENGTH * sample_rate)),
    )
    return {
        "rms_energy": round(float(np.mean(rms)), 6),
        "spectral_centroid": round(float(np.mean(centroid)), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract prosodic features for priming experiment")
    parser.add_argument("--input-dir", "--outputs-dir", default=Path("segments"), type=Path,
                        help="Root of target-only WAV directory from segment.py")
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict] = []
    failures: list[str] = []
    total = len(MODELS) * len(CONDITIONS) * N_REPS
    count = 0

    for model in MODELS:
        for condition in CONDITIONS:
            for rep in range(1, N_REPS + 1):
                wav_path = args.input_dir / model / condition / f"r{rep}.wav"
                if not wav_path.exists():
                    failures.append(f"MISSING: {wav_path}")
                    continue

                try:
                    audio, sr = load_mono(wav_path)
                    mask = voiced_frames(audio, sr)
                    f0_features = extract_f0(audio, sr)
                    pause_features = detect_pauses(audio, sr)
                    speech_rate = compute_speech_rate(mask)
                    spectral = extract_spectral(audio, sr)

                    feature_rows.append({
                        "model": model,
                        "condition": condition,
                        "rep": rep,
                        "file_path": str(wav_path),
                        **f0_features,
                        "speech_rate": speech_rate,
                        **pause_features,
                        **spectral,
                    })
                    count += 1
                    print(f"[{count}/{total}] {model}/{condition}/r{rep}")

                except Exception as e:
                    failures.append(f"ERROR {wav_path}: {e}")

    if failures:
        for f in failures:
            print(f"WARN: {f}", file=sys.stderr)

    if not feature_rows:
        print("ERROR: no features extracted", file=sys.stderr)
        return 1

    fieldnames = [
        "model", "condition", "rep", "file_path",
        "f0_mean", "f0_std", "f0_range",
        "speech_rate",
        "pause_count", "pause_duration_mean",
        "rms_energy", "spectral_centroid",
    ]

    out_csv = results_dir / "features.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"\nWrote {len(feature_rows)} rows to {out_csv}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
