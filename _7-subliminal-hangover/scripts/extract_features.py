#!/usr/bin/env python3
"""Extract acoustic features using WhisperX word-level alignment.

Loads full WAV (prime + target), runs WhisperX to find exact timestamps
of target first/last words, then extracts f0_mean, f0_std, f0_cv, energy_std,
and speaking_rate from the aligned target segment only.

This replaces VAD-based boundary detection with precise word alignment,
guaranteeing we measure the same text every time.
"""
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import whisperx
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

TEXTS_PATH = PROJECT / "prompts" / "texts.json"
OUTPUTS_DIR = PROJECT / "outputs"
FEATURES_DIR = PROJECT / "features"
FEATURES_CSV = FEATURES_DIR / "features.csv"
ALIGN_LOG = PROJECT / "results" / "alignment_log.json"

# F0 extraction parameters
F0_MIN = 50
F0_MAX = 600
WHISPER_BATCH_SIZE = 16
WHISPER_MODEL = "base"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_target_word_timestamps(
    word_segments: list[dict],
    first_word: str,
    last_word: str,
) -> tuple[float, float] | None:
    """Find the start time of first_word and end time of last_word.

    word_segments: list of dicts with keys 'word', 'start', 'end'
    Returns (start_sec, end_sec) or None if not found.
    """
    start_time = None
    end_time = None

    for seg in word_segments:
        word = seg.get("word", "").strip().lower().rstrip(".,!?;:")
        if word == first_word.lower() and start_time is None:
            start_time = seg.get("start", 0.0)
        # Track end — keep updating until we hit a word that is likely
        # the end of the target. We look for the last word before silence.
        if word == last_word.lower():
            end_time = seg.get("end", 0.0)

    if start_time is not None and end_time is not None and end_time > start_time:
        return start_time, end_time

    return None


def extract_whisperx_segment(wav_path: Path, first_word: str, last_word: str):
    """Run WhisperX on a WAV, return (segment_audio, sr, start_sec, end_sec, log_info)."""
    log = {"path": str(wav_path), "status": "ok", "method": "whisperx"}

    try:
        # 1. Transcribe
        model = whisperx.load_model(WHISPER_MODEL, device=DEVICE, compute_type="float16")
        result = model.transcribe(str(wav_path), batch_size=WHISPER_BATCH_SIZE)

        # 2. Align
        align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
        result_aligned = whisperx.align(
            result["segments"], align_model, metadata, str(wav_path), DEVICE,
        )

        # 3. Extract word segments
        word_segments = result_aligned.get("word_segments", [])
        log["words_found"] = len(word_segments)

        # Try to find target sentence word timestamps
        timestamps = find_target_word_timestamps(word_segments, first_word, last_word)

        if timestamps is not None:
            start_sec, end_sec = timestamps
            log["method"] = "whisperx_aligned"
            log["start_sec"] = round(start_sec, 3)
            log["end_sec"] = round(end_sec, 3)
            log["duration_sec"] = round(end_sec - start_sec, 3)

            # Load audio and extract segment
            y, sr = sf.read(str(wav_path))
            if y.ndim > 1:
                y = y.mean(axis=1)

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)

            if end_sample - start_sample < sr * 0.1:
                log["status"] = "error"
                log["error"] = f"segment too short: {end_sample - start_sample} samples"
                return None, 0, log

            y_seg = y[start_sample:end_sample]
            return y_seg, sr, log

        log["status"] = "error"
        log["error"] = "could not find target word timestamps"
        return None, 0, log

    except Exception as e:
        log["status"] = "error"
        log["error"] = str(e)[:200]
        return None, 0, log


def extract_features_segment(y_seg: np.ndarray, sr: int, target_syllables: int):
    """Extract features from the target audio segment.

    Returns: dict with f0_mean, f0_std, f0_cv, energy_std, speaking_rate
    """
    if len(y_seg) == 0:
        return {"f0_mean": 0.0, "f0_std": 0.0, "f0_cv": 0.0,
                "energy_std": 0.0, "speaking_rate": 0.0}

    # F0 via parselmouth (Praat)
    snd = parselmouth.Sound(y_seg, sampling_frequency=sr)
    pitch = snd.to_pitch(
        time_step=0.01,
        pitch_floor=F0_MIN,
        pitch_ceiling=F0_MAX,
    )
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]

    if len(f0_values) == 0:
        f0_mean = 0.0
        f0_std = 0.0
        f0_cv = 0.0
    else:
        f0_mean = float(np.mean(f0_values))
        f0_std = float(np.std(f0_values))
        f0_cv = f0_std / f0_mean if f0_mean > 0 else 0.0

    # RMS energy std (dynamic range proxy)
    frame_len = int(sr * 20 / 1000)  # 20ms frames
    hop_len = int(sr * 10 / 1000)    # 10ms hop
    rms_frames = np.array([
        np.sqrt(np.mean(y_seg[i:i+frame_len]**2))
        for i in range(0, len(y_seg) - frame_len, hop_len)
    ])
    energy_std = float(np.std(rms_frames)) if len(rms_frames) > 1 else 0.0

    # Speaking rate: syllables / duration
    duration_s = len(y_seg) / sr
    speaking_rate = target_syllables / duration_s if duration_s > 0 else 0.0

    return {
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
        "f0_cv": round(f0_cv, 4),
        "energy_std": round(energy_std, 6),
        "speaking_rate": round(speaking_rate, 2),
    }


def parse_name_to_meta(filename: str):
    """Parse 'chatterbox_control_run1.wav' -> (model, condition, run)."""
    stem = Path(filename).stem
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
    target_syllables = texts["target_syllables"]
    first_word = texts["target_first_word"]
    last_word = texts["target"].strip().rstrip(".!?").split()[-1]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    all_features = []
    align_logs = []

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

                # WhisperX alignment
                y_seg, sr, log = extract_whisperx_segment(
                    wav_path, first_word, last_word
                )
                align_logs.append(log)

                if y_seg is None:
                    print(f"  WARN: {filename} — {log.get('error', 'unknown error')}, skipping")
                    continue

                feats = extract_features_segment(y_seg, sr, target_syllables)
                feats["model"] = model
                feats["condition"] = condition
                feats["run"] = run
                feats["filename"] = filename
                feats["target_duration_s"] = round(len(y_seg) / sr, 3)

                all_features.append(feats)
                print(f"  [{condition} run{run}] f0_cv={feats['f0_cv']}, "
                      f"f0_std={feats['f0_std']}, rate={feats['speaking_rate']}/s, "
                      f"dur={feats['target_duration_s']}s")

    # Write features CSV
    fieldnames = [
        "model", "condition", "run", "filename",
        "f0_mean", "f0_std", "f0_cv", "energy_std",
        "speaking_rate", "target_duration_s",
    ]
    with open(FEATURES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)

    # Write alignment log
    (FEATURES_CSV.parent / ".." / "results").resolve().mkdir(parents=True, exist_ok=True)
    with open(ALIGN_LOG, "w") as f:
        json.dump(align_logs, f, indent=2)

    print(f"\nSaved {len(all_features)} rows to {FEATURES_CSV}")
    print(f"Alignment log saved to {ALIGN_LOG}")

    # Print alignment quality summary
    ok_count = sum(1 for l in align_logs if l["status"] == "ok")
    fail_count = sum(1 for l in align_logs if l["status"] == "error")
    print(f"Alignment: {ok_count} OK, {fail_count} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
