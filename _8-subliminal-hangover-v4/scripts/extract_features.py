#!/usr/bin/env python3
"""Extract WhisperX-aligned target-segment features for V4 from manifest.csv.

For each row in manifest.csv:
  - load WAV at output_relpath
  - align to find target segment boundaries (first/last target word)
  - compute f0_mean, f0_std, f0_cv, energy_std, speaking_rate

Writes:
  - features/features.csv
  - results/alignment_log.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
import torch
import whisperx


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MANIFEST_CSV = PROJECT_DIR / "manifest.csv"
TARGETS_JSON = PROJECT_DIR / "prompts" / "targets.json"

FEATURES_DIR = PROJECT_DIR / "features"
FEATURES_CSV = FEATURES_DIR / "features.csv"
RESULTS_DIR = PROJECT_DIR / "results"
ALIGN_LOG = RESULTS_DIR / "alignment_log.json"

# F0 extraction params
F0_MIN = 50
F0_MAX = 600

WHISPER_MODEL = "base"
WHISPER_BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_word(w: str) -> str:
    return w.strip().lower().rstrip(".,!?;:")


def _target_bounds_from_words(word_segments: list[dict], first_word: str, last_word: str):
    start_time = None
    end_time = None
    fw = _normalize_word(first_word)
    lw = _normalize_word(last_word)
    for seg in word_segments:
        w = _normalize_word(seg.get("word", ""))
        if start_time is None and w == fw:
            start_time = seg.get("start", None)
        if w == lw:
            end_time = seg.get("end", None)
    if start_time is None or end_time is None:
        return None
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        return None
    if end_time <= start_time:
        return None
    return float(start_time), float(end_time)


def _extract_segment(wav_path: Path, first_word: str, last_word: str, model, align_model, align_metadata):
    log = {"path": str(wav_path), "status": "ok", "device": DEVICE}
    try:
        result = model.transcribe(str(wav_path), batch_size=WHISPER_BATCH_SIZE)
        aligned = whisperx.align(result["segments"], align_model, align_metadata, str(wav_path), DEVICE)
        word_segments = aligned.get("word_segments", [])
        log["words_found"] = len(word_segments)
        bounds = _target_bounds_from_words(word_segments, first_word, last_word)
        if bounds is None:
            log["status"] = "error"
            log["error"] = "could not find target word timestamps"
            return None, 0, log

        start_sec, end_sec = bounds
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = y.mean(axis=1)

        s0 = max(0, int(start_sec * sr))
        s1 = min(len(y), int(end_sec * sr))
        if s1 <= s0:
            log["status"] = "error"
            log["error"] = "empty segment after slicing"
            return None, 0, log

        log["start_sec"] = round(start_sec, 3)
        log["end_sec"] = round(end_sec, 3)
        log["duration_sec"] = round((s1 - s0) / sr, 3)
        return y[s0:s1], sr, log
    except Exception as e:
        log["status"] = "error"
        log["error"] = str(e)[:200]
        return None, 0, log


def _features(y_seg: np.ndarray, sr: int, syllables: int) -> dict[str, float]:
    if len(y_seg) == 0:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_cv": 0.0,
            "energy_std": 0.0,
            "speaking_rate": 0.0,
        }

    snd = parselmouth.Sound(y_seg, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        f0_mean = 0.0
        f0_std = 0.0
        f0_cv = 0.0
    else:
        f0_mean = float(np.mean(f0))
        f0_std = float(np.std(f0))
        f0_cv = f0_std / f0_mean if f0_mean > 0 else 0.0

    # RMS energy std
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)
    if len(y_seg) <= frame_len:
        energy_std = 0.0
    else:
        rms = np.array(
            [np.sqrt(np.mean(y_seg[i : i + frame_len] ** 2)) for i in range(0, len(y_seg) - frame_len, hop_len)]
        )
        energy_std = float(np.std(rms)) if len(rms) > 1 else 0.0

    dur_s = len(y_seg) / sr
    rate = (syllables / dur_s) if dur_s > 0 else 0.0

    return {
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
        "f0_cv": round(f0_cv, 6),
        "energy_std": round(energy_std, 6),
        "speaking_rate": round(rate, 3),
    }


def main() -> int:
    if DEVICE != "cuda":
        print("ERROR: CUDA is required for WhisperX alignment in this workflow. Refusing to run on CPU.", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="",
        help="Optional model filter (e.g. chatterbox, kokoro, xtts). Comma-separated allowed.",
    )
    args = ap.parse_args()

    if not MANIFEST_CSV.exists():
        print(f"ERROR: missing {MANIFEST_CSV}", file=sys.stderr)
        return 2
    if not TARGETS_JSON.exists():
        print(f"ERROR: missing {TARGETS_JSON}", file=sys.stderr)
        return 2

    targets_data = _load_json(TARGETS_JSON)
    targets = {t["id"]: t for t in targets_data.get("targets", []) if t.get("id")}

    with MANIFEST_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    model_filter = None
    if args.model.strip():
        model_filter = {m.strip() for m in args.model.split(",") if m.strip()}
        rows = [r for r in rows if (r.get("model") or "").strip() in model_filter]

    # Load WhisperX models once
    wx_model = whisperx.load_model(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feats_rows = []
    logs = []

    for r in rows:
        sample_id = r["id"]
        model = r.get("model", "")
        cond = r.get("condition", "")
        target_id = r.get("target_id", "")
        rep = r.get("repetition", "")
        out_rel = r.get("output_relpath", "")

        target = targets.get(target_id)
        if not target:
            logs.append({"id": sample_id, "status": "error", "error": f"unknown target_id {target_id}"})
            continue

        target_text = str(target.get("text", "")).strip()
        words = [w for w in target_text.split() if w]
        if not words:
            logs.append({"id": sample_id, "status": "error", "error": f"empty target text for {target_id}"})
            continue

        first_word = words[0]
        last_word = words[-1].rstrip(".!?;:,")
        syllables = target.get("syllables")
        try:
            syllables_i = int(syllables) if syllables is not None else 0
        except Exception:
            syllables_i = 0

        wav_path = PROJECT_DIR / out_rel
        if not wav_path.exists():
            logs.append(
                {
                    "id": sample_id,
                    "model": model,
                    "condition": cond,
                    "target_id": target_id,
                    "repetition": rep,
                    "path": str(wav_path),
                    "status": "error",
                    "error": "missing wav",
                }
            )
            continue

        y_seg, sr, log = _extract_segment(wav_path, first_word, last_word, wx_model, align_model, metadata)
        log["id"] = sample_id
        log["model"] = model
        log["condition"] = cond
        log["target_id"] = target_id
        log["repetition"] = rep
        logs.append(log)
        if y_seg is None:
            continue

        feat = _features(y_seg, sr, syllables_i)
        feat.update(
            {
                "id": sample_id,
                "model": model,
                "condition": cond,
                "target_id": target_id,
                "repetition": rep,
                "filename": wav_path.name,
                "target_duration_s": round(len(y_seg) / sr, 3),
            }
        )
        feats_rows.append(feat)

    # Write features
    fieldnames = [
        "id",
        "model",
        "condition",
        "target_id",
        "repetition",
        "filename",
        "f0_mean",
        "f0_std",
        "f0_cv",
        "energy_std",
        "speaking_rate",
        "target_duration_s",
    ]
    with FEATURES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(feats_rows)

    ALIGN_LOG.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(f"Saved {len(feats_rows)} rows to {FEATURES_CSV}")
    print(f"Alignment log saved to {ALIGN_LOG}")

    ok = sum(1 for l in logs if l.get("status") == "ok")
    bad = sum(1 for l in logs if l.get("status") == "error")
    print(f"Alignment: {ok} OK, {bad} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
