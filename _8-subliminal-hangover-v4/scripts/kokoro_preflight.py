#!/usr/bin/env python3
"""
Kokoro determinism preflight (V4).

Goal: detect whether repeated generation is effectively deterministic under this environment.

Method:
  1) Generate N reps of the same full_text (prime + target) using Kokoro af_bella.
  2) Save WAVs to results/preflight_kokoro/.
  3) Compute sha256 of each WAV (byte-level identity).
  4) Compute a coarse f0_cv on the full audio (no alignment) via parselmouth pitch.

This is a cheap gate before committing to Stage A / full study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
TARGETS_JSON = PROMPTS_DIR / "targets.json"
PRIMES_JSON = PROMPTS_DIR / "primes.json"

RESULTS_DIR = PROJECT_DIR / "results"
OUT_DIR = RESULTS_DIR / "preflight_kokoro"
OUT_JSON = RESULTS_DIR / "kokoro_preflight.json"

SR = 24000
F0_MIN = 50
F0_MAX = 600


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load_json(path: Path) -> dict:
    if not path.exists():
        _die(f"missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Target:
    id: str
    text: str


def _load_targets() -> dict[str, Target]:
    data = _load_json(TARGETS_JSON)
    out: dict[str, Target] = {}
    for t in data.get("targets", []):
        tid = (t.get("id") or "").strip()
        text = (t.get("text") or "").strip()
        if not tid or not text:
            continue
        out[tid] = Target(id=tid, text=text)
    if not out:
        _die("targets.json has no usable targets")
    return out


def _load_primes(condition: str) -> dict[str, str]:
    data = _load_json(PRIMES_JSON)
    key = "noun_primes" if condition == "noun" else "number_primes"
    primes = data.get(key) or {}
    if not primes:
        _die(f"primes.json missing {key}")
    return {k: str(v) for k, v in primes.items()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _f0_cv_full_audio(y: np.ndarray, sr: int) -> float:
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        return 0.0
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return 0.0
    mean = float(np.mean(f0))
    std = float(np.std(f0))
    return std / mean if mean > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-id", default="t01")
    ap.add_argument("--condition", choices=["noun", "number"], default="number")
    ap.add_argument("--prime-id", default="p01")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=4242)
    ap.add_argument("--same-seed", action="store_true", help="Use identical seed for all reps.")
    args = ap.parse_args()

    if args.reps <= 1:
        _die("--reps must be >= 2")

    targets = _load_targets()
    if args.target_id not in targets:
        _die(f"unknown target id: {args.target_id}")
    target = targets[args.target_id]

    primes = _load_primes(args.condition)
    if args.prime_id not in primes:
        _die(f"unknown prime id: {args.prime_id}")
    prime_text = primes[args.prime_id].strip()

    full_text = f"{prime_text} {target.text}".strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")

    results = {
        "target_id": args.target_id,
        "condition": args.condition,
        "prime_id": args.prime_id,
        "reps": args.reps,
        "seed_base": args.seed_base,
        "same_seed": bool(args.same_seed),
        "full_text_preview": full_text[:120],
        "files": [],
        "hash_unique": None,
        "f0_cv_values": [],
        "f0_cv_std": None,
    }

    for i in range(1, args.reps + 1):
        seed = args.seed_base if args.same_seed else (args.seed_base + i)
        torch.manual_seed(seed)
        out_path = OUT_DIR / f"kokoro_preflight_{args.condition}_{args.target_id}_{args.prime_id}_r{i:02d}.wav"

        chunks = []
        for r in pipeline(full_text, voice="af_bella", split_pattern=None):
            audio = r.audio if hasattr(r, "audio") else r
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            chunks.append(audio)

        if not chunks:
            _die(f"no audio returned for rep={i}")

        y = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_path), y, SR)

        wav, sr = sf.read(str(out_path))
        f0cv = _f0_cv_full_audio(wav, sr)
        h = _sha256(out_path)
        results["files"].append(
            {
                "rep": i,
                "seed": seed,
                "path": str(out_path),
                "sha256": h,
                "duration_s": round(len(wav) / sr, 3),
                "f0_cv_full": round(float(f0cv), 6),
            }
        )
        results["f0_cv_values"].append(float(f0cv))

    hashes = [f["sha256"] for f in results["files"]]
    results["hash_unique"] = len(set(hashes))
    f0_vals = np.array(results["f0_cv_values"], dtype=float)
    results["f0_cv_std"] = float(np.std(f0_vals))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Simple terminal summary
    print(f"wrote {args.reps} files to {OUT_DIR}")
    print(f"unique_sha256={results['hash_unique']} of {args.reps}")
    print(f"f0_cv_full std={results['f0_cv_std']:.6f}")
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()

