#!/usr/bin/env python3
"""Generate subliminal hangover test utterances with Kokoro (no cloning baseline).

Concatenates prime + target as single string, forces same-context processing.
Uses split_pattern=None to prevent KPipeline from splitting at newlines.
"""
import json
import sys
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent  # _7-subliminal-hangover/

TEXTS_PATH = PROJECT / "prompts" / "texts.json"
OUT_DIR = PROJECT / "outputs" / "kokoro"
SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from kokoro import KPipeline

    torch.manual_seed(SEED)
    pipeline = KPipeline(lang_code="a")

    with open(TEXTS_PATH) as f:
        texts = json.load(f)

    target = texts["target"]
    control_prime = texts["control_prime"]
    subliminal_primes = texts["subliminal_primes"]

    # Condition A: Control
    for run in range(1, texts["n_reps"] + 1):
        out_path = OUT_DIR / f"kokoro_control_run{run}.wav"
        if out_path.exists():
            print(f"SKIP {out_path.name} (exists)")
            continue

        full_text = f"{control_prime} {target}"
        torch.manual_seed(SEED + run)
        print(f"[control run{run}] {full_text[:60]}...")

        results = list(pipeline(full_text, voice="af_bella", split_pattern=None))
        chunks = []
        for r in results:
            audio = r.audio if hasattr(r, "audio") else r
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            chunks.append(audio)

        if not chunks:
            print(f"WARN: no audio for control run{run}, skipping")
            continue
        full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_path), full, 24000)

    # Condition B: Subliminal
    for run in range(1, texts["n_reps"] + 1):
        out_path = OUT_DIR / f"kokoro_subliminal_run{run}.wav"
        if out_path.exists():
            print(f"SKIP {out_path.name} (exists)")
            continue

        prime = subliminal_primes[f"run{run}"]
        full_text = f"{prime} {target}"
        torch.manual_seed(SEED + run + 100)
        print(f"[subliminal run{run}] {full_text[:60]}...")

        results = list(pipeline(full_text, voice="af_bella", split_pattern=None))
        chunks = []
        for r in results:
            audio = r.audio if hasattr(r, "audio") else r
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            chunks.append(audio)

        if not chunks:
            print(f"WARN: no audio for subliminal run{run}, skipping")
            continue
        full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_path), full, 24000)

    print(f"Done. 10 files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
