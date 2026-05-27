#!/usr/bin/env python3
"""Generate subliminal hangover test utterances with XTTS-v2.

Concatenates prime + target as single string, forces same-context processing.
Reference: VCTK p229_002 from _3-prosody-transfer-benchmark.
"""
import json
import sys
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent  # _7-subliminal-hangover/

TEXTS_PATH = PROJECT / "prompts" / "texts.json"
REF_AUDIO = PROJECT.parent / "_3-prosody-transfer-benchmark" / "references" / "modal_p229_002.wav"
OUT_DIR = PROJECT / "outputs" / "xtts"
SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    if device == "cuda":
        tts.to(device)

    with open(TEXTS_PATH) as f:
        texts = json.load(f)

    target = texts["target"]
    control_prime = texts["control_prime"]
    subliminal_primes = texts["subliminal_primes"]

    # Condition A: Control
    for run in range(1, texts["n_reps"] + 1):
        out_path = OUT_DIR / f"xtts_control_run{run}.wav"
        if out_path.exists():
            print(f"SKIP {out_path.name} (exists)")
            continue

        full_text = f"{control_prime} {target}"
        torch.manual_seed(SEED + run)
        print(f"[control run{run}] {full_text[:60]}...")
        tts.tts_to_file(
            text=full_text,
            speaker_wav=str(REF_AUDIO),
            language="en",
            file_path=str(out_path),
        )

    # Condition B: Subliminal
    for run in range(1, texts["n_reps"] + 1):
        out_path = OUT_DIR / f"xtts_subliminal_run{run}.wav"
        if out_path.exists():
            print(f"SKIP {out_path.name} (exists)")
            continue

        prime = subliminal_primes[f"run{run}"]
        full_text = f"{prime} {target}"
        torch.manual_seed(SEED + run + 100)
        print(f"[subliminal run{run}] {full_text[:60]}...")
        tts.tts_to_file(
            text=full_text,
            speaker_wav=str(REF_AUDIO),
            language="en",
            file_path=str(out_path),
        )

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Done. 10 files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
