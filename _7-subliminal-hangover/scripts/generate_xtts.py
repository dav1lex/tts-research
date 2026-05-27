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
    noun_primes = texts["noun_primes"]

    # Mapping condition name -> (prime_source, seed_offset)
    conditions = {
        "control":    (control_prime, 0),
        "subliminal": (subliminal_primes, 100),
        "noun":       (noun_primes, 200),
    }

    for cond_name, (prime_source, seed_off) in conditions.items():
        for run in range(1, texts["n_reps"] + 1):
            out_path = OUT_DIR / f"xtts_{cond_name}_run{run}.wav"
            if out_path.exists():
                print(f"SKIP {out_path.name} (exists)")
                continue

            if cond_name == "control":
                prime_text = prime_source
            else:
                prime_text = prime_source[f"run{run}"]

            full_text = f"{prime_text} {target}"
            torch.manual_seed(SEED + seed_off + run)
            print(f"[{cond_name} run{run}] {full_text[:60]}...")
            tts.tts_to_file(
                text=full_text,
                speaker_wav=str(REF_AUDIO),
                language="en",
                file_path=str(out_path),
            )

    if device == "cuda":
        torch.cuda.empty_cache()

    n_total = len(conditions) * texts["n_reps"]
    print(f"Done. {n_total} files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
