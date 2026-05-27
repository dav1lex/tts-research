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
    noun_primes = texts["noun_primes"]

    # Mapping condition name -> (prime_source, seed_offset)
    conditions = {
        "control":    (control_prime, 0),
        "subliminal": (subliminal_primes, 100),
        "noun":       (noun_primes, 200),
    }

    for cond_name, (prime_source, seed_off) in conditions.items():
        for run in range(1, texts["n_reps"] + 1):
            out_path = OUT_DIR / f"kokoro_{cond_name}_run{run}.wav"
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

            results = list(pipeline(full_text, voice="af_bella", split_pattern=None))
            chunks = []
            for r in results:
                audio = r.audio if hasattr(r, "audio") else r
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                chunks.append(audio)

            if not chunks:
                print(f"WARN: no audio for {cond_name} run{run}, skipping")
                continue
            full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            sf.write(str(out_path), full, 24000)

    n_total = len(conditions) * texts["n_reps"]
    print(f"Done. {n_total} files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
