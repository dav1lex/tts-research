#!/usr/bin/env python3
"""Generate punctuation test utterances with F5-TTS."""
import csv, sys, soundfile as sf
from pathlib import Path

PROJECT = Path("/home/davilex/tts-research/_5-punctuation-sensitivity")
CORPUS = PROJECT / "data" / "test_corpus.csv"
REF_AUDIO = "/home/davilex/tts-research/_2-breathiness-preservation-benchmark/references/neutral_p229_002.wav"
REF_TEXT = "Please call Stella."
OUT_DIR = PROJECT / "outputs" / "f5tts"
SEED = 42

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from f5_tts.api import F5TTS

    tts = F5TTS()

    with open(CORPUS) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        text = row["text"]
        out_path = OUT_DIR / f"{row['id']}.wav"
        if out_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP {row['id']} (exists)")
            continue

        print(f"[{i+1}/{len(rows)}] {row['id']}: {text[:60]}...")
        wav, sr, _ = tts.infer(
            ref_file=REF_AUDIO,
            ref_text=REF_TEXT,
            gen_text=text,
            nfe_step=32,
            seed=SEED + i,
        )
        sf.write(str(out_path), wav, sr)

    print(f"Done. {len(rows)} files in {OUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
