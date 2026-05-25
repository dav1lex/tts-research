#!/usr/bin/env python3
"""Generate punctuation test utterances with XTTS-v2."""
import csv, sys, torch
from pathlib import Path

PROJECT = Path("/home/davilex/tts-research/_5-punctuation-sensitivity")
CORPUS = PROJECT / "data" / "test_corpus.csv"
REF_AUDIO = "/home/davilex/tts-research/_2-breathiness-preservation-benchmark/references/neutral_p229_002.wav"
OUT_DIR = PROJECT / "outputs" / "xtts"
SEED = 42

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    if device == "cuda":
        tts.to(device)

    with open(CORPUS) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        text = row["text"]
        out_path = OUT_DIR / f"{row['id']}.wav"
        if out_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP {row['id']} (exists)")
            continue

        torch.manual_seed(SEED + i)
        print(f"[{i+1}/{len(rows)}] {row['id']}: {text[:60]}...")
        tts.tts_to_file(text=text, speaker_wav=REF_AUDIO, language="en", file_path=str(out_path))

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Done. {len(rows)} files in {OUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
