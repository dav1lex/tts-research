#!/usr/bin/env python3
"""Generate punctuation test utterances with Kokoro (no cloning baseline)."""
import csv, sys, torch, soundfile as sf
from pathlib import Path

PROJECT = Path("/home/davilex/tts-research/_5-punctuation-sensitivity")
CORPUS = PROJECT / "data" / "test_corpus.csv"
OUT_DIR = PROJECT / "outputs" / "kokoro"
SEED = 42

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from kokoro import KPipeline

    torch.manual_seed(SEED)
    pipeline = KPipeline(lang_code="a")

    with open(CORPUS) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        text = row["text"]
        out_path = OUT_DIR / f"{row['id']}.wav"
        if out_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP {row['id']} (exists)")
            continue

        print(f"[{i+1}/{len(rows)}] {row['id']}: {text[:60]}...")
        results = list(pipeline(text, voice="af_heart"))
        chunks = []
        for r in results:
            audio = r.audio if hasattr(r, "audio") else r
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            chunks.append(audio)

        import numpy as np
        full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_path), full, 24000)

    print(f"Done. {len(rows)} files in {OUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
