#!/usr/bin/env python3
"""Generate punctuation test utterances with Chatterbox."""
import csv
import sys
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent  # _5-punctuation-sensitivity/

CORPUS = PROJECT / "data" / "test_corpus.csv"
REF_AUDIO = PROJECT.parent / "_2-breathiness-preservation-benchmark" / "references" / "neutral_p229_002.wav"
OUT_DIR = PROJECT / "outputs" / "chatterbox"
SEED = 42


def main():
    # Monkey-patch broken perth watermarker
    import perth

    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)

    with open(CORPUS) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        text = row["text"]
        out_path = OUT_DIR / f"{row['id']}.wav"
        if out_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP {row['id']} (exists)")
            continue

        torch.manual_seed(SEED + i)
        if device == "cuda":
            torch.cuda.manual_seed(SEED + i)

        print(f"[{i+1}/{len(rows)}] {row['id']}: {text[:60]}...")
        wav = model.generate(text=text, audio_prompt_path=REF_AUDIO, exaggeration=0.3)
        if wav.ndim == 2:
            wav = wav.squeeze(0)
        import soundfile as sf

        sf.write(str(out_path), wav.cpu().numpy(), 24000)

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. {len(rows)} files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())