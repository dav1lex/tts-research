#!/usr/bin/env python3
"""Generate subliminal hangover test utterances with Chatterbox.

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
OUT_DIR = PROJECT / "outputs" / "chatterbox"
SEED = 42


def main():
    # Monkey-patch broken perth watermarker (Chatterbox quirk)
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(TEXTS_PATH) as f:
        texts = json.load(f)

    target = texts["target"]
    control_prime = texts["control_prime"]
    subliminal_primes = texts["subliminal_primes"]

    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=device)

    # Condition A: Control
    for run in range(1, texts["n_reps"] + 1):
        out_path = OUT_DIR / f"chatterbox_control_run{run}.wav"
        if out_path.exists():
            print(f"SKIP {out_path.name} (exists)")
            continue

        full_text = f"{control_prime} {target}"
        torch.manual_seed(SEED + run)
        if device == "cuda":
            torch.cuda.manual_seed(SEED + run)

        print(f"[control run{run}] {full_text[:60]}...")
        wav = model.generate(text=full_text, audio_prompt_path=str(REF_AUDIO), exaggeration=0.3)
        if wav.ndim == 2:
            wav = wav.squeeze(0)

        import soundfile as sf
        sf.write(str(out_path), wav.cpu().numpy(), 24000)
        if device == "cuda":
            torch.cuda.empty_cache()

    # Condition B: Subliminal (5 different number sequences)
    for run in range(1, texts["n_reps"] + 1):
        out_path = OUT_DIR / f"chatterbox_subliminal_run{run}.wav"
        if out_path.exists():
            print(f"SKIP {out_path.name} (exists)")
            continue

        prime = subliminal_primes[f"run{run}"]
        full_text = f"{prime} {target}"
        torch.manual_seed(SEED + run + 100)
        if device == "cuda":
            torch.cuda.manual_seed(SEED + run + 100)

        print(f"[subliminal run{run}] {full_text[:60]}...")
        wav = model.generate(text=full_text, audio_prompt_path=str(REF_AUDIO), exaggeration=0.3)
        if wav.ndim == 2:
            wav = wav.squeeze(0)

        import soundfile as sf
        sf.write(str(out_path), wav.cpu().numpy(), 24000)
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. 10 files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
