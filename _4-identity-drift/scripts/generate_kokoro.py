#!/usr/bin/env python3
"""Generate long-form audio with Kokoro TTS for identity drift measurement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import librosa

# --- Constants ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
TEXT_PATH = PROJECT / "data" / "long_form_text.txt"
GEN_REFERENCE = "/home/davilex/tts-research/_2-breathiness-preservation-benchmark/references/neutral_p229_002.wav"
OUT_DIR = PROJECT / "data" / "long_form" / "kokoro"
OUT_PATH = OUT_DIR / "full.wav"
SEED = 42


def main() -> int:
    text = TEXT_PATH.read_text().strip()
    print(f"Loaded text: {len(text.split())} words from {TEXT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("=" * 60)
    print('GENERATING: Kokoro (voice=af_heart, no cloning)')
    print("=" * 60)

    try:
        from kokoro import KPipeline

        torch.manual_seed(SEED)
        pipeline = KPipeline(lang_code="a")
        results = list(pipeline(text, voice="af_heart"))

        # Concatenate all pipeline chunks
        all_audio = []
        for result in results:
            if hasattr(result, "audio"):
                audio_chunk = result.audio
            else:
                audio_chunk = result
            if isinstance(audio_chunk, torch.Tensor):
                audio_chunk = audio_chunk.cpu().numpy()
            all_audio.append(audio_chunk.flatten())

        full_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
        sf.write(str(OUT_PATH), full_audio.astype(np.float32), 24000)

        del pipeline, full_audio

        duration = librosa.get_duration(path=str(OUT_PATH))
        print(f"Kokoro saved to {OUT_PATH}")
        print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")

        if device == "cuda":
            torch.cuda.empty_cache()

    except ImportError:
        print("ERROR: kokoro not installed. Run: pip install kokoro", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR generating Kokoro: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
