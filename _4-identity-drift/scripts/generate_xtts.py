#!/usr/bin/env python3
"""Generate long-form audio with XTTS-v2 for identity drift measurement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import librosa

# --- Constants ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
TEXT_PATH = PROJECT / "data" / "long_form_text.txt"
GEN_REFERENCE = "/home/davilex/tts-research/_2-breathiness-preservation-benchmark/references/neutral_p229_002.wav"
OUT_DIR = PROJECT / "data" / "long_form" / "xtts"
OUT_PATH = OUT_DIR / "full.wav"
SEED = 42


def main() -> int:
    text = TEXT_PATH.read_text().strip()
    print(f"Loaded text: {len(text.split())} words from {TEXT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("=" * 60)
    print("GENERATING: XTTS-v2")
    print("=" * 60)

    try:
        from TTS.api import TTS

        torch.manual_seed(SEED)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        if device == "cuda":
            tts.to(device)

        tts.tts_to_file(
            text=text,
            speaker_wav=GEN_REFERENCE,
            language="en",
            file_path=str(OUT_PATH),
        )
        del tts

        duration = librosa.get_duration(path=str(OUT_PATH))
        print(f"XTTS-v2 saved to {OUT_PATH}")
        print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")

        if device == "cuda":
            torch.cuda.empty_cache()

    except ImportError:
        print("ERROR: TTS (Coqui) not installed. Run: pip install TTS", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR generating XTTS-v2: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
