#!/usr/bin/env python3
"""
Generate long-form audio with Chatterbox TTS for identity drift measurement.

LIMITATION: Chatterbox has a limited context window and crashes with a
CUDA "indexSelectLargeIndex" assertion when fed very long text (e.g., 868
words).  To work around this, we split the input text into sentence-aligned
chunks of ~100 words, synthesise each chunk independently with the same
reference audio prompt (incrementing the random seed per chunk to avoid
identical output), and concatenate the resulting audio tensors before saving.
Papers 2 and 3 used ~15-30 word sentences and did not require this split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchaudio
import librosa

# --- Constants ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
TEXT_PATH = PROJECT / "data" / "long_form_text.txt"
GEN_REFERENCE = (
    "/home/davilex/tts-research/_2-breathiness-preservation-benchmark/"
    "references/neutral_p229_002.wav"
)
OUT_DIR = PROJECT / "data" / "long_form" / "chatterbox"
OUT_PATH = OUT_DIR / "full.wav"
BASE_SEED = 42
CHUNK_SIZE_WORDS = 100  # target words per chunk (sentences kept intact)


def split_into_chunks(text: str, max_words: int = CHUNK_SIZE_WORDS) -> list[str]:
    """Split *text* into segments of at most *max_words* words,
    respecting sentence boundaries (period + space)."""
    # Split on ". " to isolate sentences without losing the delimiter.
    raw_sentences = [s.strip() for s in text.split(". ")]
    # Restore trailing period on each sentence that needs one.
    sentences: list[str] = []
    for i, s in enumerate(raw_sentences):
        if not s:
            continue
        needs_period = (
            not s.endswith(".")
            and (i < len(raw_sentences) - 1 or text.rstrip().endswith("."))
        )
        sentences.append(s + "." if needs_period else s)

    chunks: list[str] = []
    current: list[str] = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current and current_word_count + sent_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_word_count = 0
        current.append(sent)
        current_word_count += sent_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def main() -> int:
    text = TEXT_PATH.read_text().strip()
    total_words = len(text.split())
    print(f"Loaded text: {total_words} words from {TEXT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("=" * 60)
    print("GENERATING: Chatterbox (chunked)")
    print("=" * 60)

    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        print(
            "ERROR: chatterbox not installed. Run: pip install chatterbox-tts",
            file=sys.stderr,
        )
        return 1

    model = ChatterboxTTS.from_pretrained(device=device)

    chunks = split_into_chunks(text, max_words=CHUNK_SIZE_WORDS)
    print(
        f"Split into {len(chunks)} chunks "
        f"(target ≤{CHUNK_SIZE_WORDS} words each):"
    )
    for i, ch in enumerate(chunks):
        print(f"  chunk {i + 1}: {len(ch.split())} words")

    wavs: list[torch.Tensor] = []

    for i, chunk in enumerate(chunks):
        seed = BASE_SEED + i
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

        print(
            f"[{i + 1}/{len(chunks)}] Generating chunk "
            f"({len(chunk.split())} words, seed={seed})…"
        )

        wav = model.generate(
            text=chunk,
            audio_prompt_path=GEN_REFERENCE,
            exaggeration=0.3,
        )

        # Squeeze channel dim if present so concatenation along last axis works.
        if wav.ndim == 2:
            wav = wav.squeeze(0)

        wavs.append(wav.cpu())

        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatenate along the time axis (last / only dimension).
    full_wav = torch.cat(wavs, dim=-1)
    torchaudio.save(str(OUT_PATH), full_wav.unsqueeze(0), model.sr)

    del model, wavs, full_wav
    if device == "cuda":
        torch.cuda.empty_cache()

    duration = librosa.get_duration(path=str(OUT_PATH))
    print(f"\nChatterbox saved to {OUT_PATH}")
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
