#!/usr/bin/env python3
"""Generate Kokoro TTS outputs for semantic priming experiment.

CRITICAL: Uses split_pattern=None to prevent the default newline-splitting
behavior. Without this, Kokoro splits "prime\n\ntarget" into independent
segments, each G2P'd and synthesized in isolation — killing the priming
effect entirely.

With split_pattern=None, the full text stays as one segment. The
transformer's self-attention sees the complete phoneme sequence, so
priming context can influence the target sentence's prosody.

Side effect: Kokoro may apply sentence-final intonation across the
prime paragraphs. This is expected — maximum context bleed is desirable.

Additionally: verifies character count against the 510-phoneme chunking
floor. If exceeded, en_tokenize silently splits — corrupting the control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import (  # noqa: E402
    CONDITIONS,
    KOKORO_SAFE_CHAR_LIMIT,
    KOKORO_VOICE,
    N_REPS,
    SEED,
    TARGET_SR,
    make_prompt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kokoro TTS outputs for priming experiment")
    parser.add_argument("--output-dir", default=Path("outputs/kokoro"), type=Path)
    parser.add_argument("--voice", default=KOKORO_VOICE, help="Kokoro voice preset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt lengths and exit without generating")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # --- Phase 0: char-count verification ---
    print("Sanity check: character counts vs phoneme ceiling")
    print(f"  Safe char limit: {KOKORO_SAFE_CHAR_LIMIT} (below 510-phoneme floor)")
    warnings = []
    for condition in CONDITIONS:
        text = make_prompt(condition)
        n_chars = len(text)
        n_words = len(text.split())
        status = "OK" if n_chars < KOKORO_SAFE_CHAR_LIMIT else "WARNING: exceeds safe limit"
        if n_chars >= KOKORO_SAFE_CHAR_LIMIT:
            warnings.append(condition)
        print(f"  {condition:15s}  {n_chars:4d} chars  {n_words:3d} words  {status}")

    if warnings:
        print(f"\nDANGER: {', '.join(warnings)} exceed the ~390 char safety margin.", file=sys.stderr)
        print("Kokoro may silently split at a punctuation boundary, corrupting the control.", file=sys.stderr)
        print("Proceed only if you understand the risk.", file=sys.stderr)
        # Don't hard-fail — let the user decide. But return non-zero.

    if args.dry_run:
        return 1 if warnings else 0

    if args.device != "cuda":
        print("ERROR: CPU generation is disabled for this experiment. Use --device cuda.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Run on a CUDA machine or do not generate.", file=sys.stderr)
        return 1

    # --- Phase 1: load model ---
    print(f"\nLoading Kokoro on {args.device}...")
    try:
        from kokoro import KPipeline
    except ImportError:
        print("ERROR: kokoro not installed. Run: pip install kokoro", file=sys.stderr)
        return 1

    pipeline = KPipeline(lang_code="a")  # American English

    total = len(CONDITIONS) * N_REPS
    count = 0

    for condition_index, condition in enumerate(CONDITIONS):
        cond_dir = output_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        full_prompt = make_prompt(condition)

        for rep in range(1, N_REPS + 1):
            seed = SEED + condition_index * 100 + rep
            torch.manual_seed(seed)

            try:
                # CRITICAL: split_pattern=None keeps prime+sentence as one unit.
                # Default r'\n+' would split on newlines → independent segments → no priming.
                results = list(pipeline(full_prompt, voice=args.voice, split_pattern=None))

                if not results:
                    raise ValueError("No audio generated")

                # Concatenate all result chunks into one waveform.
                # With split_pattern=None, there should only be one chunk — but
                # if en_tokenize still split at 510-phoneme boundary, there will
                # be multiple. We concatenate to preserve the full audio.
                chunks = []
                for result in results:
                    audio = result.audio
                    if audio.ndim == 0:
                        audio = audio.unsqueeze(0)
                    chunks.append(audio)

                wav = torch.cat(chunks, dim=-1)

                # Kokoro outputs at 24kHz. Resample to TARGET_SR.
                wav = torchaudio.functional.resample(wav, 24000, TARGET_SR)
                if wav.ndim > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                out_path = cond_dir / f"r{rep}.wav"
                torchaudio.save(str(out_path), wav.unsqueeze(0).cpu() if wav.ndim == 1 else wav.cpu(), TARGET_SR)
                count += 1

                n_chunks = len(results)
                chunk_info = f" ({n_chunks} chunk{'s' if n_chunks > 1 else ''})" if n_chunks > 1 else ""
                print(f"[{count}/{total}] Generated: {out_path}{chunk_info} (seed={seed})")

            except Exception as e:
                print(f"ERROR generating {condition}/r{rep}: {e}", file=sys.stderr)

    print(f"\nDone. {count}/{total} files generated.")
    ret = 0
    if warnings:
        ret = 1
    if count != total:
        ret = 2
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
