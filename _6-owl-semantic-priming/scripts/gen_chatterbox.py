#!/usr/bin/env python3
"""Generate Chatterbox TTS outputs for semantic priming experiment.

Feeds the full prime+sentence string as one text input. Chatterbox's
transformer sees the full sequence, so priming context should influence
the target sentence's prosody.

Uses the VCTK p229 reference for speaker identity only, not for
prosody cloning. Seeds vary by condition and repetition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import CONDITIONS, N_REPS, SEED, TARGET_SR, VCTK_REFERENCE, make_prompt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Chatterbox TTS outputs for priming experiment")
    parser.add_argument("--output-dir", default=Path("outputs/chatterbox"), type=Path)
    parser.add_argument("--reference", default=VCTK_REFERENCE, type=str,
                        help="Reference WAV for speaker identity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Exaggeration scalar (0.0-1.0, fixed for all runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without loading the model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ref_path = Path(args.reference)

    if not ref_path.exists():
        print(f"ERROR: reference WAV not found at {ref_path.resolve()}", file=sys.stderr)
        return 1

    if args.dry_run:
        for condition in CONDITIONS:
            text = make_prompt(condition)
            print(f"[dry-run] {condition}: {len(text)} chars, {len(text.split())} words")
        return 0

    if args.device != "cuda":
        print("ERROR: CPU generation is disabled for this experiment. Use --device cuda.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Run on a CUDA machine or do not generate.", file=sys.stderr)
        return 1

    print(f"Loading Chatterbox on {args.device}...")
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=args.device)
    except ImportError:
        print("ERROR: chatterbox not installed. Run: pip install chatterbox-tts", file=sys.stderr)
        return 1

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
                wav = model.generate(
                    text=full_prompt,
                    audio_prompt_path=str(ref_path),
                    exaggeration=args.exaggeration,
                )

                out_path = cond_dir / f"r{rep}.wav"

                # Standardize to TARGET_SR mono
                if wav.ndim > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if model.sr != TARGET_SR:
                    wav = torchaudio.functional.resample(wav, model.sr, TARGET_SR)

                torchaudio.save(str(out_path), wav.cpu(), TARGET_SR)
                count += 1
                print(f"[{count}/{total}] Generated: {out_path} (seed={seed})")

                if args.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"ERROR generating {condition}/r{rep}: {e}", file=sys.stderr)

    print(f"\nDone. {count}/{total} files generated.")
    return 0 if count == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
