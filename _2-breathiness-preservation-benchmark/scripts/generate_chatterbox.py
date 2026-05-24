#!/usr/bin/env python3
"""Generate Chatterbox TTS outputs for breathiness benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchaudio

SEED = 42


def load_metadata(metadata_path: Path) -> list[dict]:
    with metadata_path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_unique_outputs(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    """Return list of (sample_id, pair_id, condition, reference_path) tuples."""
    seen = set()
    unique = []
    for row in rows:
        key = (row["sample_id"], row["pair_id"], row["condition"])
        if key not in seen:
            seen.add(key)
            unique.append((row["sample_id"], row["pair_id"], row["condition"], row["reference_path"]))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Chatterbox TTS outputs")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration scalar (0.0-1.0)")
    args = parser.parse_args()

    print(f"Loading Chatterbox on {args.device}...")
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=args.device)
    except ImportError:
        print("ERROR: chatterbox not installed. Run: pip install chatterbox-tts", file=__import__("sys").stderr)
        return 1

    rows = load_metadata(args.metadata)
    outputs = get_unique_outputs(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(outputs)
    count = 0

    for sample_id, pair_id, condition, ref_path in outputs:
        # Resolve reference path relative to metadata location
        ref_full = args.metadata.parent / ref_path
        
        if not ref_full.exists():
            print(f"SKIP {sample_id}: reference not found at {ref_full}")
            continue

        # Extract text
        text = next((r["text"] for r in rows if r["sample_id"] == sample_id), "")
        if not text:
            print(f"SKIP {sample_id}: no text found")
            continue

        torch.manual_seed(SEED)

        try:
            # Chatterbox: clone from reference + exaggeration control
            wav = model.generate(text=text, audio_prompt_path=str(ref_full), exaggeration=args.exaggeration)
            
            out_path = args.output_dir / f"{sample_id}_{condition}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            torchaudio.save(str(out_path), wav.cpu(), model.sr)
            
            count += 1
            print(f"[{count}/{total}] Generated: {out_path}")
            
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"ERROR generating {sample_id}_{condition}: {e}")

    print(f"\nDone. {count}/{total} files generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
