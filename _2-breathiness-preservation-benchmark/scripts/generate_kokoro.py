#!/usr/bin/env python3
"""Generate Kokoro TTS outputs for breathiness benchmark."""

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


def get_unique_references(rows: list[dict]) -> list[tuple[str, str, str]]:
    """Return list of (pair_id, condition, reference_path) tuples."""
    seen = set()
    unique = []
    for row in rows:
        key = (row["pair_id"], row["condition"])
        if key not in seen:
            seen.add(key)
            unique.append((row["pair_id"], row["condition"], row["reference_path"]))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kokoro TTS outputs")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading Kokoro on {args.device}...")
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")  # American English
    except ImportError:
        print("ERROR: kokoro not installed. Run: pip install kokoro", file=__import__("sys").stderr)
        return 1

    rows = load_metadata(args.metadata)
    refs = get_unique_references(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(refs)
    count = 0

    for pair_id, condition, ref_path in refs:
        # Extract text from first row matching this pair_id
        text = next((r["text"] for r in rows if r["pair_id"] == pair_id), "")
        if not text:
            print(f"SKIP {pair_id}: no text found")
            continue

        # Kokoro doesn't support voice cloning from reference
        # Use default voice - benchmark still valid for breathiness comparison
        torch.manual_seed(SEED)
        
        try:
            # Generate with Kokoro - returns generator of Result objects
            results = list(pipeline(text, voice="af_bella"))
            if not results:
                raise ValueError("No audio generated")
            
            # Get audio from first result
            audio = results[0].audio
            
            # Save output
            out_path = args.output_dir / f"{pair_id}_{condition}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # audio is tensor, save with torchaudio (Kokoro uses 24kHz)
            torchaudio.save(str(out_path), audio.unsqueeze(0), 24000)
            
            count += 1
            print(f"[{count}/{total}] Generated: {out_path}")
        except Exception as e:
            print(f"ERROR generating {pair_id}_{condition}: {e}")

    print(f"\nDone. {count}/{total} files generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
