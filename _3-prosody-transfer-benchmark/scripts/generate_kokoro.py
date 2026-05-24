#!/usr/bin/env python3
"""Generate Kokoro TTS outputs for prosody transfer benchmark.

Same structure as _2-breathiness-preservation-benchmark.
Kokoro does not support voice cloning — uses default voice.
"""

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


def get_unique_outputs(rows: list[dict]) -> list[tuple[str, str, str, str, str]]:
    """Return list of (sample_id, pair_id, condition, reference_path, output_path) tuples."""
    seen = set()
    unique = []
    for row in rows:
        if row["model"].strip().lower() != "kokoro":
            continue
        key = (row["sample_id"], row["pair_id"], row["condition"], row["output_path"])
        if key not in seen:
            seen.add(key)
            unique.append((row["sample_id"], row["pair_id"], row["condition"],
                           row["reference_path"], row["output_path"]))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kokoro TTS outputs for prosody benchmark")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading Kokoro on {args.device}...")
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")  # American English
    except ImportError:
        print("ERROR: kokoro not installed. Run: pip install kokoro",
              file=__import__("sys").stderr)
        return 1

    rows = load_metadata(args.metadata)
    outputs = get_unique_outputs(rows)
    base_dir = args.metadata.parent

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(outputs)
    count = 0

    # Kokoro doesn't do voice cloning — generate once per unique text+condition
    # but still save per metadata entry
    cache: dict[str, torch.Tensor] = {}

    for sample_id, pair_id, condition, ref_path, out_path in outputs:
        text = next((r["text"] for r in rows if r["sample_id"] == sample_id), "")
        if not text:
            print(f"SKIP {sample_id}: no text found")
            continue

        # Cache by text+condition since Kokoro has fixed voice
        cache_key = f"{text}_{condition}"
        if cache_key in cache:
            audio = cache[cache_key]
        else:
            torch.manual_seed(SEED)
            try:
                results = list(pipeline(text, voice="af_bella"))
                if not results:
                    raise ValueError("No audio generated")
                audio = results[0].audio
                cache[cache_key] = audio
            except Exception as e:
                print(f"ERROR generating {sample_id}_{condition}: {e}")
                continue

        try:
            out_full = args.output_dir / Path(out_path).name
            out_full.parent.mkdir(parents=True, exist_ok=True)

            # Kokoro outputs at 24 kHz
            torchaudio.save(str(out_full), audio.unsqueeze(0), 24000)

            count += 1
            print(f"[{count}/{total}] Generated: {out_full}")
        except Exception as e:
            print(f"ERROR saving {sample_id}_{condition}: {e}")

    print(f"\nDone. {count}/{total} files generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
