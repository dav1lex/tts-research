#!/usr/bin/env python3
"""Generate XTTS-v2 TTS outputs for prosody transfer benchmark.

Same structure as _2-breathiness-preservation-benchmark.
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
        if row["model"].strip().lower() != "xtts":
            continue
        key = (row["sample_id"], row["pair_id"], row["condition"], row["output_path"])
        if key not in seen:
            seen.add(key)
            unique.append((row["sample_id"], row["pair_id"], row["condition"],
                           row["reference_path"], row["output_path"]))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate XTTS-v2 TTS outputs for prosody benchmark")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading XTTS-v2 on {args.device}...")
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        if args.device == "cuda":
            tts.to(args.device)
    except ImportError:
        print("ERROR: TTS (Coqui) not installed. Run: pip install TTS",
              file=__import__("sys").stderr)
        return 1

    rows = load_metadata(args.metadata)
    outputs = get_unique_outputs(rows)
    base_dir = args.metadata.parent

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(outputs)
    count = 0

    for sample_id, pair_id, condition, ref_path, out_path in outputs:
        ref_full = base_dir / ref_path
        if not ref_full.exists():
            print(f"SKIP {sample_id}: reference not found at {ref_full}")
            continue

        text = next((r["text"] for r in rows if r["sample_id"] == sample_id), "")
        if not text:
            print(f"SKIP {sample_id}: no text found")
            continue

        torch.manual_seed(SEED)

        try:
            out_full = args.output_dir / Path(out_path).name
            out_full.parent.mkdir(parents=True, exist_ok=True)

            tts.tts_to_file(
                text=text,
                speaker_wav=str(ref_full),
                language="en",
                file_path=str(out_full),
            )

            count += 1
            print(f"[{count}/{total}] Generated: {out_full}")
        except Exception as e:
            print(f"ERROR generating {sample_id}_{condition}: {e}")

    print(f"\nDone. {count}/{total} files generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
