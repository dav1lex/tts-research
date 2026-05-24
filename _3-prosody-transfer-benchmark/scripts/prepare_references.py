#!/usr/bin/env python3
"""Convert hardcoded VCTK FLAC pairs to 16 kHz mono WAV references.

Pairs (hardcoded — do not re-select):
  Pair 001: p240 (breathy) vs p229 (modal), sentence 002
  Pair 002: p253 (breathy) vs p301 (modal), sentence 005
  Pair 003: p264 (breathy) vs p282 (modal), sentence 006
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import soundfile as sf

VCTK_DIR = Path("/home/davilex/Downloads/VCTK-Corpus-0.92/wav48_silence_trimmed")
TARGET_SR = 16000

PAIRS = {
    "pair_001": {
        "breathy": ("p240", "002"),
        "modal": ("p229", "002"),
        "text": "Ask her to bring these things with her from the store.",
    },
    "pair_002": {
        "breathy": ("p253", "005"),
        "modal": ("p301", "005"),
        "text": "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
    },
    "pair_003": {
        "breathy": ("p264", "006"),
        "modal": ("p282", "006"),
        "text": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
    },
}


def convert_flac_to_wav(flac_path: Path, wav_path: Path, target_sr: int = TARGET_SR) -> None:
    """Load FLAC, resample to target_sr mono, write WAV."""
    audio, sr = librosa.load(str(flac_path), sr=target_sr, mono=True)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, target_sr)


def build_metadata_rows(ref_dir_name: str = "references") -> list[dict[str, str]]:
    """Build metadata.csv rows for all model outputs (same structure as _2)."""
    rows = []
    models = [("chatterbox", "outputs/chatterbox"), ("xtts", "outputs/xtts"), ("kokoro", "outputs/kokoro")]

    conditions = {"breathy", "modal"}
    for pair_id, pair in PAIRS.items():
        for condition in conditions:
            if condition not in pair:
                continue
            speaker, sentence = pair[condition]
            sample_id = f"{speaker}_{sentence}"
            ref_name = f"{condition}_{speaker}_{sentence}.wav"
            ref_path = f"{ref_dir_name}/{ref_name}"

            for model_name, out_dir in models:
                if model_name == "kokoro":
                    out_name = f"{pair_id}_{condition}.wav"
                else:
                    out_name = f"{sample_id}_{condition}.wav"

                out_path = f"{out_dir}/{out_name}"

                rows.append({
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "text": pair["text"],
                    "condition": condition,
                    "reference_path": ref_path,
                    "output_path": out_path,
                    "model": model_name,
                    "seed": "42",
                    "notes": f"{speaker} {sentence}",
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare VCTK reference WAVs for prosody transfer benchmark")
    parser.add_argument("--vctk-dir", default=VCTK_DIR, type=Path, help="VCTK corpus root (wav48_silence_trimmed)")
    parser.add_argument("--ref-dir", default=Path("references"), type=Path, help="Output directory for reference WAVs")
    parser.add_argument("--metadata", default=Path("metadata.csv"), type=Path, help="Output metadata CSV path")
    args = parser.parse_args()

    # Resolve paths relative to script location or absolute
    ref_dir = args.ref_dir.resolve()
    metadata_path = args.metadata.resolve() if not args.metadata.is_absolute() else args.metadata
    vctk_wav = args.vctk_dir.resolve()

    if not vctk_wav.exists():
        print(f"ERROR: VCTK directory not found at {vctk_wav}", file=sys.stderr)
        return 1

    # Convert FLACs to WAVs
    converted = 0
    conditions = {"breathy", "modal"}
    for pair_id, pair in PAIRS.items():
        for condition in conditions:
            if condition not in pair:
                continue
            speaker, sentence = pair[condition]
            flac_name = f"{speaker}_{sentence}_mic1.flac"
            flac_path = vctk_wav / speaker / flac_name

            if not flac_path.exists():
                print(f"WARN: {flac_path} not found, skipping", file=sys.stderr)
                continue

            wav_name = f"{condition}_{speaker}_{sentence}.wav"
            wav_path = ref_dir / wav_name
            convert_flac_to_wav(flac_path, wav_path)
            print(f"  {flac_name} -> {wav_path}")
            converted += 1

    print(f"Converted {converted} files to {ref_dir}")

    # Write metadata.csv (relative paths like _2)
    rows = build_metadata_rows(ref_dir.name)
    fieldnames = ["sample_id", "pair_id", "text", "condition", "reference_path", "output_path", "model", "seed", "notes"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
