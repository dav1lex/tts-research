#!/usr/bin/env python3
"""Generate V4 utterances with Kokoro from manifest.csv.

Reads manifest rows where model == "kokoro", synthesizes `text` as a single context window,
and writes WAVs to the path in `output_relpath`.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MANIFEST_CSV = PROJECT_DIR / "manifest.csv"

SR = 24000
VOICE = "af_bella"


def main() -> int:
    # Default to offline so we don't accidentally hit the network in normal runs.
    # If the model isn't cached yet, run kokoro_preflight.py once with network enabled.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not MANIFEST_CSV.exists():
        print(f"ERROR: missing {MANIFEST_CSV}", file=sys.stderr)
        return 2

    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")

    with MANIFEST_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kokoro_rows = [r for r in rows if (r.get("model") or "").strip() == "kokoro"]
    if not kokoro_rows:
        print("No kokoro rows found in manifest.csv")
        return 0

    ok = 0
    fail = 0
    for r in kokoro_rows:
        sample_id = r["id"]
        text = (r.get("text") or "").strip()
        out_rel = (r.get("output_relpath") or "").strip()
        seed_s = (r.get("seed") or "").strip()

        if not text or not out_rel:
            print(f"SKIP {sample_id}: missing text/output_relpath")
            fail += 1
            continue

        try:
            seed = int(seed_s) if seed_s else 0
        except ValueError:
            seed = 0

        out_path = PROJECT_DIR / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"SKIP {out_path} (exists)")
            ok += 1
            continue

        try:
            torch.manual_seed(seed)
            chunks = []
            for res in pipeline(text, voice=VOICE, split_pattern=None):
                audio = res.audio if hasattr(res, "audio") else res
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                chunks.append(audio)
            if not chunks:
                raise RuntimeError("no audio chunks returned")

            y = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            sf.write(str(out_path), y, SR)
            ok += 1
            print(f"OK  {sample_id} -> {out_rel}")
        except Exception as e:
            fail += 1
            print(f"FAIL {sample_id}: {type(e).__name__}: {str(e)[:160]}")

    print(f"Done. ok={ok} fail={fail} out_dir=outputs/kokoro")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
