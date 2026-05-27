#!/usr/bin/env python3
"""Generate V4 utterances with XTTS-v2 from manifest.csv.

Reads manifest rows where model == "xtts", synthesizes `text` as a single context window,
and writes WAVs to the path in `output_relpath`.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MANIFEST_CSV = PROJECT_DIR / "manifest.csv"

REF_AUDIO = PROJECT_DIR.parent / "_3-prosody-transfer-benchmark" / "references" / "modal_p229_002.wav"


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required (torch.cuda.is_available() is False). Refusing to run on CPU.", file=sys.stderr)
        return 2

    if not MANIFEST_CSV.exists():
        print(f"ERROR: missing {MANIFEST_CSV}", file=sys.stderr)
        return 2
    if not REF_AUDIO.exists():
        print(f"ERROR: missing reference audio {REF_AUDIO}", file=sys.stderr)
        return 2

    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    tts.to("cuda")

    with MANIFEST_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    xtts_rows = [r for r in rows if (r.get("model") or "").strip() == "xtts"]
    if not xtts_rows:
        print("No xtts rows found in manifest.csv")
        return 0

    ok = 0
    fail = 0
    for r in xtts_rows:
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
            torch.cuda.manual_seed(seed)
            tts.tts_to_file(
                text=text,
                speaker_wav=str(REF_AUDIO),
                language="en",
                file_path=str(out_path),
            )
            ok += 1
            print(f"OK  {sample_id} -> {out_rel}")
        except Exception as e:
            fail += 1
            print(f"FAIL {sample_id}: {type(e).__name__}: {str(e)[:160]}")

    torch.cuda.empty_cache()
    print(f"Done. ok={ok} fail={fail} out_dir=outputs/xtts")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
