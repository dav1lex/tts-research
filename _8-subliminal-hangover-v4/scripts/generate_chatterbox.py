#!/usr/bin/env python3
"""Generate V4 utterances with Chatterbox from manifest.csv.

Reads manifest rows where model == "chatterbox", synthesizes `text` as a single context window,
and writes WAVs to the path in `output_relpath`.
"""

from __future__ import annotations

import argparse
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

REF_AUDIO = PROJECT_DIR.parent / "_3-prosody-transfer-benchmark" / "references" / "modal_p229_002.wav"

SR = 24000
SEED_BASE = 42
SEED_OFFSETS = {
    # Mirror _7 naming: numbers == "subliminal"
    "number": 100,
    "noun": 200,
}


def main() -> int:
    # Avoid accidental network touches during generation.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required (torch.cuda.is_available() is False). Refusing to run on CPU.", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--exaggeration", type=float, default=0.3)
    args = ap.parse_args()

    if not MANIFEST_CSV.exists():
        print(f"ERROR: missing {MANIFEST_CSV}", file=sys.stderr)
        return 2
    if not REF_AUDIO.exists():
        print(f"ERROR: missing reference audio {REF_AUDIO}", file=sys.stderr)
        return 2

    # Monkey-patch broken perth watermarker (Chatterbox quirk).
    import perth

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker

    from chatterbox.tts import ChatterboxTTS

    device = "cuda"
    model = ChatterboxTTS.from_pretrained(device=device)

    with MANIFEST_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cb_rows = [r for r in rows if (r.get("model") or "").strip() == "chatterbox"]
    if not cb_rows:
        print("No chatterbox rows found in manifest.csv")
        return 0

    ok = 0
    fail = 0
    for r in cb_rows:
        sample_id = r["id"]
        text = (r.get("text") or "").strip()
        out_rel = (r.get("output_relpath") or "").strip()
        cond = (r.get("condition") or "").strip()
        rep_s = (r.get("repetition") or "").strip()

        if not text or not out_rel:
            print(f"SKIP {sample_id}: missing text/output_relpath")
            fail += 1
            continue

        # Match _7 seed schedule exactly:
        #   seed = 42 + condition_offset + run_index
        try:
            rep = int(rep_s) if rep_s else 0
        except ValueError:
            rep = 0
        seed_off = SEED_OFFSETS.get(cond, 0)
        seed = SEED_BASE + seed_off + rep

        out_path = PROJECT_DIR / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"SKIP {out_path} (exists)")
            ok += 1
            continue

        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            wav = model.generate(
                text=text,
                audio_prompt_path=str(REF_AUDIO),
                exaggeration=float(args.exaggeration),
            )
            if wav.ndim == 2:
                wav = wav.squeeze(0)
            y = wav.detach().cpu().numpy()
            if y.ndim > 1:
                y = np.mean(y, axis=0)

            sf.write(str(out_path), y, SR)
            ok += 1
            print(f"OK  {sample_id} -> {out_rel}")
            torch.cuda.empty_cache()
        except Exception as e:
            fail += 1
            print(f"FAIL {sample_id}: {type(e).__name__}: {str(e)[:160]}")

    print(f"Done. ok={ok} fail={fail} out_dir=outputs/chatterbox")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
