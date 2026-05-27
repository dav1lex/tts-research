#!/usr/bin/env python3
"""
Create manifest.csv for V4 study.

This is intentionally conservative: it refuses to run if prompts contain REPLACE_ME.

Output: manifest.csv at project root.
Columns:
  id,model,voice,condition,target_id,prime_id,repetition,seed,text,output_relpath
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
TARGETS_JSON = PROMPTS_DIR / "targets.json"
PRIMES_JSON = PROMPTS_DIR / "primes.json"
MANIFEST_CSV = PROJECT_DIR / "manifest.csv"


MODELS_DEFAULT = ["chatterbox", "xtts", "kokoro"]
CONDITIONS_DEFAULT = ["noun", "number"]


@dataclass(frozen=True)
class Target:
    id: str
    text: str


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load_json(path: Path) -> dict:
    if not path.exists():
        _die(f"missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_no_placeholders(obj: object, path: str) -> None:
    if isinstance(obj, str):
        if "REPLACE_ME" in obj:
            _die(f"placeholder found in {path}")
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_placeholders(v, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_placeholders(v, f"{path}.{k}")
        return


def _load_targets() -> list[Target]:
    data = _load_json(TARGETS_JSON)
    _assert_no_placeholders(data, "targets.json")
    targets = data.get("targets", [])
    out: list[Target] = []
    for t in targets:
        tid = (t.get("id") or "").strip()
        text = (t.get("text") or "").strip()
        if not tid or not text:
            _die("targets.json contains empty id/text")
        out.append(Target(id=tid, text=text))
    if not out:
        _die("targets.json has no targets")
    return out


def _load_primes() -> dict[str, dict[str, str]]:
    data = _load_json(PRIMES_JSON)
    _assert_no_placeholders(data, "primes.json")

    noun = data.get("noun_primes") or {}
    number = data.get("number_primes") or {}
    if not noun or not number:
        _die("primes.json must include noun_primes and number_primes")
    return {"noun": noun, "number": number}


def _pick_prime_ids(primes: dict[str, str], n_needed: int) -> list[str]:
    keys = list(primes.keys())
    if len(keys) < n_needed:
        _die(f"need at least {n_needed} prime variants, found {len(keys)}")
    return keys[:n_needed]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--stage-a", action="store_true", help="Use reps=2 (sanity pilot).")
    ap.add_argument("--models", type=str, default=",".join(MODELS_DEFAULT))
    ap.add_argument("--conditions", type=str, default=",".join(CONDITIONS_DEFAULT))
    ap.add_argument("--seed-base", type=int, default=1000, help="Optional numeric seed base for determinism control.")
    args = ap.parse_args()

    reps = 2 if args.stage_a else args.reps
    if reps <= 0:
        _die("--reps must be >= 1")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    for c in conditions:
        if c not in ("noun", "number"):
            _die(f"unsupported condition: {c} (only noun,number scaffolded)")

    targets = _load_targets()
    primes_by_condition = _load_primes()

    # Minimum viable expectation: at least as many prime variants as reps,
    # so each rep can use a distinct prime variant (still paired by rep index).
    prime_ids_by_condition: dict[str, list[str]] = {}
    for cond in conditions:
        prime_ids_by_condition[cond] = _pick_prime_ids(primes_by_condition[cond], reps)

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for model in models:
        voice = ""  # model-specific; generators can map default voice if empty
        for target in targets:
            for rep in range(1, reps + 1):
                for cond in conditions:
                    prime_id = prime_ids_by_condition[cond][rep - 1]
                    prime_text = primes_by_condition[cond][prime_id].strip()
                    full_text = f"{prime_text} {target.text}".strip()
                    seed = str(args.seed_base + rep)
                    sample_id = f"{model}_{cond}_{target.id}_r{rep:02d}"
                    out_rel = f"outputs/{model}/{sample_id}.wav"
                    rows.append(
                        {
                            "id": sample_id,
                            "model": model,
                            "voice": voice,
                            "condition": cond,
                            "target_id": target.id,
                            "prime_id": prime_id,
                            "repetition": str(rep),
                            "seed": seed,
                            "text": full_text,
                            "output_relpath": out_rel,
                        }
                    )

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "model",
                "voice",
                "condition",
                "target_id",
                "prime_id",
                "repetition",
                "seed",
                "text",
                "output_relpath",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()

