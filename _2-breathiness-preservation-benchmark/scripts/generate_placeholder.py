#!/usr/bin/env python3
"""Generate crude smoke-test WAVs. Do not use these as scientific fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def synth(duration: float, breathiness: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration * SR)) / SR
    f0 = 180.0
    harmonics = sum((1 / n) * np.sin(2 * np.pi * f0 * n * time) for n in range(1, 7))
    harmonics *= 1.0 - 0.65 * breathiness
    noise = rng.normal(0.0, 0.18 * breathiness, len(time))
    envelope = np.sin(np.linspace(0, np.pi, len(time))) ** 0.25
    audio = (harmonics + noise) * envelope
    peak = np.max(np.abs(audio))
    return audio / peak * 0.85 if peak > 0 else audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate non-scientific placeholder WAV")
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--breathiness", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, synth(args.duration, args.breathiness, args.seed), SR)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
