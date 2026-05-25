# Punctuation Sensitivity Benchmark

**Goal:** Do TTS models respect punctuation — producing different prosody for periods, questions, exclamations, commas, dashes, ellipses, quotation marks, and capitalization?

**Status:** Gate passed. Analysis complete.

## What we did

1. **Designed 28 test utterances** across 5 categories: sentence-end (period/!/?), pause hierarchy (comma/em-dash/ellipsis/semicolon), quotation (reported vs quoted speech), trailing (ellipsis vs period), capitalization (ALL-CAPS vs normal vs lowercase).

2. **Generated audio** with 4 models: Chatterbox, XTTS-v2, Kokoro, F5-TTS. Same VCTK p229 reference audio as papers 2-4.

3. **Extracted pause metrics**: energy-based VAD pause detection, terminal F0 slope (pyin), amplitude decay rate, RMS, overall F0 statistics.

4. **Gate check**: all 4 models produce period pauses ≥150ms (100% pass rate).

## Key Results

| Model | Question F0 Rise | Pause Hierarchy | Quotation Shift | Trailing | Best At |
|-------|-----------------|-----------------|-----------------|----------|---------|
| **Chatterbox** | +68.5 Hz (best) | 0.67 | YES | Wrong direction | F0-based cues |
| **XTTS-v2** | -5.0 Hz (none) | **1.00** (perfect) | NO | **YES** (correct) | Pause timing |
| **F5-TTS** | +31.8 Hz | 0.33 (inverted) | YES | No diff | Raw pauses |
| **Kokoro** | +25.7 Hz | 0.67 | NO | Wrong direction | (baseline) |

**The split**: XTTS dominates temporal cues (pause hierarchy, trailing ellipsis detection) but ignores terminal F0. Chatterbox dominates F0-based cues (question rise, quotation prosody shift) but has weak pause hierarchy. No model does both well.

## Caveats

- Energy-based VAD is crude — 30ms threshold, no forced alignment
- F0 extraction (pyin) noisy on very short utterances (c01-c03)
- 2 items per subcategory — increase for publication
- Kokoro has no voice cloning, included as "no-adaptation baseline"
- F5-TTS generates excessive internal pauses (28+ for some sentences), inflating counts
- Quotation detection uses F0 range shift >10Hz threshold (arbitrary)

## Outputs

- `results/analysis.json` — full per-model per-category analysis
- `results/gate_check.json` — gate results
- `results/features/pause_features.csv` — 112 rows of extracted metrics
- `results/sensitivity_scores.json` — overall scores
- `outputs/` — 112 generated WAV files (28 per model)
- `data/test_corpus.csv` — test utterance definitions

## Run yourself

```bash
# Generation
../venvs/py312/bin/python scripts/generate_chatterbox.py
../venvs/py312/bin/python scripts/generate_kokoro.py
../venvs/py310/bin/python scripts/generate_xtts.py
../venvs/f5tts/bin/python scripts/generate_f5tts.py

# Analysis
../venvs/py312/bin/python scripts/extract_pauses.py
../venvs/py312/bin/python scripts/gate_check.py
../venvs/py312/bin/python scripts/analyze.py
```
