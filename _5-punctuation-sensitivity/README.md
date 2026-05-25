# Punctuation Sensitivity Benchmark

**Goal:** Do TTS models respect punctuation — producing different prosody for periods, questions, exclamations, commas, dashes, ellipses, quotation marks, and capitalization?

**Status:** Gate passed. Analysis complete. 3 models (F5-TTS excluded — see caveats).

## What we did

1. **Designed 28 test utterances** across 5 categories: sentence-end (period/!/?), pause hierarchy (comma/em-dash/ellipsis/semicolon), quotation (reported vs quoted speech), trailing (ellipsis vs period), capitalization (ALL-CAPS vs normal vs lowercase).

2. **Generated audio** with 3 models: Chatterbox, XTTS-v2, Kokoro (no-adaptation baseline). Same VCTK p229 reference audio as papers 2-4.

3. **Extracted pause metrics**: energy-based VAD pause detection, terminal F0 slope (pyin), amplitude decay rate, RMS, overall F0 statistics.

4. **Gate check**: all 3 models produce period pauses ≥150ms (100% pass rate).

## Key Results

| Model | Question F0 Rise | Pause Hierarchy | Quotation Shift | Trailing | Best At |
|-------|-----------------|-----------------|-----------------|----------|---------|
| **Chatterbox** | **+68.5 Hz** (best) | 0.67 | **YES** | Wrong direction | F0-based cues |
| **XTTS-v2** | −5.0 Hz (none) | **1.00** (perfect) | NO | Correct direction | Pause timing |
| **Kokoro** | +25.7 Hz | 0.67 | NO | Wrong direction | (baseline) |

**The split**: XTTS dominates temporal cues (pause hierarchy, trailing ellipsis detection) but ignores terminal F0. Chatterbox dominates F0-based cues (question rise, quotation prosody shift) but has weak pause ordering. Kokoro sits between them on F0 but tracks Chatterbox on pause. No model does both well — F0 sensitivity and pause ordering appear to be independent abilities.

## Caveats

- Energy-based VAD is crude — 30ms threshold, no forced alignment
- F0 extraction (pyin) noisy on very short utterances (c01-c03)
- 2 items per subcategory — increase for publication
- Kokoro has no voice cloning, included as "no-adaptation baseline"
- Quotation detection uses F0 range shift >10Hz threshold (arbitrary)
- **F5-TTS excluded**: its architecture concatenates ref_text + gen_text internally, causing reference-text bleed into output. Not fixable within this benchmark paradigm.

## Outputs

- `results/analysis.json` — full per-model per-category analysis
- `results/gate_check.json` — gate results
- `results/features/pause_features.csv` — 84 rows of extracted metrics
- `results/sensitivity_scores.json` — overall scores
- `outputs/` — 84 generated WAV files (28 per model)
- `data/test_corpus.csv` — test utterance definitions

## Run yourself

```bash
# Generation
../venvs/py312/bin/python scripts/generate_chatterbox.py
../venvs/py312/bin/python scripts/generate_kokoro.py
../venvs/py310/bin/python scripts/generate_xtts.py

# Analysis
../venvs/py312/bin/python scripts/extract_pauses.py
../venvs/py312/bin/python scripts/gate_check.py
../venvs/py312/bin/python scripts/analyze.py
../venvs/py312/bin/python scripts/plot_figures.py
../venvs/py312/bin/python scripts/make_report.py
```
