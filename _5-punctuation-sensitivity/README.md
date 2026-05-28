# Punctuation Sensitivity Probe

**Blog post:** [do tts models actually read punctuation?](https://dav1lex.github.io/blog/punctuation-sensitivity-benchmark)

**Goal:** Do TTS models respect punctuation -- producing different prosody for periods, questions, exclamations, commas, dashes, ellipses, quotation marks, and capitalization?

**Status:** Gate passed. Analysis complete. This is a **preliminary smoke test**, not a validated benchmark.

## What we did

1. **Designed 28 test utterances** across 5 categories: sentence-end (period/!/?), pause hierarchy (comma/em-dash/ellipsis/semicolon), quotation (reported vs quoted speech), trailing (ellipsis vs period), capitalization (ALL-CAPS vs normal vs lowercase).

2. **Generated audio** with 3 models: Chatterbox, XTTS-v2, Kokoro (no-adaptation baseline). Same VCTK p229 reference audio as papers 2-4 (except Kokoro, which has no voice cloning).

3. **Extracted pause metrics**: energy-based VAD pause detection, terminal F0 slope (pyin), amplitude decay rate, RMS, overall F0 statistics.

4. **Gate check**: all 3 models produce period pauses >= 150ms (100% pass rate).

## Project Structure

```
_5-punctuation-sensitivity/
  config.yaml              # Single config for all tunable parameters
  data/
    test_corpus.csv        # 28 test utterance definitions
  scripts/
    common.py              # Shared module: paths, constants, utilities
    generate_chatterbox.py # Audio generation (needs py312 venv)
    generate_kokoro.py     # Audio generation (needs py312 venv)
    generate_xtts.py       # Audio generation (needs py310 venv)
    extract_pauses.py      # Feature extraction
    gate_check.py          # Minimum viability check
    analyze.py             # Descriptive analysis with bootstrap CIs
    plot_figures.py        # Figure generation
    make_report.py         # HTML + PDF report
  outputs/
    chatterbox/            # 28 WAV files
    kokoro/                # 28 WAV files
    xtts/                  # 28 WAV files
  results/
    gate_check.json        # Gate pass/fail per model
    analysis.json          # Full analysis with CIs and effect sizes
    sensitivity_scores.json # Overall scores
    text_normalization_log.json  # Text sent vs corpus expected
    features/
      pause_features.csv   # 84 rows of extracted metrics
    figures/               # 6 PNG figures
    report.html            # Self-contained HTML report
    report.pdf             # PDF version
```

## Configuration

All tunable parameters are in `config.yaml`:

```yaml
vad:
  energy_threshold_factor: 0.05   # fraction of max(rms) for silence
  min_pause_ms: 30                # minimum silence to count as pause
  frame_ms: 20                    # RMS frame length
  hop_ms: 10                      # RMS hop length
gate:
  period_min_ms: 150              # min period pause to pass
  pass_rate: 0.80                 # fraction that must pass
f0:
  fmin: 50
  fmax: 600
  frame_length: 2048
  win_length: 1024
  hop_length: 256
quotation:
  range_shift_threshold_hz: 10    # F0 range diff > this = shift detected
analysis:
  terminal_window_ms: 400
  bootstrap_iterations: 1000
```

## Key Results (descriptive only, n=28)

| Model | Question F0 Rise | Pause Hierarchy | Quotation Shift | Notes |
|-------|-----------------|-----------------|-----------------|-------|
| **Chatterbox** | +68.5 Hz/s (d=large) | 0.67 | Flagged | F0-based cues |
| **XTTS-v2** | -5.0 Hz/s (d=negligible) | 1.00 | Not flagged | Pause timing |
| **Kokoro** | +25.7 Hz/s (d=medium) | 0.67 | Not flagged | No-adaptation baseline |

**Descriptive observation**: F0 sensitivity and pause ordering vary independently
in this 3-model, 28-item probe. This is hypothesis-generating, not conclusive.

## Caveats

- **n is small**: 28 utterances, 3 models, many subcategories with 2-6 examples.
- **No perceptual ground truth**: All metrics are acoustic proxies without listening tests.
- **No forced alignment**: Pauses are pooled by RMS threshold, not anchored to punctuation tokens.
- **Crude VAD**: Energy threshold is fragile across models.
- **F0 extraction noisy**: pyin can fail on creaky voice and vocoder artifacts.
- **Arbitrary gate**: 150ms period pause threshold has no empirical justification.
- **Unfair comparison**: Kokoro has no voice cloning (different speaker/prosody prior).
- **No punctuation-stripped controls**: Cannot isolate punctuation effect from tokenizer artifacts.
- **F5-TTS excluded**: Its architecture concatenates ref_text + gen_text internally,
  causing reference-text bleed into output. Not fixable within this benchmark paradigm.

## Run yourself

```bash
# Generation (each in its own venv)
../venvs/py312/bin/python scripts/generate_chatterbox.py
../venvs/py312/bin/python scripts/generate_kokoro.py
../venvs/py310/bin/python scripts/generate_xtts.py

# Analysis (all in py312 venv)
../venvs/py312/bin/python scripts/extract_pauses.py
../venvs/py312/bin/python scripts/gate_check.py
../venvs/py312/bin/python scripts/analyze.py
../venvs/py312/bin/python scripts/plot_figures.py
../venvs/py312/bin/python scripts/make_report.py
```