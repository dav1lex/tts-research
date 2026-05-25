# Identity Drift Benchmark

**Goal:** Does TTS voice identity drift over 5-minute synthesized monologues compared to a short reference clip?

**Status:** Gate passed. Drift measured. Report + PDF generated.

## What we did

1. **Generated 5-minute monologues** from 3 models (Chatterbox, XTTS-v2, Kokoro) using an 868-word narrative text. Same reference audio as papers 2 and 3 (VCTK p229, neutral_p229_002.wav).

2. **Sliced into 15-second windows** (60 total: chatterbox=19, xtts=21, kokoro=20) and extracted acoustic features per window: F0 (pyin), breathiness (Praat CPP, copied from Paper 2), spectral features, MFCC, RMS, centroid.

3. **Gate check:** Reference audio quality validated (voiced ratio >= 10%, no clipping). Per-model generation quality: >= 50% of windows must pass voicing threshold. All 3 models passed.

4. **Measured drift** as IQR-scaled Euclidean distance from reference per window. Computed drift_increase (first 5 vs last 5 windows) and drift_slope (linear regression across all windows).

5. **Single-listener sanity check:** Compared start vs end segments. Acoustic drift measurable but below perceptual threshold at 5 minutes.

## Results

| Model | Drift Mean | Drift Increase | Drift Slope | Notes |
|-------|-----------|----------------|-------------|-------|
| Chatterbox | 0.491 | +0.080 | +0.00422 | Closest to reference, drifts fastest |
| XTTS-v2 | 0.583 | +0.076 | +0.00431 | More variable, RMS instability |
| Kokoro | 1.290 | +0.070 | +0.00473 | No-adaptation baseline, different voice |

Key: Breathiness (CPP) is the least stable channel across all voice-cloning models. Drift rates near-identical (~0.004/window). 5 minutes insufficient for perceptible drift.

## Caveats

- Euclidean distance in feature space is an acoustic proxy, not perceptual
- Single homogeneous narrative text may conceal drift that would emerge under content variation
- Single-listener check only, no formal perceptual validation
- Chatterbox generation chunked (10 x ~100 words) due to CUDA context limit
- XTTS-v2 had 3 clipped windows (documented, included in analysis)
- 5 minutes may be too short for significant drift
- IQR scaling pooled across all models rather than reference-only

## Outputs

- `results/report.html` -- full 10-page report
- `results/report.pdf` -- PDF export
- `results/drift_summary.csv` -- per-model scores
- `results/drift_by_window.csv` -- per-window drift (60 rows)
- `results/features/window_features.csv` -- all features per window
- `results/figures/` -- 3 plots (drift over time, heatmap, early vs late)
- `data/listening_test/` -- 15s segments for perceptual check

## Run yourself

```bash
# Generation (needs 2 venvs: Python 3.12 for Chatterbox/Kokoro, 3.10 for XTTS)
python3 scripts/generate_chatterbox.py
python3 scripts/generate_kokoro.py
# Switch to XTTS venv
python3 scripts/generate_xtts.py

# Analysis (Python 3.12 venv)
python3 scripts/extract_windows.py
python3 scripts/gate_check.py
python3 scripts/measure_drift.py
python3 scripts/plot_drift.py
python3 scripts/make_report.py
```

## Reference
VCTK Corpus 0.92: https://datashare.ed.ac.uk/handle/10283/3443
