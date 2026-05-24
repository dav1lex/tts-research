# Breathiness Preservation Benchmark

**Goal:** Does voice-cloning TTS preserve breathy vs neutral voice quality from reference audio?

**Status:** ✅ Gate passed. Rankings valid. Report + PDF generated.

## What we did

1. **Picked 3 female speaker pairs** from VCTK dataset (Southern England, Welsh/American, Scottish/English NE). Same text within each pair. Breathy candidates chosen by lowest CPP+HNR scores, then confirmed by human listening.

2. **Generated audio** with 3 models: Chatterbox, XTTS-v2, Kokoro (each produces 6 WAVs).

3. **Extracted breathiness metrics** using Praat (CPP primary, HNR + spectral tilt supporting).

4. **Gate check:** Reference clips must clearly separate breathy from neutral before any ranking is trusted. Our references pass (CPP d=-4.15, HNR d=-2.03).

## Results

| Rank | Model | Score | Notes |
|------|-------|-------|-------|
| 1 | XTTS-v2 | 0.097 | Best breathiness preservation |
| 2 | Chatterbox | 0.311 | Close on retention, higher absolute error |
| 3 | Kokoro | 1.448 | No voice cloning = zero retention (baseline) |

## Caveats

- CPP and HNR are acoustic proxies, not perception — breathy-sounding doesn't always mean low CPP
- Unmatched accents in pairs 002 and 003 may introduce phonetic confounds
- Only 3 pairs, 1 sentence per pair — increase for publication
- Kokoro has no voice cloning, included as "no adaptation" baseline only
- Reference breathiness labels verified by perceptual listening check

## Outputs

- `results/report.html` — full 7-page report
- `results/report.pdf` — PDF export
- `results/model_rankings.csv` — raw numbers
- `features/features.csv` — all extracted metrics per file

## Run yourself

```bash
source venv-new/bin/activate
python3 scripts/generate_chatterbox.py --metadata metadata.csv --output-dir outputs/chatterbox
python3 scripts/generate_kokoro.py --metadata metadata.csv --output-dir outputs/kokoro
source venv-xtts/bin/activate
python3 scripts/generate_xtts.py --metadata metadata.csv --output-dir outputs/xtts
source venv-new/bin/activate
python3 scripts/extract_features.py --metadata metadata.csv --features-dir features
python3 scripts/gate_check.py --features-dir features --results-dir results
python3 scripts/analyze.py --features-dir features --results-dir results
python3 scripts/make_report.py --results-dir results
```

## Dataset
VCTK Corpus 0.92: https://datashare.ed.ac.uk/handle/10283/3443
