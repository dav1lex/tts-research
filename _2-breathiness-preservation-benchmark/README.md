# Breathiness Preservation Benchmark

Objective benchmark for whether voice-cloning TTS preserves breathiness from reference audio.

This benchmark intentionally treats model outputs as data. It does not hard-wire any TTS model.

## Layout

```text
references/          human reference WAVs
outputs/<model>/     generated WAVs from any TTS system
metadata.csv         manifest tying references, outputs, and breathy/neutral pairs
features/            extracted acoustic metrics
results/             gate result, preservation CSVs, report
scripts/             pipeline code
```

## Metadata Schema

Required columns:

```text
sample_id,pair_id,text,condition,reference_path,output_path,model,seed,notes
```

Rules:

- `condition` must be `breathy` or `neutral`.
- Every `pair_id` must contain one breathy row and one neutral row.
- Paths are resolved relative to the `metadata.csv` file, not the shell working directory.
- `output_path` can point to any model output; model-specific generation stays outside this benchmark.

## Metrics

- Primary: Praat CPPS/CPP over voiced intervals.
- Supporting: Praat harmonicity/HNR and voiced-frame spectral tilt.
- Voiced filtering: low-energy, unvoiced, and low-confidence frames are excluded before metric extraction.

## Gate

The gate checks reference clips only. Analysis is blocked unless:

- CPP separates breathy from neutral in the expected direction with sufficient effect size.
- At least one supporting metric also separates in the expected direction.
- Paired breathy-neutral contrasts are directionally consistent.

This prevents model rankings when the measurement signal is not detectable.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/extract_features.py --metadata metadata.csv --features-dir features
python3 scripts/gate_check.py --features-dir features --results-dir results
python3 scripts/analyze.py --features-dir features --results-dir results
python3 scripts/make_report.py --results-dir results
```

For smoke testing only:

```bash
python3 scripts/generate_placeholder.py references/breathy_001.wav --breathiness 0.85
python3 scripts/generate_placeholder.py references/neutral_001.wav --breathiness 0.10
```

Generated placeholder audio is not valid scientific data.

## Outputs

- `features/features.csv`: one row per reference/output audio file.
- `results/gate_check.json`: metric-level gate decision and effect sizes.
- `results/per_sample_preservation.csv`: output-reference distances.
- `results/paired_contrast_preservation.csv`: breathy-neutral contrast retention by model and pair.
- `results/model_rankings.csv`: model-level summary.
- `results/report.md`: human-readable summary.

## Caveats

CPP, HNR, and spectral tilt are acoustic proxies, not direct perception. Use matched text, matched recording conditions, clean audio, and enough pairs before treating results as meaningful.
