# Prosody Transfer Benchmark

**Blog post:** [how well does voice cloning preserve pitch and prosody?](https://dav1lex.titancode.pl/blog/prosody-transfer-benchmark)

Measure how well voice-cloning TTS models preserve speaker-level prosody (F0 characteristics) from reference audio.

## Reference Pairs (hardcoded — do not re-select)

| Pair | Breathy | Modal | Sentence |
|------|---------|-------|----------|
| 001 | p240 (s002) | p229 (s002) | "Ask her to bring these things with her from the store." |
| 002 | p253 (s005) | p301 (s005) | "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station." |
| 003 | p264 (s006) | p282 (s006) | "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow." |

Source: VCTK Corpus 0.92 

## Pipeline

```
prepare_references.py   → 6 VCTK FLACs → 16kHz mono WAV in references/
generate_[model].py      → 6 WAVs per model in outputs/{chatterbox,xtts,kokoro}/
extract_features.py      → F0 mean/std/range, speaking rate, F0 contour .npy
gate_check.py            → Cohen's d ≥ 0.5 on F0 mean OR F0 range
analyze.py               → DTW distance + stat preservation + contrast retention
make_report.py           → HTML report
```

## Models Tested

- **Chatterbox** — voice cloning via `ChatterboxTTS`
- **XTTS-v2** — voice cloning via Coqui TTS
- **Kokoro** — fixed default voice (af_bella), no voice cloning

## Scoring

```
score = (stat_scaled_abs_delta + dtw_distance) / 2 − mean_contrast_retention
```

Lower is better.

- `stat_scaled_abs_delta`: mean of 4 per-metric `abs(delta) / robust_scale(reference_values)`
- `dtw_distance`: normalized DTW between reference and output F0 contours (voiced frames only)
- `mean_contrast_retention`: mean of clamped ratio `output_contrast / reference_contrast`, clamped to [0, 1]

## Gate

Passes if Cohen's d ≥ 0.5 on F0 mean **or** F0 range between breathy and modal reference groups.

## Results

| Model | Score | DTW ↓ | Stat Δ ↓ | Contrast Retention ↑ |
|-------|-------|-------|----------|-------------------|
| Chatterbox | **6.09** | 11.27 | 1.88 | 0.48 |
| XTTS-v2 | **6.69** | 13.20 | 1.76 | 0.79 |
| Kokoro | **9.05** | 16.41 | 1.69 | 0.00 |

## Perceptual Check

**Chatterbox** and **XTTS** outputs are recognizable as the reference speaker. Chatterbox loses breathiness texture while preserving pitch trajectory. XTTS preserves speaker character with slight pitch elevation. **Kokoro** outputs are perceptually distinct from all references.

## Caveats

- **F0 estimation** is sensitive to microphone, noise, and phonetic content.
- **Speaking rate** estimation via RMS peak counting is a proxy, not ground truth.
- **DTW alignment** may be affected by durational differences between reference and output.
- **Kokoro** does not support voice cloning — its outputs reflect the default voice (af_bella), not the reference speaker.
- **Fragile contrast ratios near zero:** Contrast retention ratios use reference contrast as denominator. When reference contrast is near zero, the ratio becomes unstable. Pair 001 f0_range has reference contrast = 1.52 Hz — effectively zero — producing a ratio of −26.56 for Chatterbox. Clamping to [0, 1] neutralizes the damage, but this pair contributes no f0_range signal to contrast retention scores. This is expected: VCTK pairs were selected for breathiness contrast, not F0 range contrast. Some metrics will have near-zero reference contrasts by coincidence.
