# Breathiness Preservation Benchmark Report

Gate status: **PASSED**

## Gate Metrics

| metric | breathy_mean | neutral_mean | cohens_d | passed |
| --- | --- | --- | --- | --- |
| cpp_mean | 10.3945 | 14.5228 | -4.1514 | 1.0000 |
| hnr_mean | 19.6466 | 21.6700 | -2.0313 | 1.0000 |
| spectral_tilt_mean | -7.3766 | -8.5429 | 0.7561 | 0.0000 |


## Model Rankings

Lower scaled distance is better. Higher contrast retention is better. Lower score is better.

| model | n_outputs | n_pairs | mean_scaled_abs_delta | mean_contrast_retention | score |
| --- | --- | --- | --- | --- | --- |
| xtts | 6.0000 | 3.0000 | 0.6152 | 0.5179 | 0.0973 |
| chatterbox | 6.0000 | 3.0000 | 0.8215 | 0.5104 | 0.3111 |
| kokoro | 6.0000 | 3.0000 | 1.4483 | 0.0000 | 1.4483 |


## Paired Contrast Preservation

| model | pair_id | cpp_mean_contrast_ratio | hnr_mean_contrast_ratio | spectral_tilt_mean_contrast_ratio | mean_contrast_retention |
| --- | --- | --- | --- | --- | --- |
| chatterbox | pair_001 | 0.1461 | 0.1622 | 1.4753 | 0.4361 |
| chatterbox | pair_002 | 0.6280 | -0.0438 | 0.7712 | 0.4664 |
| chatterbox | pair_003 | 0.8857 | -1.2008 | 1.1614 | 0.6286 |
| kokoro | pair_001 | -0.0000 | -0.0000 | 0.0000 | 0.0000 |
| kokoro | pair_002 | -0.0000 | -0.0000 | 0.0000 | 0.0000 |
| kokoro | pair_003 | -0.0000 | -0.0000 | -0.0000 | 0.0000 |
| xtts | pair_001 | 0.1707 | -0.2198 | 1.6260 | 0.3902 |
| xtts | pair_002 | 0.4393 | 0.7259 | 0.6870 | 0.6174 |
| xtts | pair_003 | 0.3535 | 0.6449 | 0.6396 | 0.5460 |


## Method

- Primary metric: Praat CPPS/CPP over voiced intervals.
- Supporting metrics: Praat harmonicity/HNR and voiced-frame spectral tilt.
- Gate: reference clips must separate breathy from neutral before analysis is allowed.
- Scoring: output-reference metric distance uses robust dataset-level reference scales.
- Contrast retention: paired breathy-neutral differences test whether the model preserves the breathiness contrast.

## Caveats

- Objective breathiness proxies are sensitive to microphone, noise, loudness, and phonetic content.
- Use matched text and recording conditions wherever possible.
- Treat rankings as invalid unless the gate passes.
