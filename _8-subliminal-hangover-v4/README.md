# _8 Subliminal Hangover (V4 Preregistered Replication)

**Blog post:** [does reading numbers break tts prosody?](https://dav1lex.github.io/blog/subliminal-hangover)

This folder is a continuation of `_7-subliminal-hangover`.

`_7` is the pilot (V3). `_8` is the credibility pass: preregistered, manifest-driven,
multi-target, larger-n replication.

Status: Stage B complete. See:
- [RESULTS.md](RESULTS.md) — Stage B mixed-effects analysis
- [STORY.md](STORY.md) — full research narrative (_7 → _8)
- [results/combined_report.pdf](results/combined_report.pdf) — combined _7 + _8 PDF report with figures

## CUDA Requirement

This V4 workflow is intended to run on GPU only. The `_8` scripts will refuse to run if
`torch.cuda.is_available()` is false.

## Order Of Operations (do not skip)

1. Write/lock `PROTOCOL.md` (preregistration).
2. Finalize `prompts/targets.json` and `prompts/primes.json`.
3. Generate `manifest.csv` via `scripts/make_manifest.py`.
4. (Optional) Run Kokoro determinism preflight first.
5. Generate audio from `manifest.csv`.
6. Align/extract features, then analyze + report.

## Known Limitations

- **Chatterbox f0_cv mixed model**: convergence failures at boundary. f0_cv coefficient for Chatterbox in Stage B is unreliable — treat as null pending re-fit with alternative optimizer or prior.

## Folder Layout

```
_8-subliminal-hangover-v4/
  PROTOCOL.md
  prompts/
    targets.json
    primes.json
  manifest.csv
  scripts/
    make_manifest.py
    validate_manifest.py
    analyze_mixedlm.py
    make_report.py
    generate_chatterbox.py
    generate_xtts.py
    generate_kokoro.py
    extract_features.py
    gate_check.py
  outputs/   # WAVs (not tracked)
  features/
    features.csv
  results/
    report.html
    report.pdf
    stats.json
    gate_check.json
    alignment_log.json
```
