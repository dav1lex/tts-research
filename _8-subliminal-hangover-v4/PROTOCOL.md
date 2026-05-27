# V4 Protocol (Preregistered)

Date: 2026-05-27

This protocol must be finalized before generating any new audio in `_8-subliminal-hangover-v4`.

## 1. Research Question

Does a robotic/list-like prime (numbers) suppress pitch variation in a subsequent emotional target sentence,
relative to a length-matched neutral noun-list prime, when generated in the same model context window?

## 2. Hypotheses

Primary hypothesis (per model):

- `f0_cv(target | noun_prime) > f0_cv(target | number_prime)`

Secondary hypothesis (per model):

- `speaking_rate(target | number_prime) > speaking_rate(target | noun_prime)` (tempo acceleration phenotype)

## 3. Design

Models:

- chatterbox
- xtts
- kokoro (included only if repetition is non-deterministic; see Section 8)

Conditions (main):

- noun: length-matched neutral noun list
- number: number list (robotic)

Optional conditions (exploratory, not required for primary claim):

- control: neutral prose sentence prime

Targets:

- `N_TARGETS = 10` emotional target sentences.
- Targets are fixed in `prompts/targets.json` before generation.

Repetitions:

- `N_REPS = 10` paired repetitions per (model, condition, target).
- Pairing is by `repetition` index: for each `target_id` and `rep`, generate both noun and number conditions.
- **Seed independence requirement:** noun vs number generations must not share the same PRNG seed. Use condition-specific
  seed offsets so paired runs are comparable but not artificially correlated (seed coupling can mask or flip effects).

Total sample size (main study):

- `3 models × 2 conditions × 10 targets × 10 reps = 600` WAVs.

Staging:

- Stage A pilot: `3 × 2 × 10 × 2 = 120` WAVs.
- Stage B full: expand to reps=10.

## 4. Stimuli

Targets:

- Requirements: avoid commas/semicolons; keep roughly similar syllable counts; avoid theatrical phrasing.
- Each target has: `id`, `text`, `emotion_tag`, `syllables` (best-effort), `notes`.

Primes:

- Noun primes: comma-separated neutral nouns, variants matched to number primes by item count and punctuation.
- Number primes: comma-separated numbers; specify whether digits, spelled-out, or mixed. (Lock choice here.)

## 5. Primary Metric

- `f0_cv = f0_std / f0_mean` measured on the aligned target segment only.

## 6. Segmentation / Alignment

- Use WhisperX word alignment to extract the target segment (first target word to last target word).
- Exclusion rule: if alignment cannot locate target word timestamps, the sample is excluded (logged).

## 7. Analysis Plan

Per-model primary analysis:

- Mixed effects model (Python `statsmodels` MixedLM or equivalent):
  - `f0_cv ~ condition + (1 | target_id) + (1 | rep)`

Report:

- condition effect size + confidence interval
- p-value (descriptive; interpret with effect size)
- per-target distribution and leave-one-target-out robustness

Secondary analyses:

- speaking_rate mixed model with the same structure
- descriptive comparisons for f0_std, f0_mean, energy_std

Exploratory:

- interactions: `condition × emotion_tag` (only if pre-specified here)

Multiple comparisons:

- Primary claim is per model for `f0_cv`.
- Secondary tests are labeled exploratory unless corrected.

## 8. Kokoro Determinism Check (Gate)

Before including Kokoro in Stage A:

- Run a determinism preflight (same prompts, same settings, multiple reps).
- If outputs are near-identical across reps (duration, f0_cv, audio hash), treat Kokoro as:
  - either excluded from the main replication, or
  - included with a modified repetition strategy (explicit seed control or prompt perturbation), documented here.

## 9. Exclusion Rules (Pre-Registered)

- Missing alignment timestamps for target boundaries.
- Target segment duration < 0.3s.
- Unvoiced F0 too low or implausible pitch (use thresholds from `_7` unless updated here).

## 10. Provenance

- All generations come from `manifest.csv`.
- No ad hoc runs are allowed in the analysis folder.
