# Subliminal Hangover: The Story

## 1. The Question

TTS models don't process sentences in isolation. They maintain acoustic context across their generation window — the model "remembers" how it was speaking before, and that memory bleeds forward. This isn't speculation; every autoregressive or streaming TTS pipeline carries forward some representation of what it just said, whether it's an encoder state, a style embedding, or raw acoustic conditioning.

The question is simple: **if you feed a TTS model a monotone, robotic prime (a list of numbers), does the flatness bleed into a subsequent emotional sentence?**

We called this the "subliminal hangover" — not because it's truly subliminal (the prime is fully audible), but because it's an unintended context effect that the user didn't ask for and the model wasn't designed to produce. If real, it means that **what you concatenate in a TTS prompt matters for prosody**, even when the two segments are semantically unrelated.

The claim: `f0_cv(emotional target | number prime) < f0_cv(emotional target | noun prime)`, with speaking rate held constant so we know it's pitch, not tempo.

## 2. The Pilot (_7)

### What We Built

Single emotional target sentence: _"You absolutely cannot be serious about this ridiculous idea!"_ Three conditions per model, 5 repetitions each (5 shuffled prime variants per condition). 45 WAVs total across Chatterbox, XTTS-v2, and Kokoro.

| Condition | Prime | Purpose |
|-----------|-------|---------|
| Control | Neutral prose sentence | Baseline (not used in final analysis) |
| Noun | 14 length-matched nouns | Controls for "time on task" and attention drift |
| Number | 14 digit-string numbers | The robotic prime |

Primes and target were generated as a single string in one context window. Target extraction used **WhisperX V3 word alignment** — we found the timestamps of the target's first and last word in the full audio and sliced out only the target segment for analysis. This guaranteed we measured exactly the same words every time, regardless of how the preceding prime affected speech rhythm.

The primary metric was **f0_cv** (coefficient of variation: `f0_std / f0_mean`), extracted on the aligned target segment using parselmouth (a Praat Python binding). f0_cv normalizes pitch variance by the speaker's baseline, making it comparable across models with different pitch ranges. Secondary metric: speaking rate (syllables / aligned duration).

Statistical tests were paired Wilcoxon signed-rank (target-number vs target-noun pairs, matched by prime variant).

### What We Found

| Model | f0_cv drop | p | Rate stable |
|-------|-----------|----|-------------|
| Chatterbox | -39% | 0.031 | yes |
| XTTS-v2 | -29% | 0.094 | yes |
| Kokoro | -13% | 0.22 | borderline |

Chatterbox showed a clean, significant hangover. The robotic number prime suppressed pitch variation by nearly 40% compared to the length-matched noun control, and speaking rate didn't change. This wasn't a tempo artifact — the model genuinely flattened its prosody after reading numbers.

Two distinct acoustic phenotypes emerged:

- **Pitch compression**: reduced f0_cv after robotic primes (Chatterbox, XTTS-v2 trend)
- **Tempo acceleration**: increased speaking rate after robotic primes (Kokoro borderline)

### Methodology Notes

The V3 pipeline (as we called it) had real strengths. WhisperX alignment was rigorous — 44 of 45 files aligned successfully. The length-matched noun baseline was a genuine improvement over earlier versions that used a short nature sentence control (which introduced too many confounds: duration differences, prosodic style, and simple attention decay). Using f0_cv instead of raw f0_std normalized out between-model pitch range differences.

But the weaknesses were fundamental, not cosmetic.

### What Was Wrong

1. **n=5 with Wilcoxon.** Five repetitions per condition is enough to *notice* an effect but not enough to *measure* one. Wilcoxon at n=5 has almost no power — it can only return significant p-values for perfect or near-perfect orderings. The Chatterbox result (p=0.031) was one of only three possible significant outcomes: all 5 pairs showing the same direction, or 4 out of 5 with one tie. This is fragile.

2. **Single target sentence.** Any effect we found could be specific to that one sentence, its syllable structure, its emotional valence, its phonetic composition. You can't generalize from one target.

3. **No pre-registration.** We ran the analysis, saw the results, and then wrote the report. Classic HARKing risk (Hypothesizing After Results are Known). The two-phenotype discovery was genuinely interesting, but we had no way to separate genuine finding from over-interpretation of a small dataset.

4. **Seed handling was undocumented and potentially buggy.** The _7 generation scripts set seeds but didn't document the schedule or guarantee that noun and number conditions for the same target/rep used different seeds. If both conditions shared the same random seed (or seeds that produced correlated noise), the paired comparison would be artificially correlated — potentially masking real effects or creating false positives. We didn't know which until we looked.

5. **Mixed-effects wasn't the analysis.** With 10 sentences and 10 reps in a real replication, you need random effects for target and repetition. Wilcoxon doesn't do that. The _7 analysis treated each condition as a flat list of 5 values per model, ignoring structure.

6. **The Kacper critique.** Kacper Wikiel, whose Zonos emotion channel work inspired this whole line of research, pointed out the fragility: n=5, single sentence, no registration. His point wasn't that the finding was false — it was that the evidence wasn't strong enough to be *convincing*. He was right.

## 3. The Credibility Problem

_7 gave us a signal. That's all it gave us. The Chatterbox result was intriguing, the two-phenotype pattern was surprising, and the methodology was sound enough to justify investing more effort. But no reviewer, no skeptical colleague, and honestly not even ourselves should be fully convinced by n=5 on one sentence.

To make the claim credible, we needed:

- **More targets.** Not one sentence. Enough to estimate a per-target distribution and check robustness across different emotional prosody patterns.
- **More repetitions.** n=5 is descriptive. n=10 starts to approach useful mixed-effects estimation.
- **Pre-registration.** Write down the hypotheses, the exclusion rules, the analysis plan, and the primary metric *before* generating any audio. Lock it. Then execute.
- **Seed independence.** Guarantee that paired noun/number runs don't share the same PRNG seed. If the model uses a deterministic schedule (seed = some base + condition offset + repetition), document it and verify.
- **Mixed effects, not Wilcoxon.** Random intercepts for target_id and repetition. Effect sizes with confidence intervals. P-values as descriptive, not confirmatory with correction.
- **Kokoro determinism check.** Kokoro might be fully deterministic (same prompt, same output). If it is, you can't run a reproducibility experiment — every "repetition" is the same file. We needed to measure this before including Kokoro in the design.

## 4. The Replication (_8)

### Design

`_8-subliminal-hangover-v4` locked the protocol in `PROTOCOL.md` before generation:

- **10 emotional target sentences** (angry/indignant style, 5–18 syllables each), fixed in `prompts/targets.json`
- **10 repetitions** per (model, condition, target)
- **3 models**: Chatterbox, Kokoro (after determinism preflight), XTTS-v2
- **2 conditions**: noun (length-matched neutral list) and number (digit-string list), plus optional control
- **Total**: 600 WAVs (3 × 2 × 10 × 10)
- **Staging**: Stage A pilot at 2 reps (120 WAVs) to smoke-test the pipeline. Stage B full expansion to 10 reps.

### What Changed from _7

**Manifest-driven generation.** Instead of hardcoded generation scripts that loop internally, every single audio file is a row in `manifest.csv` with its full text, condition, target_id, seed, and output path. Generation scripts read the manifest and produce exactly what it says — no ad-hoc parameter choices, no "oh I ran this one differently". The manifest *is* the experimental design, and it can be validated analytically (checking seed distributions, condition balance, target coverage).

**Seed coupling prevention.** This was the bug we discovered mid-build. The manifest generator assigns condition-specific seed offsets: `number` gets offset +100, `noun` gets +200, both added to a base seed + repetition index. This guarantees that the paired noun/number runs for the same target and repetition use *different* seeds. In _7, the generation scripts likely used the same seed or seeds within a correlated range, which could artificially inflate or deflate paired differences. We caught this because `analyze_mixedlm.py` includes a seed coupling check — scanning the manifest for any (target_id, repetition) pair where noun and number share the same seed. The check passed (0/100 identical pairs), but building the check forced us to think about the problem.

**Kokoro determinism preflight.** Before including Kokoro in the main design, we ran a preflight: 5 repetitions of the same prime+target text, each with a different seed. The result: all 5 WAVs had different SHA-256 hashes, different f0_cv values (range 0.195–0.211, std=0.005), but identical durations (19.65s exactly). The model is near-deterministic — acoustic parameters vary slightly across seeds, but the timing and high-level prosody don't. This is enough variation to include in the main study, but it means the per-model power is lower for Kokoro because between-rep variability is small. In effect, each Kokoro repetition provides less independent information than a Chatterbox repetition.

**Append-mode feature extraction.** The `extract_features.py` pipeline checks `features.csv` for existing rows by ID and skips already-processed files. This sounds trivial but it was crucial for the real workflow: we generated audio in batches across multiple days, and restarting 600 WhisperX alignments from scratch every time would have been hours of wasted GPU time. The append mode meant we could generate a batch of WAVs for one model, extract features, generate another batch, extract more — and the CSV just grows without losing work.

**Mixed-effects analysis.** Instead of Wilcoxon, we fit per-model random-intercept models: `f0_cv ~ condition_num + (1 | target_id) + (1 | repetition)` (and similarly for speaking_rate). This accounts for target-level and repetition-level variability. Effect size is the condition_num coefficient with confidence intervals from `statsmodels.MixedLM`.

### Discoveries During Build

1. **The Chatterbox `PerthImplicitWatermarker` monkey-patch.** Chatterbox's internal watermarker module crashes on import in certain environments. Our generation script monkey-patches `perth.PerthImplicitWatermarker` to a dummy before importing the TTS class. Without this, half the pipeline fails before it starts.

2. **WhisperX on CPU is impossible at scale.** The CUDA requirement in the PROTOCOL isn't optional — WhisperX alignment at 600 files takes minutes on GPU and days on CPU. The scripts enforce `torch.cuda.is_available()` with a hard exit.

3. **Gate checks catch real problems.** 593 of 594 generated files passed gate (f0 in range, speaking rate in range, duration above minimum). The one failure was XTTS producing an implausibly slow reading (0.65 syllables/second) — likely a model artifact or an alignment glitch, not a systematic issue. But having the gate means we don't include garbage in the analysis.

## 5. The Result

### Primary Finding: Tempo Acceleration

`s peaking_rate ~ condition_num`, mixed effects, N=594:

| Model | Coef (syllables/s) | 95% CI | p |
|-------|-------------------|--------|---|
| Chatterbox | +0.352 | [+0.103, +0.601] | 0.006 |
| Kokoro | +1.356 | [+1.157, +1.555] | 1.5e-40 |
| XTTS | +0.704 | [+0.391, +1.016] | 1e-05 |

All three models show a statistically significant tempo increase after robotic primes. All confidence intervals exclude zero. This is not ambiguous.

### Secondary Finding: Pitch Compression

`f0_cv ~ condition_num`, mixed effects:

| Model | Coef | 95% CI | p | Verdict |
|-------|------|--------|---|---------|
| Kokoro | -0.045 | [-0.061, -0.029] | 2.1e-08 | Confirmed |
| Chatterbox | -0.006 | [-0.031, +0.020] | 0.668 | Null (unconverged) |
| XTTS | -0.004 | [-0.030, +0.022] | 0.762 | Null |

Kokoro shows a clear, significant f0_cv drop — the only model that does. Chatterbox and XTTS show nothing.

But there's a caveat: the Chatterbox f0_cv model failed to converge. The optimizer hit the boundary of the parameter space and gave up. The coefficient (-0.006) and p-value (0.668) should be treated as unreliable — we don't know whether the real effect is null, small, or the optimizer just couldn't find it.

### What Changed from _7

The V3 pilot found pitch compression in Chatterbox and XTTS, and tempo acceleration in Kokoro. V4 flipped this: **tempo acceleration is universal, pitch compression is Kokoro-specific.**

What happened to Chatterbox's -39% f0_cv effect from _7?

Three possibilities, not mutually exclusive:

1. **It was real but target-specific.** The single _7 target sentence ("You absolutely cannot be serious...") might have been unusually sensitive to the hangover. With 10 diverse targets averaging across different prosodic profiles, the per-target effect washes out.

2. **The _7 Chatterbox result was a false positive inflated by seed coupling.** If noun and number conditions shared a seed, small artifacts could drive a spurious paired difference that looks like an effect at n=5.

3. **The _7 Chatterbox result was genuine but the _8 model failed to detect it due to convergence failure.** A non-converged mixed model is not evidence of absence — it's just a broken measurement instrument.

We can't distinguish these from the data we have. This is the honest answer, and it's why the convergence failure matters: we don't know.

## 6. What It Means

### Robotic Primes Don't Just Flatten Pitch — They Make Models Rush

The strongest, most robust finding is tempo acceleration. This isn't subtle. Kokoro speeds up by over 1.3 syllables per second after reading a list of numbers. XTTS by 0.7. Even Chatterbox, the smallest effect, is +0.35 with a confidence interval that doesn't touch zero.

This means the "hangover" isn't primarily about pitch — it's about rhythm. The robotic prime changes the model's internal sense of pace, and that change persists into the emotional target. This is a different kind of acoustic inertia than we hypothesized.

### Pitch Compression is a Model Architecture Trait

Only Kokoro compressed f0_cv. Why? Kokoro doesn't do voice cloning — it uses fixed speaker embeddings. Its prosodic control may be more globally coupled than the cloning-based architectures (Chatterbox and XTTS), meaning a flat prime affects the entire acoustic generation pipeline rather than just timing. Or it could be the opposite: Kokoro's limited prosodic flexibility means it gets "stuck" in the flat mode more easily, while cloning-based models recover faster.

We don't know the mechanism. What we know is that the phenotype exists and it's model-specific.

### Context-Dependent Prosody Is Real and Measurable

The practical implication is straightforward: **don't concatenate acoustically dissimilar text styles in a single TTS generation if prosodic consistency matters.**

If you're building an audiobook pipeline and you generate chapter headings ("Chapter 3") in the same context window as the emotional opening paragraph, you might be unknowingly flattening or accelerating the narration. If you're building a virtual agent that reads product codes and then delivers empathetic customer service, the codes are bleeding into the empathy.

The fix isn't complicated — generate the robotic segment and the emotional segment in separate context windows, stitch the audio afterward. But you have to know to do it.

### The Chatterbox f0_cv Convergence Problem Is a Real Limitation

We can't say whether Chatterbox shows pitch compression or not. The mixed model couldn't fit the data. This is a methodological lesson: convergence checks aren't optional. If the optimizer hits a boundary, the point estimate is not trustworthy regardless of what the p-value says. Future work should try alternative optimizers (Nelder-Mead doesn't use gradients, Powell is derivative-free) or switch to Bayesian estimation with weak priors.

## 7. What's Next

### Perceptual Validation

Everything we've measured is acoustic. We don't know if listeners can hear any of this. A proper ABX listening test — "did these two versions of the same sentence come from the same condition?" — would tell us whether the measured effects cross the perceptual threshold. Without this, the findings are interesting but not actionable from a product perspective.

### Spelled-Out Numbers vs Digit Strings

Is the tempo acceleration driven by the *semantics* of numbers (the concept of counting, the rhythmic pattern of digit strings) or the *surface form* (short tokens, absence of semantic content, a "list-like" prosodic template)? Generating the same primes with spelled-out numbers ("eight hundred forty seven, twenty three, five thousand ninety one...") would isolate whether it's the rhythm or the meaning.

### Multiple Speakers Per Model

A single voice per model tells you about that model with that voice. Replicating across 3–5 diverse speaker embeddings per model would tell you whether the hangover is a model property or a voice property within each model.

### Broader Target Set

All 10 targets were angry/indignant sentences. A broader emotional range (sad, happy, neutral, fearful) would reveal whether the hangover affects all emotions equally or interacts with valence and arousal.

### The Causal Question

These are observational measurements — we prime the model with numbers and observe what happens. The causal question — *which* acoustic parameters propagate across the context window, and through what mechanism — requires a different experimental design, likely involving controlled perturbations of specific model internals. That's a much harder experiment, and it's not clear it's worth it relative to the practical fix of just generating separately.

---

*The subliminal hangover isn't a crisis for TTS. It's a quirk — a measurable, replicable quirk — that tells us something about how these models carry acoustic state forward. The pilot gave us a signal. The replication sharpened it, confirmed some parts, and left one big question mark. That's how it's supposed to work.*
