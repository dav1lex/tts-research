# _6 Owl - Semantic Priming Experiment

**Blog post:** [does prior context leak into tts prosody?](https://dav1lex.titancode.pl/blog/semantic-priming-experiment)

Does pre-pending a semantically charged paragraph (neutral vs owl vs death)
change the prosody of a subsequent neutral sentence? If TTS transformer
self-attention carries context across the full input, we should see measurable
prosodic drift in the target sentence between `cold` and `primed_*` conditions,
and the drift direction should differ by prime content.

## Hypothesis

If `cold != primed_neutral != primed_owl != primed_death`: content-specific
priming.
If `cold != primed_neutral == primed_owl == primed_death`: just context length
or paragraph presence affecting output.
If all four are equal: no measurable priming effect.

## Design

4 conditions per model, 5 repetitions each, 1 fixed neutral test sentence.
60 audio clips total. Base `SEED=42`, with unique condition/repetition seeds.
Generation scripts require CUDA and will not fall back to CPU.

```
cold:            "The quarterly figures were reviewed and submitted before the deadline."
primed_neutral:  NEUTRAL paragraph -> same sentence
primed_owl:      OWL paragraph -> same sentence
primed_death:    DEATH paragraph -> same sentence
```

All text lives in `prompts.py`, which is imported by all scripts.
Feature analysis is valid only on target-sentence segments extracted by
`scripts/segment.py`.

## Models

| Model | Venv | Reference |
|-------|------|-----------|
| Chatterbox | `venvs/py312` | VCTK p229_002 (speaker identity only) |
| Kokoro | `venvs/py312` | `af_bella` voice preset (no reference cloning) |
| XTTS-v2 | `venvs/py310` | VCTK p229_002 (speaker identity only) |

## Folder Structure

```
_6-owl-semantic-priming/
├── prompts.py
├── scripts/
│   ├── gen_chatterbox.py
│   ├── gen_kokoro.py
│   ├── gen_xtts.py
│   ├── segment.py
│   └── analyze.py
├── outputs/
│   ├── chatterbox/{cold,primed_neutral,primed_owl,primed_death}/r1..r5.wav
│   ├── kokoro/{cold,primed_neutral,primed_owl,primed_death}/r1..r5.wav
│   └── xtts/{cold,primed_neutral,primed_owl,primed_death}/r1..r5.wav
├── segments/
│   ├── chatterbox/{cold,primed_neutral,primed_owl,primed_death}/r1..r5.wav
│   ├── kokoro/{cold,primed_neutral,primed_owl,primed_death}/r1..r5.wav
│   └── xtts/{cold,primed_neutral,primed_owl,primed_death}/r1..r5.wav
├── results/
│   ├── segment_qc.csv
│   └── features.csv
└── README.md
```

## How to Run

```bash
# 1. Sanity check (dry-run, no model loading)
cd _6-owl-semantic-priming
../venvs/py312/bin/python scripts/gen_kokoro.py --dry-run

# 2. Generate per model
../venvs/py312/bin/python scripts/gen_chatterbox.py
../venvs/py312/bin/python scripts/gen_kokoro.py
../venvs/py310/bin/python scripts/gen_xtts.py

# 3. Segment target sentence only
../venvs/py312/bin/python scripts/segment.py --clean

# 4. Extract features from segments/
../venvs/py312/bin/python scripts/analyze.py
```

## Critical: Analyze Target Segments Only

The primed WAVs contain both the prime paragraph and the neutral target
sentence. Measuring the full WAV only proves that different full texts sound
different. That is not a priming test.

The valid pipeline is:

1. Generate full `outputs/` clips.
2. Run `scripts/segment.py` to extract only the final neutral target sentence
   into `segments/`.
3. Run `scripts/analyze.py`, which defaults to `segments/`.

`scripts/segment.py` copies cold clips as full target clips. For primed clips,
it finds the target boundary using a VAD/RMS pause detector, cuts with a small
start pad, and writes `results/segment_qc.csv`. Review any rows with QC flags
before interpreting features.

Existing `results/features.csv` files from before this segmentation step are
stale and should not be interpreted.

## Measured Features

| Feature | Tool | Captures |
|---------|------|----------|
| F0 mean | parselmouth (Praat) | Average pitch |
| F0 std | parselmouth | Pitch variance |
| F0 range | parselmouth | Pitch excursion |
| Speech rate | target word count + VAD | Target words/sec in voiced frames |
| Pause count | RMS threshold | Number of silence segments |
| Pause duration mean | RMS threshold | Average pause length |
| RMS energy | librosa | Loudness |
| Spectral centroid | librosa | Voice brightness |

## Critical: Kokoro `split_pattern=None`

Kokoro's `KPipeline` defaults to `split_pattern=r'\n+'`, splitting
multi-paragraph input into independent segments. Each segment is G2P'd and
synthesized in isolation with zero cross-segment state. The `"prime\n\ntarget"`
string would produce two completely independent utterances, killing any priming
test.

The fix is `pipeline(text, voice="af_bella", split_pattern=None)`. This keeps
the full prime + target as one segment, so the transformer self-attention sees
the complete phoneme sequence.

### Side Effect: Prosodic Bleed

With `split_pattern=None`, Kokoro processes the full string as a single prosodic
unit. It may apply sentence-final intonation contours across the prime
paragraphs too, not just the target sentence. This is expected for this design,
but the target-only segmentation step is required before feature analysis.

### 510-Phoneme Chunking Floor

Even with `split_pattern=None`, Kokoro's `en_tokenize` has a hardcoded
510-phoneme ceiling. If exceeded, it splits at a punctuation boundary silently.
The target sentence could end up in its own chunk, losing all priming influence.

Safety check: English averages ~1.3 phonemes per character, so 510 phonemes is
roughly a 390 character proxy. All conditions are below that margin:

- `cold`: 70 chars
- `primed_neutral`: 251 chars
- `primed_owl`: 262 chars
- `primed_death`: 265 chars

## Sanity Check

Before trusting results:

1. Run `gen_kokoro.py --dry-run` and confirm all conditions are below 390 chars.
2. Generate full clips with unique seeds.
3. Run `scripts/segment.py --clean`.
4. Inspect `results/segment_qc.csv`; target segments should usually be
   2.5-5.0s with no QC flags.
5. Only then run `scripts/analyze.py`.

## Reference Audio

Uses VCTK p229_002 (female, Southern England, f0=192.7Hz) from
`_3-prosody-transfer-benchmark/references/modal_p229_002.wav`. Same speaker used
across prior benchmarks (_2 breathiness, _3 prosody transfer, _4 identity drift).

## Notes

- XTTS requires a speaker reference WAV fixed to the same p229 clip for all conditions.
- Chatterbox `exaggeration=0.5` is fixed across all runs.
- Kokoro uses `af_bella` voice preset; no reference cloning available.
- All outputs are standardized to 22050Hz mono WAV before analysis.
- Repetitions use unique seeds: `SEED + condition_index * 100 + rep`.
- Generation scripts default to `--device cuda` and fail if CUDA is unavailable.
