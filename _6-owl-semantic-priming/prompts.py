#!/usr/bin/env python3
"""Single source of truth for all text used in the semantic priming experiment.

All generation scripts and analyze.py import from here.
"""

from __future__ import annotations

NEUTRAL_SENTENCE = (
    "The quarterly figures were reviewed and submitted before the deadline."
)

PRIME_OWL = (
    "The owl is a masterful nocturnal predator. "
    "Its feathers are specially adapted for silent flight. "
    "Owls can rotate their heads nearly 270 degrees. "
    "The snowy owl inhabits Arctic tundra regions."
)

PRIME_DEATH = (
    "The funeral was held on a grey Tuesday morning. "
    "Mourners gathered in silence around the graveside. "
    "The final rites were spoken quietly into cold air. "
    "Loss settled heavily over everyone present."
)

PRIME_NEUTRAL = (
    "The office was quiet on Wednesday morning. "
    "Staff members worked at their desks as usual. "
    "Phone calls were answered in standard tones. "
    "Routine tasks proceeded without interruption."
)

CONDITIONS = ("cold", "primed_neutral", "primed_owl", "primed_death")
N_REPS = 5
MODELS = ("chatterbox", "kokoro", "xtts")
SEED = 42
TARGET_WORD_COUNT = len(NEUTRAL_SENTENCE.split())

# Reference audio for voice-cloning models (Chatterbox, XTTS).
# Same VCTK p229 speaker used across all prior benchmarks.
# Path relative to the _6 experiment directory.
VCTK_REFERENCE = "../_3-prosody-transfer-benchmark/references/modal_p229_002.wav"

# Kokoro: English voice preset (no reference cloning available).
KOKORO_VOICE = "af_bella"

# Kokoro phoneme chunking ceiling in en_tokenize (hardcoded).
# If phonemized text exceeds 510 characters, it splits at punctuation
# boundaries silently. ~390 raw characters is the rough safety margin
# (English averages ~1.3 phonemes per character).
KOKORO_PHONEME_CEILING = 510
KOKORO_SAFE_CHAR_LIMIT = 390

# Target sample rate for all outputs.
TARGET_SR = 22050


def make_prompt(condition: str, sentence: str | None = None) -> str:
    """Build the full prompt for a given condition.

    Args:
        condition: One of 'cold', 'primed_neutral', 'primed_owl',
            'primed_death'.
        sentence: Override the neutral sentence (defaults to NEUTRAL_SENTENCE).

    Returns:
        The full text to feed into the TTS model.
    """
    if sentence is None:
        sentence = NEUTRAL_SENTENCE

    if condition == "cold":
        return sentence
    elif condition == "primed_neutral":
        return f"{PRIME_NEUTRAL}\n\n{sentence}"
    elif condition == "primed_owl":
        return f"{PRIME_OWL}\n\n{sentence}"
    elif condition == "primed_death":
        return f"{PRIME_DEATH}\n\n{sentence}"
    else:
        raise ValueError(f"Unknown condition: {condition}")


def check_char_limits() -> dict[str, int]:
    """Return character counts for all conditions. For sanity check."""
    return {cond: len(make_prompt(cond)) for cond in CONDITIONS}
