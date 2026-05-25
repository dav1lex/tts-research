# TTS Research

Independent research on TTS quality evaluation and emotion encoding.

## Projects

### 1. Chatterbox Emotion Channel Analysis
Replication of Kacper Wikiel's Zonos emotion channel methodology on Chatterbox.
Finding: scalar collapse onto pauses. Vector control routes to timbre.
→ _1-Chatterbox-emotiononal-channel-analysis/

- Original Zonos: https://kwikiel.github.io/polish-tts-weekend/

### 2. Breathiness Preservation Benchmark
Objective benchmark measuring whether voice-cloning TTS preserves breathiness from reference audio.
Finding: XTTS-v2 preserves breathiness best (score 0.097), Chatterbox close on retention (0.510), Kokoro baseline zero.
→ _2-breathiness-preservation-benchmark/

### 3. Prosody Transfer Benchmark
Objective benchmark measuring whether voice-cloning TTS preserves speaker-level F0/prosody from reference audio.
Finding: Chatterbox best F0 contour match (DTW 11.27), XTTS-v2 best breathy-modal contrast retention (0.79). Kokoro baseline zero.
→ _3-prosody-transfer-benchmark/


### 4. Identity Drift Benchmark
Measures whether TTS voice characteristics (F0, breathiness via Praat CPP, spectral features) drift over 5-minute synthesized monologues compared to a short reference clip.
Finding: all models drift measurably (+0.07–0.08 increase), Chatterbox stays closest to reference (0.491) but drifts fastest. Breathiness (CPP) is the least stable channel. Acoustic drift below perceptual threshold at 5 minutes (single-listener check).
→ _4-identity-drift/


### 5. Punctuation Sensitivity Benchmark
Measures whether TTS models respect punctuation: periods, questions, exclamations, commas, dashes, ellipses, quotation marks, capitalization.
Finding: XTTS has perfect pause hierarchy but zero question F0 rise. Chatterbox has best F0-based differentiation but weak pause timing. No model does both well. F5-TTS added as new model.
→ _5-punctuation-sensitivity/

### Related
- CodeSOTA TTS ELO: https://codesota.com/text-to-speech/elo
