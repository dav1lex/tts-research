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

## Related
- CodeSOTA TTS ELO: https://codesota.com/text-to-speech/elo
