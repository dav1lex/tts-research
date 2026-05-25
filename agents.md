# agents.md — TTS Research Benchmarks

## Models & Venvs

`venvs/` = central Python environments. Benchmarks use `../venvs/NAME/bin/python`.

| Venv | Python | Models | pip install |
|------|--------|--------|-------------|
| `venvs/py312` | 3.12.3 | Chatterbox, Kokoro + all analysis tools | `chatterbox-tts==0.1.7 kokoro==0.9.4 torch librosa pandas scipy matplotlib seaborn praat-parselmouth soundfile fastdtw scikit-learn` |
| `venvs/py310` | 3.10.20 | XTTS-v2 (Coqui TTS) | `TTS==0.22.0 torch numpy scipy soundfile` |
| `venvs/f5tts` | 3.12.3 | F5-TTS (flow matching) | `f5-tts torch soundfile` |

Model checkpoints auto-download to `~/.cache/huggingface/` on first use. No weights in repo.

## How to use a venv

```bash
source venvs/py312/bin/activate    # Chatterbox, Kokoro, or any analysis
source venvs/py310/bin/activate    # XTTS-v2
source venvs/f5tts/bin/activate    # F5-TTS
```

Scripts reference venv by path:
```python
#!../venvs/py312/bin/python
from chatterbox.tts import ChatterboxTTS
```

## Adding a new model

```bash
python3.XX -m venv venvs/MODELNAME
source venvs/MODELNAME/bin/activate
pip install PACKAGE torch torchaudio soundfile  # + any model-specific deps
# Verify:
python -c "from PACKAGE import MODELCLASS; print('OK')"
```

Add entry to this file.

## Benchmarks

| # | Name | Models Tested | Key Metric |
|---|------|--------------|------------|
| 1 | Emotional Channel Analysis | Chatterbox | Emotion vector routing |
| 2 | Breathiness Preservation | Chatterbox, XTTS-v2, Kokoro | CPP/HNR retention |
| 3 | Prosody Transfer | Chatterbox, XTTS-v2, Kokoro | F0 DTW, breathy-modal contrast |
| 4 | Identity Drift | Chatterbox, XTTS-v2, Kokoro | IQR-scaled Euclidean drift over 5min |
| 5 | Punctuation Sensitivity | TBD | Pause duration, F0 slope per punctuation |

## Reference Audio

All voice-cloning benchmarks use VCTK `p229_002.wav` (female, Southern England, 3.88s, f0=192.7Hz).

## To-Do

- [ ] Add CosyVoice 3 venv (needs `git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git` — slow submodules, timed out)
- [ ] Add Fish Speech S2, Spark-TTS venvs
