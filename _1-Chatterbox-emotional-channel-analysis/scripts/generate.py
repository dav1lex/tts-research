import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

SENTENCES = {
    "s1": "The system failed to respond within the expected time frame.",
    "s2": "She walked into the room and nobody noticed.",
    "s3": "The package was delivered three weeks after the expected date.",
    "s4": "He opened the letter and read it twice before putting it down.",
    "s5": "The meeting has been rescheduled for the third time this month."
}

EMOTION_MAP = {
    "neutral": 0.3,
    "happy": 0.7,
    "sad": 0.4,
    "angry": 0.9,
    "fearful": 0.8,
    "surprised": 0.85,
    "low_energy": 0.1
}

SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
model = ChatterboxTTS.from_pretrained(device=device)

total = len(EMOTION_MAP) * len(SENTENCES)
count = 0

for emotion, exaggeration in EMOTION_MAP.items():
    for sent_id, text in SENTENCES.items():
        torch.manual_seed(SEED)
        wav = model.generate(text, exaggeration=exaggeration)
        out_dir = Path(f"audio/{emotion}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sent_id}.wav"
        torchaudio.save(str(out_path), wav.cpu(), model.sr)
        count += 1
        print(f"[{count}/{total}] Generated: {out_path}")
        torch.cuda.empty_cache()

print(f"\nDone. {count} files generated.")
