#!/usr/bin/env python3
"""Extract CPP and HNR for ALL VCTK speakers on sentence 002 to find best breathy/modal pairs."""

from pathlib import Path
import csv
import numpy as np
import parselmouth
from parselmouth.praat import call
import math

VCTK_DIR = Path("/home/davilex/Downloads/VCTK-Corpus-0.92/wav48_silence_trimmed")
SENTENCE = "002"
TARGET_SR = 16000

F0_MIN = 60.0
F0_MAX = 400.0
FRAME_LENGTH = 0.04
HOP_LENGTH = 0.01
MIN_VOICED_RATIO = 0.10


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import librosa
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("empty audio")
    if sample_rate != TARGET_SR:
        audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=TARGET_SR)
        sample_rate = TARGET_SR
    return audio.astype(float), sample_rate


def voiced_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_threshold = max(float(np.percentile(rms, 20)) * 0.5, float(np.max(rms)) * 0.01, 1e-8)

    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio, fmin=F0_MIN, fmax=F0_MAX, sr=sample_rate,
        frame_length=frame_samples, hop_length=hop_samples, center=False,
    )
    frame_count = min(len(rms), len(voiced_flag), len(voiced_probability))
    mask = (
        (rms[:frame_count] >= energy_threshold)
        & voiced_flag[:frame_count].astype(bool)
        & (voiced_probability[:frame_count] >= 0.50)
    )
    return mask


def voiced_intervals(mask: np.ndarray) -> list[tuple[float, float]]:
    intervals = []
    start = None
    for index, is_voiced in enumerate(mask):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            intervals.append((start * HOP_LENGTH, index * HOP_LENGTH + FRAME_LENGTH))
            start = None
    if start is not None:
        intervals.append((start * HOP_LENGTH, len(mask) * HOP_LENGTH + FRAME_LENGTH))
    return intervals


def praat_cpp(sound: parselmouth.Sound, intervals: list[tuple[float, float]]) -> float:
    values = []
    for start, end in intervals:
        if end - start < 0.10:
            continue
        part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
        try:
            power_cepstrogram = call(part, "To PowerCepstrogram", max(F0_MIN, 75.0), HOP_LENGTH, 5000, 50)
            cpp = call(
                power_cepstrogram, "Get CPPS", True, 0.02, 0.0005,
                max(F0_MIN, 75.0), 330, 0.05, "Parabolic", 0.001, 0.0,
                "Exponential decay", "Robust",
            )
        except Exception:
            continue
        if math.isfinite(cpp):
            values.append(float(cpp))
    if not values:
        raise ValueError("Praat CPPS failed on voiced intervals")
    return float(np.mean(values))


def praat_hnr(sound: parselmouth.Sound, intervals: list[tuple[float, float]]) -> float:
    values = []
    for start, end in intervals:
        if end - start < 0.10:
            continue
        part = sound.extract_part(from_time=start, to_time=min(end, sound.xmax), preserve_times=False)
        try:
            harmonicity = part.to_harmonicity_cc(time_step=HOP_LENGTH, minimum_pitch=max(F0_MIN, 75.0))
            hnr = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            continue
        if math.isfinite(hnr):
            values.append(float(hnr))
    if not values:
        raise ValueError("Praat harmonicity failed on voiced intervals")
    return float(np.mean(values))


def spectral_tilt(audio: np.ndarray, sample_rate: int, mask: np.ndarray) -> float:
    import librosa
    import numpy as np
    frame_samples = int(round(FRAME_LENGTH * sample_rate))
    hop_samples = int(round(HOP_LENGTH * sample_rate))
    frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples).T
    usable_count = min(len(frames), len(mask))
    voiced_frames = frames[:usable_count][mask[:usable_count]]
    if len(voiced_frames) == 0:
        raise ValueError("no voiced frames for spectral tilt")

    slopes = []
    frequencies = np.fft.rfftfreq(frame_samples, 1 / sample_rate)
    band = (frequencies >= 100) & (frequencies <= 5000)
    log_frequency = np.log2(frequencies[band])
    design = np.vstack([log_frequency, np.ones(len(log_frequency))]).T
    window = np.hanning(frame_samples)

    for frame in voiced_frames:
        spectrum = np.abs(np.fft.rfft(frame * window))
        db = 20 * np.log10(np.maximum(spectrum[band], 1e-10))
        slope, _ = np.linalg.lstsq(design, db, rcond=None)[0]
        slopes.append(float(slope))
    return float(np.mean(slopes))


def extract_one(path: Path) -> dict:
    audio, sample_rate = load_mono(path)
    mask = voiced_mask(audio, sample_rate)
    intervals = voiced_intervals(mask)
    
    voiced_ratio = float(np.mean(mask))
    if voiced_ratio < MIN_VOICED_RATIO:
        raise ValueError(f"too little voiced material: ratio={voiced_ratio:.4f}")
    
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    
    return {
        "cpp_mean": round(praat_cpp(sound, intervals), 6),
        "hnr_mean": round(praat_hnr(sound, intervals), 6),
        "spectral_tilt_mean": round(spectral_tilt(audio, sample_rate, mask), 6),
        "voiced_ratio": round(voiced_ratio, 4),
    }


def main():
    # Get all speaker directories
    all_speakers = sorted([d.name for d in VCTK_DIR.iterdir() if d.is_dir() and d.name.startswith('p')])
    
    results = []
    
    for speaker in all_speakers:
        flac_path = VCTK_DIR / speaker / f"{speaker}_{SENTENCE}_mic1.flac"
        
        if not flac_path.exists():
            continue
        
        try:
            features = extract_one(flac_path)
            # Get region info
            results.append({
                "speaker": speaker,
                "sentence": SENTENCE,
                **features,
            })
        except Exception as e:
            continue
    
    # Sort by CPP (lower = breathier)
    results.sort(key=lambda x: x["cpp_mean"])
    
    # Write ranked CSV
    output_path = Path("/home/davilex/tts-research/_2-breathiness-preservation-benchmark/vctk_s002_ranked.csv")
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "sentence", "cpp_mean", "hnr_mean", "spectral_tilt_mean", "voiced_ratio"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Wrote {len(results)} speakers to {output_path}")
    print(f"\n=== TOP 5 BREATHY (lowest CPP) ===")
    for r in results[:5]:
        print(f"{r['speaker']}: CPP={r['cpp_mean']:.2f}, HNR={r['hnr_mean']:.2f}, tilt={r['spectral_tilt_mean']:.2f}")
    
    print(f"\n=== TOP 5 MODAL (highest CPP) ===")
    for r in results[-5:]:
        print(f"{r['speaker']}: CPP={r['cpp_mean']:.2f}, HNR={r['hnr_mean']:.2f}, tilt={r['spectral_tilt_mean']:.2f}")


if __name__ == "__main__":
    main()
