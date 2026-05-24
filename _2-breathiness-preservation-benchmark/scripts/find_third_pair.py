#!/usr/bin/env python3
"""Expand to ALL female VCTK speakers. Find best 3rd breathy candidate (lowest CPP + HNR)."""

from pathlib import Path
import json, math, numpy as np
import parselmouth, librosa, soundfile as sf
from parselmouth.praat import call

VCTK_DIR = Path("/home/davilex/Downloads/VCTK-Corpus-0.92/wav48_silence_trimmed")
SENTENCES = [f"{i:03d}" for i in range(1, 11)]
SR = 16000; F0_MIN = 60; F0_MAX = 400; FL = 0.04; HL = 0.01; MIN_VR = 0.15

def load_mono(p):
    a, sr = sf.read(p)
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != SR: a = librosa.resample(a.astype(float), orig_sr=sr, target_sr=SR)
    return a.astype(float), SR

def extract_one(path):
    a, sr = load_mono(path)
    fs, hs = int(round(FL*sr)), int(round(HL*sr))
    frames = librosa.util.frame(a, frame_length=fs, hop_length=hs).T
    rms = np.sqrt(np.mean(frames*frames, axis=1))
    ethresh = max(np.percentile(rms, 20)*0.5, np.max(rms)*0.01, 1e-8)
    f0, vf, vp = librosa.pyin(a, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=fs, hop_length=hs, center=False)
    fc = min(len(rms), len(vf), len(vp))
    mask = (rms[:fc] >= ethresh) & vf[:fc].astype(bool) & (vp[:fc] >= 0.50)
    vr = float(np.mean(mask))
    if vr < MIN_VR: raise ValueError(f"low voiced={vr:.3f}")
    ivs = []; s = None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        elif not v and s is not None: ivs.append((s*HL, i*HL+FL)); s = None
    if s is not None: ivs.append((s*HL, len(mask)*HL+FL))
    sound = parselmouth.Sound(a, sampling_frequency=sr)

    cpp_v = []
    for s, e in ivs:
        if e - s < 0.10: continue
        p = sound.extract_part(from_time=s, to_time=min(e, sound.xmax), preserve_times=False)
        try:
            pc = call(p, "To PowerCepstrogram", max(F0_MIN,75.0), HL, 5000, 50)
            c = call(pc, "Get CPPS", True, 0.02, 0.0005, max(F0_MIN,75.0), 330, 0.05, "Parabolic", 0.001, 0.0, "Exponential decay", "Robust")
            if math.isfinite(c): cpp_v.append(float(c))
        except: pass
    if not cpp_v: raise ValueError("CPPS failed")

    hnr_v = []
    for s, e in ivs:
        if e - s < 0.10: continue
        p = sound.extract_part(from_time=s, to_time=min(e, sound.xmax), preserve_times=False)
        try:
            h = p.to_harmonicity_cc(time_step=HL, minimum_pitch=max(F0_MIN,75.0))
            hnr = call(h, "Get mean", 0, 0)
            if math.isfinite(hnr): hnr_v.append(float(hnr))
        except: pass
    if not hnr_v: raise ValueError("HNR failed")

    return {"cpp": round(float(np.mean(cpp_v)), 4), "hnr": round(float(np.mean(hnr_v)), 4), "voiced": round(vr, 3)}

# Get all female speakers from VCTK speaker-info.txt
info_path = Path("/home/davilex/Downloads/VCTK-Corpus-0.92/speaker-info.txt")
female_speakers = []
with info_path.open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3: continue
        spk_id = parts[0]
        if not spk_id.startswith("p"): continue
        gender = parts[2]
        if gender == "F":
            female_speakers.append(spk_id)

print(f"Found {len(female_speakers)} female speakers in VCTK")

# Extract best sentence for each female speaker
results = []
for spk in female_speakers:
    best = None
    for s in SENTENCES:
        p = VCTK_DIR / spk / f"{spk}_{s}_mic1.flac"
        if not p.exists(): continue
        try:
            f = extract_one(p)
            if best is None or f["voiced"] > best["voiced"]:
                best = {**f, "sentence": s}
        except: pass
    if best:
        results.append({"speaker": spk, **best})

print(f"Extracted {len(results)} female speakers with voiced > 0.15")

# Find best breathy candidate: lowest CPP with HNR also low (both below median)
cpps = [r["cpp"] for r in results]
hnrs = [r["hnr"] for r in results]
median_cpp = float(np.median(cpps))
median_hnr = float(np.median(hnrs))

# Rank by CPP (lower = breathier)
breathy_sorted = sorted(results, key=lambda x: x["cpp"])
print(f"\nMedian CPP: {median_cpp:.2f}, Median HNR: {median_hnr:.2f}")

# Find best breathy: lowest CPP where HNR is also below median
breathy_best = None
for r in breathy_sorted:
    if r["hnr"] < median_hnr:
        breathy_best = r
        break

print(f"\n=== BEST BREATHY CANDIDATE ===")
print(f"  {breathy_best['speaker']}: s{breathy_best['sentence']} CPP={breathy_best['cpp']:.2f} HNR={breathy_best['hnr']:.2f}")

# Find best modal: highest CPP where HNR is also above median
modal_sorted = sorted(results, key=lambda x: x["cpp"], reverse=True)
modal_best = None
for r in modal_sorted:
    if r["hnr"] > median_hnr:
        modal_best = r
        break

print(f"\n=== BEST MODAL CANDIDATE ===")
print(f"  {modal_best['speaker']}: s{modal_best['sentence']} CPP={modal_best['cpp']:.2f} HNR={modal_best['hnr']:.2f}")

# Check if same sentence
same_sent = breathy_best["sentence"] == modal_best["sentence"]
print(f" Same sentence: {'✓' if same_sent else '✗'}")

# Print top 10 breathy and top 10 modal for context
print(f"\n=== TOP 10 BREATHY (lowest CPP, HNR must also be low) ===")
count = 0
for r in breathy_sorted:
    if r["hnr"] < median_hnr:
        print(f"  {r['speaker']}: s{r['sentence']} CPP={r['cpp']:.2f} HNR={r['hnr']:.2f}")
        count += 1
        if count >= 10: break

print(f"\n=== TOP 10 MODAL (highest CPP, HNR must also be high) ===")
count = 0
for r in modal_sorted:
    if r["hnr"] > median_hnr:
        print(f"  {r['speaker']}: s{r['sentence']} CPP={r['cpp']:.2f} HNR={r['hnr']:.2f}")
        count += 1
        if count >= 10: break

# Output the pair info
pair = {
    "breathy": {"speaker": breathy_best["speaker"], "sentence": breathy_best["sentence"], "cpp": breathy_best["cpp"], "hnr": breathy_best["hnr"]},
    "modal": {"speaker": modal_best["speaker"], "sentence": modal_best["sentence"], "cpp": modal_best["cpp"], "hnr": modal_best["hnr"]},
    "same_sentence": same_sent
}

out = Path("/home/davilex/tts-research/_2-breathiness-preservation-benchmark/selected_pair3.json")
with out.open("w") as f:
    json.dump(pair, f, indent=2)
print(f"\nWrote pair to {out}")
