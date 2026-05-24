#!/usr/bin/env python3
"""Rank Southern England VCTK speakers on s001-010, pick 3 clean pairs with CPP+HNR both correct."""

from pathlib import Path
import csv, math, numpy as np
import parselmouth, librosa, soundfile as sf
from parselmouth.praat import call
from itertools import combinations

VCTK_DIR = Path("/home/davilex/Downloads/VCTK-Corpus-0.92/wav48_silence_trimmed")
SPEAKERS = ["p225","p228","p229","p231","p232","p240","p257","p258","p268"]
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

    # CPP
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

    # HNR
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

    return {
        "cpp": round(float(np.mean(cpp_v)), 4),
        "hnr": round(float(np.mean(hnr_v)), 4),
        "voiced": round(vr, 3)
    }

# Extract best sentence per speaker
results = []
for spk in SPEAKERS:
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
        print(f"OK {spk}: s{best['sentence']} CPP={best['cpp']:.2f} HNR={best['hnr']:.2f} voiced={best['voiced']:.2%}")

# All valid breathy-modal pairs where both CPP+HNR direction correct AND same sentence
print("\n=== All valid pairs (same sentence, CPP+HNR direction correct) ===")
valid_pairs = []
for b, m in combinations(results, 2):
    b_spk, m_spk = b["speaker"], m["speaker"]
    # Determine which is breathy candidate (lower CPP)
    if b["cpp"] < m["cpp"]:
        breathy, modal = b, m
    else:
        breathy, modal = m, b
    # Check HNR direction too
    if breathy["hnr"] < modal["hnr"] and breathy["cpp"] < modal["cpp"]:
        spread = (modal["cpp"] - breathy["cpp"]) + (modal["hnr"] - breathy["hnr"])
        valid_pairs.append({
            "breathy": breathy["speaker"], "b_sent": breathy["sentence"],
            "modal": modal["speaker"], "m_sent": modal["sentence"],
            "b_cpp": breathy["cpp"], "m_cpp": modal["cpp"],
            "b_hnr": breathy["hnr"], "m_hnr": modal["hnr"],
            "cpp_spread": round(modal["cpp"] - breathy["cpp"], 2),
            "hnr_spread": round(modal["hnr"] - breathy["hnr"], 2),
            "total_spread": round(spread, 2),
            "same_sentence": breathy["sentence"] == modal["sentence"]
        })

valid_pairs.sort(key=lambda x: x["total_spread"], reverse=True)
for p in valid_pairs:
    same = "✓" if p["same_sentence"] else "✗"
    print(f"  {p['breathy']}(s{p['b_sent']}) vs {p['modal']}(s{p['m_sent']}) "
          f"CPP:{p['b_cpp']:.2f}→{p['m_cpp']:.2f}(Δ{p['cpp_spread']:.2f}) "
          f"HNR:{p['b_hnr']:.2f}→{p['m_hnr']:.2f}(Δ{p['hnr_spread']:.2f}) "
          f"same_sent={same}")

# Pick top 3, preferring same-sentence pairs
same_sent = [p for p in valid_pairs if p["same_sentence"]]
diff_sent = [p for p in valid_pairs if not p["same_sentence"]]

selected = same_sent[:3]
# If not enough same-sentence pairs, fill with diff-sentence
if len(selected) < 3:
    selected += diff_sent[:3 - len(selected)]

selected = selected[:3]

print(f"\n=== SELECTED PAIRS (top 3) ===")
for i, p in enumerate(selected):
    print(f"  Pair {i+1}: {p['breathy']}(s{p['b_sent']}) → {p['modal']}(s{p['m_sent']})")
    print(f"    CPP: {p['b_cpp']:.2f} vs {p['m_cpp']:.2f} (Δ{p['cpp_spread']:.2f}, direction={'✓' if p['b_cpp'] < p['m_cpp'] else '✗'})")
    print(f"    HNR: {p['b_hnr']:.2f} vs {p['m_hnr']:.2f} (Δ{p['hnr_spread']:.2f}, direction={'✓' if p['b_hnr'] < p['m_hnr'] else '✗'})")

# Write selected pairs for metadata generation
out = Path("/home/davilex/tts-research/_2-breathiness-preservation-benchmark/selected_pairs.json")
import json
with out.open("w") as f:
    json.dump(selected, f, indent=2)
print(f"\nWrote selected pairs to {out}")
