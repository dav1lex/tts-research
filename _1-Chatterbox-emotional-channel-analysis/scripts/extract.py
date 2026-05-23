import librosa
import numpy as np
import pandas as pd
from pathlib import Path

def extract_features(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=None)

    # --- PITCH ---
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([0.0])
    f0_mean = float(np.nanmean(f0_voiced))
    f0_std = float(np.nanstd(f0_voiced))
    f0_range = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced))
    voiced_ratio = float(voiced_flag.sum() / len(voiced_flag))

    # --- TEMPO ---
    duration = librosa.get_duration(y=y, sr=sr)
    words_per_sec = 0.0
    syllables_per_sec = 0.0

    # --- ENERGY ---
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # --- PAUSES ---
    intervals = librosa.effects.split(y, top_db=30)
    speech_samples = sum(end - start for start, end in intervals)
    total_samples = len(y)
    pause_total = float((total_samples - speech_samples) / sr)
    pause_count = float(max(0, len(intervals) - 1))
    if len(intervals) > 1:
        gaps = [intervals[i+1][0] - intervals[i][1] for i in range(len(intervals)-1)]
        pause_max = float(max(gaps) / sr)
    else:
        pause_max = 0.0

    # --- TIMBRE ---
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))
    mfcc_std = float(np.std(mfcc))

    # --- VOICE QUALITY ---
    if len(f0_voiced) > 1:
        f0_diffs = np.abs(np.diff(f0_voiced))
        jitter_proxy = float(np.mean(f0_diffs) / (f0_mean + 1e-8))
    else:
        jitter_proxy = 0.0
    rms_diffs = np.abs(np.diff(rms))
    shimmer_proxy = float(np.mean(rms_diffs) / (rms_mean + 1e-8))

    return {
        "f0_mean": f0_mean, "f0_std": f0_std,
        "f0_range": f0_range, "voiced_ratio": voiced_ratio,
        "duration": duration,
        "words_per_sec": words_per_sec, "syllables_per_sec": syllables_per_sec,
        "rms_mean": rms_mean, "rms_std": rms_std,
        "pause_count": pause_count, "pause_total": pause_total, "pause_max": pause_max,
        "spectral_centroid": spectral_centroid,
        "mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std,
        "jitter_proxy": jitter_proxy, "shimmer_proxy": shimmer_proxy
    }

# RUN EXTRACTION
failed = []
rows = []
for wav_path in sorted(Path("audio").rglob("*.wav")):
    try:
        emotion = wav_path.parent.name
        sent_id = wav_path.stem
        feats = extract_features(str(wav_path))
        rows.append({"emotion": emotion, "sent_id": sent_id, **feats})
        print(f"Extracted: {wav_path}")
    except Exception as e:
        failed.append(str(wav_path))
        print(f"FAILED: {wav_path} — {e}")

df = pd.DataFrame(rows)
Path("features").mkdir(exist_ok=True)
df.to_csv("features/raw_features.csv", index=False)
print(f"\nExtracted {len(df)} rows → features/raw_features.csv")

# COMPUTE NEUTRAL-RELATIVE Z-SCORE
feature_cols = [c for c in df.columns if c not in ["emotion", "sent_id"]]
neutral = df[df["emotion"] == "neutral"].set_index("sent_id")[feature_cols]

# Use global std as denominator floor to avoid division-by-near-zero
# when neutral baseline is ~0 (e.g., Pauses). This matches robust z-score practice.
global_std = df[feature_cols].std()
# Exclude constant columns (std=0) from normalization — they carry no signal
variable_cols = [c for c in feature_cols if global_std[c] > 0]
floor = global_std[variable_cols] * 0.1  # floor at 10% of global std

normalized_rows = []
for _, row in df.iterrows():
    neutral_row = neutral.loc[row["sent_id"]]
    z_row = {}
    for col in feature_cols:
        if col not in variable_cols:
            z_row[col] = 0.0  # constant columns → z=0
        else:
            d = abs(neutral_row[col])
            if d < floor[col]:
                d = floor[col]
            z_row[col] = (row[col] - neutral_row[col]) / d
    normalized_rows.append({"emotion": row["emotion"],
                             "sent_id": row["sent_id"], **z_row})

df_norm = pd.DataFrame(normalized_rows)
df_norm.to_csv("features/normalized_features.csv", index=False)
print("Normalization complete → features/normalized_features.csv")

# LOG FAILURES
if failed:
    with open("results/failed_extractions.txt", "w") as f:
        for path in failed:
            f.write(path + "\n")
    print(f"\n{len(failed)} files failed — logged to results/failed_extractions.txt")
