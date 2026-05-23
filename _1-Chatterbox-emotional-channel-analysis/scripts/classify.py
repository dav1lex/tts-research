import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

df_norm = pd.read_csv("features/normalized_features.csv")

CHANNELS = {
    "Pitch":         ["f0_mean", "f0_std", "f0_range", "voiced_ratio"],
    "Tempo":         ["duration"],
    "Energy":        ["rms_mean", "rms_std"],
    "Pauses":        ["pause_count", "pause_total", "pause_max"],
    "Timbre":        ["spectral_centroid", "mfcc_mean", "mfcc_std"],
    "Voice quality": ["jitter_proxy", "shimmer_proxy"]
}

# === STEP 4: CHANNEL HEATMAP ===
print("=== CHANNEL HEATMAP ===")
df_emotion = df_norm[df_norm["emotion"] != "neutral"]
emotions = ["happy", "sad", "angry", "fearful", "surprised", "low_energy"]
heatmap_data = {}

for channel, cols in CHANNELS.items():
    row = {}
    for emotion in emotions:
        sub = df_emotion[df_emotion["emotion"] == emotion][cols]
        row[emotion] = float(sub.abs().mean().mean())
    heatmap_data[channel] = row

heatmap_df = pd.DataFrame(heatmap_data).T
heatmap_df = heatmap_df.reindex(sorted(heatmap_df.index,
    key=lambda x: heatmap_df.loc[x].mean(), reverse=True))

plt.figure(figsize=(10, 5))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Chatterbox: Which Channel Moves Most?\n(mean absolute neutral-relative z)")
plt.tight_layout()
plt.savefig("results/heatmap_channel_by_emotion.png", dpi=150)
print("Saved: results/heatmap_channel_by_emotion.png")

heatmap_df.to_csv("results/heatmap_values.csv")
print("Saved: results/heatmap_values.csv")

# === STEP 5: ABLATION CLASSIFIER ===
print("\n=== ABLATION CLASSIFIER ===")
logo = LeaveOneGroupOut()
groups = df_norm["sent_id"].values
y = df_norm["emotion"].values
results = {}

for channel, cols in CHANNELS.items():
    X = df_norm[cols].values
    preds = []
    truths = []
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        preds.extend(clf.predict(X[test_idx]))
        truths.extend(y[test_idx])
    acc = accuracy_score(truths, preds)
    results[channel] = round(acc, 3)
    print(f"{channel}: {acc:.3f}")

results_df = pd.DataFrame(
    sorted(results.items(), key=lambda x: x[1], reverse=True),
    columns=["Channel", "Accuracy"]
)
results_df.to_csv("results/ablation_accuracy.csv", index=False)
print("\n" + results_df.to_string(index=False))

plt.figure(figsize=(8, 4))
plt.barh(results_df["Channel"], results_df["Accuracy"], color="steelblue")
plt.axvline(x=0.143, color="red", linestyle="--", label="Random baseline (0.143)")
plt.xlabel("Leave-one-sentence-out accuracy")
plt.title("Chatterbox: Channel Ablation Classifier")
plt.legend()
plt.tight_layout()
plt.savefig("results/ablation_bar.png", dpi=150)
print("Saved: results/ablation_bar.png")
