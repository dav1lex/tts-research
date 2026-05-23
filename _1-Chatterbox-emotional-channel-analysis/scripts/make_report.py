"""Generate a clean PDF report from the Chatterbox emotion analysis results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
ablation = pd.read_csv("results/ablation_accuracy.csv")
heatmap = pd.read_csv("results/heatmap_values.csv", index_col=0)
comparison = pd.read_csv("results/comparison_table.csv")
raw = pd.read_csv("features/raw_features.csv")
norm = pd.read_csv("features/normalized_features.csv")

pdf = PdfPages("results/report.pdf")

# ============================================================
# PAGE 1: TITLE
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.72, "Chatterbox Emotion Channel Analysis",
        ha="center", va="center", fontsize=26, fontweight="bold")
ax.text(0.5, 0.65, "Replication of Kacper Wikiel's Methodology",
        ha="center", va="center", fontsize=16, color="#555555")

ax.axhline(0.58, 0.2, 0.8, color="#333333", linewidth=1)

details = [
    ("Date", datetime.now().strftime("%Y-%m-%d")),
    ("Model", "Chatterbox (Resemble AI)"),
    ("Sentences", "5 (neutral-valence)"),
    ("Conditions", "7 (neutral, happy, sad, angry, fearful, surprised, low_energy)"),
    ("Total audio files", "35 WAV files"),
    ("Classifier", "RandomForest (n=100), LeaveOneGroupOut by sentence"),
    ("Random baseline", "0.143 (1/7 classes)"),
]

y = 0.52
for label, value in details:
    ax.text(0.15, y, label, fontsize=12, fontweight="bold", va="top")
    ax.text(0.35, y, value, fontsize=12, va="top")
    y -= 0.045

ax.text(0.5, 0.12,
        "This report reproduces Kacper Wikiel's emotion channel analysis\n"
        "on the Chatterbox TTS model and compares results to his Zonos findings.",
        ha="center", va="center", fontsize=11, color="#666666", style="italic")

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 2: METHODOLOGY
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "Methodology", ha="center", va="top", fontsize=20, fontweight="bold")
ax.axhline(0.92, 0.1, 0.9, color="#333333", linewidth=1)

sections = [
    ("1. Audio Generation",
     "5 neutral sentences were spoken by Chatterbox in 7 emotion conditions.\n"
     "Chatterbox has one control parameter: exaggeration (0.0 to 1.0).\n"
     "Each condition was mapped to an exaggeration value:\n"
     "  neutral=0.3, happy=0.7, sad=0.4, angry=0.9, fearful=0.8, surprised=0.85, low_energy=0.1\n"
     "Seed was fixed at 42 for all generations."),

    ("2. Feature Extraction",
     "17 acoustic features were extracted per WAV file using librosa:\n"
     "  Pitch: f0_mean, f0_std, f0_range, voiced_ratio\n"
     "  Tempo: duration\n"
     "  Energy: rms_mean, rms_std\n"
     "  Pauses: pause_count, pause_total, pause_max\n"
     "  Timbre: spectral_centroid, mfcc_mean, mfcc_std\n"
     "  Voice quality: jitter_proxy, shimmer_proxy"),

    ("3. Normalization",
     "Neutral-relative z-scores were computed per sentence:\n"
     "  z = (value - neutral) / max(|neutral|, 0.1 * global_std)\n"
     "A denominator floor prevents division-by-zero when neutral baseline is near zero.\n"
     "Constant features (words_per_sec, syllables_per_sec) were set to z=0."),

    ("4. Channel Heatmap",
     "Mean absolute z-score was computed per channel per emotion.\n"
     "Neutral condition was excluded (z=0 by construction)."),

    ("5. Ablation Classifier",
     "RandomForest (100 trees) was trained on each channel group separately.\n"
     "Validation: LeaveOneGroupOut, where groups = sentence ID.\n"
     "This tests whether each channel alone can predict the emotion condition."),

    ("6. Key Difference from Zonos",
     "Zonos uses an 8D emotion vector (independent weight per emotion).\n"
     "Chatterbox uses a scalar exaggeration value (single intensity control).\n"
     "This means Chatterbox cannot isolate individual emotions. It only controls\n"
     "overall expressiveness intensity. This is an architectural difference."),
]

y = 0.88
for title, body in sections:
    ax.text(0.1, y, title, fontsize=13, fontweight="bold", va="top")
    y -= 0.035
    ax.text(0.1, y, body, fontsize=10, va="top", linespacing=1.6)
    y -= 0.035 * (body.count("\n") + 1)
    y -= 0.015

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 3: ABLATION RESULTS
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "Ablation Classifier Results", ha="center", va="top", fontsize=20, fontweight="bold")
ax.axhline(0.92, 0.1, 0.9, color="#333333", linewidth=1)

ax.text(0.1, 0.87,
        "Each channel group was tested independently. Higher accuracy means that channel\n"
        "carries more emotion-distinguishable information. Random baseline is 0.143.",
        fontsize=11, va="top")

# Table
table_data = []
for _, row in ablation.iterrows():
    ch = row["Channel"]
    acc = row["Accuracy"]
    bar = acc / 0.457  # normalize to best
    table_data.append((ch, acc, bar))

y = 0.78
ax.text(0.15, y, "Channel", fontsize=12, fontweight="bold")
ax.text(0.55, y, "Accuracy", fontsize=12, fontweight="bold")
ax.text(0.70, y, "Relative", fontsize=12, fontweight="bold")
y -= 0.04

for ch, acc, bar in table_data:
    ax.text(0.15, y, ch, fontsize=11, va="center")
    ax.text(0.55, y, f"{acc:.3f}", fontsize=11, va="center", fontweight="bold")
    # mini bar
    rect = mpatches.Rectangle((0.70, y - 0.012), bar * 0.2, 0.024,
                               facecolor="#4472C4", alpha=0.8)
    ax.add_patch(rect)
    ax.text(0.70 + bar * 0.2 + 0.01, y, f"{bar:.0%}", fontsize=10, va="center")
    y -= 0.045

# Baseline line
ax.axhline(y + 0.01, 0.68, 0.92, color="red", linestyle="--", linewidth=0.8)
ax.text(0.70, y + 0.015, "Random baseline: 0.143", fontsize=9, color="red")

# Bar chart
ax.text(0.5, 0.35, "Visual", fontsize=14, fontweight="bold", ha="center")
ax2 = fig.add_axes([0.15, 0.08, 0.7, 0.22])
colors = ["#4472C4" if a > 0.143 else "#CCCCCC" for a in ablation["Accuracy"]]
bars = ax2.barh(ablation["Channel"][::-1], ablation["Accuracy"][::-1], color=colors[::-1], edgecolor="white")
ax2.axvline(x=0.143, color="red", linestyle="--", linewidth=1, label="Random (0.143)")
ax2.set_xlabel("Leave-one-sentence-out accuracy", fontsize=10)
ax2.set_xlim(0, 0.55)
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
for bar, val in zip(bars, ablation["Accuracy"][::-1]):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
             va="center", fontsize=9, fontweight="bold")

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 4: HEATMAP
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "Channel Heatmap", ha="center", va="top", fontsize=20, fontweight="bold")
ax.axhline(0.92, 0.1, 0.9, color="#333333", linewidth=1)

ax.text(0.1, 0.87,
        "Mean absolute neutral-relative z-score per channel per emotion.\n"
        "Higher values mean the channel deviates more from neutral speech.",
        fontsize=11, va="top")

# Heatmap image
ax_heatmap = fig.add_axes([0.1, 0.35, 0.8, 0.45])
emotions = ["happy", "sad", "angry", "fearful", "surprised", "low_energy"]
hm_data = heatmap[emotions].values
im = ax_heatmap.imshow(hm_data, cmap="YlOrRd", aspect="auto")
ax_heatmap.set_xticks(range(len(emotions)))
ax_heatmap.set_xticklabels(emotions, fontsize=10)
ax_heatmap.set_yticks(range(len(heatmap.index)))
ax_heatmap.set_yticklabels(heatmap.index, fontsize=10)
for i in range(len(heatmap.index)):
    for j in range(len(emotions)):
        val = hm_data[i, j]
        color = "white" if val > hm_data.max() * 0.6 else "black"
        ax_heatmap.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
ax_heatmap.set_xlabel("Emotion condition", fontsize=11)
ax_heatmap.set_ylabel("Channel", fontsize=11)
fig.colorbar(im, ax=ax_heatmap, shrink=0.4, label="Mean |z-score|")

# Summary
ax.text(0.1, 0.28, "Largest physical mover by channel:", fontsize=12, fontweight="bold")
means = heatmap.mean(axis=1).sort_values(ascending=False)
y = 0.23
for ch, val in means.items():
    ax.text(0.15, y, f"  {ch}: {val:.2f}", fontsize=11, va="top")
    y -= 0.03

ax.text(0.1, 0.05,
        "Note: Pauses z-scores are high because neutral baseline pause features are near zero.\n"
        "A denominator floor (10% of global std) was applied to prevent division-by-zero.\n"
        "The classifier result is unaffected by normalization scale.",
        fontsize=9, color="#666666", va="top")

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 5: COMPARISON TO ZONOS
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "Comparison to Zonos (Kacper Wikiel)", ha="center", va="top", fontsize=20, fontweight="bold")
ax.axhline(0.92, 0.1, 0.9, color="#333333", linewidth=1)

ax.text(0.1, 0.87,
        "Side-by-side comparison of ablation classifier accuracy.",
        fontsize=11, va="top")

# Comparison table
y = 0.80
ax.text(0.10, y, "Channel", fontsize=12, fontweight="bold")
ax.text(0.40, y, "Zonos", fontsize=12, fontweight="bold")
ax.text(0.65, y, "Chatterbox", fontsize=12, fontweight="bold")
ax.text(0.85, y, "Rank", fontsize=12, fontweight="bold")
y -= 0.04

for _, row in comparison.iterrows():
    ax.text(0.10, y, row["Channel"], fontsize=11, va="center")
    ax.text(0.40, y, f"{row['Zonos_accuracy']:.2f}", fontsize=11, va="center")
    ax.text(0.65, y, f"{row['Chatterbox_accuracy']:.3f}", fontsize=11, va="center")
    ax.text(0.85, y, row["Rank_change"], fontsize=10, va="center")
    y -= 0.04

# Side by side bar chart
ax.text(0.5, 0.52, "Classifier Accuracy by Channel", fontsize=14, fontweight="bold", ha="center")
ax_bar = fig.add_axes([0.1, 0.15, 0.8, 0.32])

channels = comparison["Channel"].tolist()
x = np.arange(len(channels))
width = 0.35
bars1 = ax_bar.bar(x - width/2, comparison["Zonos_accuracy"], width, label="Zonos", color="#E74C3C", alpha=0.8)
bars2 = ax_bar.bar(x + width/2, comparison["Chatterbox_accuracy"], width, label="Chatterbox", color="#3498DB", alpha=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(channels, fontsize=9)
ax_bar.axhline(y=0.143, color="gray", linestyle="--", linewidth=0.8, label="Random (0.143)")
ax_bar.set_ylabel("Accuracy", fontsize=10)
ax_bar.legend(fontsize=9)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

for bar in bars1:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 6: KEY FINDINGS
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "Key Findings", ha="center", va="top", fontsize=20, fontweight="bold")
ax.axhline(0.92, 0.1, 0.9, color="#333333", linewidth=1)

findings = [
    ("Finding 1: The mismatch does not replicate",
     "Kacper found that in Zonos, timbre moves most physically but pitch classifies best.\n"
     "In Chatterbox, pauses both move most AND classify best (0.457).\n"
     "The mismatch between physical movement and classification does not appear here."),

    ("Finding 2: Scalar control collapses onto timing",
     "Chatterbox has one control knob (exaggeration). It cannot route expressiveness\n"
     "to specific acoustic channels. The model defaults to timing and pauses as its\n"
     "primary expressive signal. Higher exaggeration creates more dramatic prosodic breaks.\n"
     "Lower exaggeration creates flatter, more monotone delivery."),

    ("Finding 3: All channels beat random baseline",
     "Every channel scored above 0.143 (random). This means Chatterbox's exaggeration\n"
     "parameter does affect all acoustic dimensions, just to different degrees.\n"
     "Pauses (0.457) and Tempo (0.400) are the strongest signals.\n"
     "Pitch (0.314) and Timbre (0.314) are moderate.\n"
     "Energy (0.286) is the weakest."),

    ("Finding 4: Architecture shapes the emotion-acoustic relationship",
     "Zonos (8D vector) routes emotion to timbre.\n"
     "Chatterbox (scalar) routes emotion to pauses.\n"
     "This is not a quality judgment. It shows that model design determines which\n"
     "acoustic channels carry emotion information."),
]

y = 0.87
for title, body in findings:
    ax.text(0.1, y, title, fontsize=13, fontweight="bold", va="top")
    y -= 0.035
    ax.text(0.1, y, body, fontsize=10.5, va="top", linespacing=1.7)
    y -= 0.035 * (body.count("\n") + 1)
    y -= 0.025

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 7: RAW DATA SUMMARY
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "Appendix: Raw Feature Summary", ha="center", va="top", fontsize=20, fontweight="bold")
ax.axhline(0.92, 0.1, 0.9, color="#333333", linewidth=1)

ax.text(0.1, 0.87,
        "Mean raw feature values per emotion condition (before normalization).",
        fontsize=11, va="top")

# Summary table
numeric_cols = raw.select_dtypes(include="number").columns
summary = raw.groupby("emotion")[numeric_cols].mean()
cols = ["f0_mean", "f0_std", "f0_range", "duration", "rms_mean",
        "pause_count", "pause_total", "pause_max", "spectral_centroid"]

y = 0.82
ax.text(0.05, y, "Feature", fontsize=10, fontweight="bold")
x_pos = 0.20
for emo in summary.index:
    ax.text(x_pos, y, emo, fontsize=9, fontweight="bold", ha="center")
    x_pos += 0.11

y -= 0.035
for col in cols:
    ax.text(0.05, y, col, fontsize=9, va="center")
    x_pos = 0.20
    for emo in summary.index:
        val = summary.loc[emo, col]
        ax.text(x_pos, y, f"{val:.2f}", fontsize=8, ha="center", va="center")
        x_pos += 0.11
    y -= 0.028

ax.text(0.1, y - 0.05,
        "Full data files:\n"
        "  features/raw_features.csv -- 35 rows, 19 columns\n"
        "  features/normalized_features.csv -- 35 rows, 19 columns\n"
        "  results/heatmap_values.csv -- 6 channels x 6 emotions\n"
        "  results/ablation_accuracy.csv -- 6 channels with accuracy scores\n"
        "  results/comparison_table.csv -- Zonos vs Chatterbox side by side",
        fontsize=10, va="top", linespacing=1.6)

pdf.savefig(fig)
plt.close()

pdf.close()
print("Report saved to results/report.pdf")
