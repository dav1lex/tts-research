#!/usr/bin/env python3
"""Generate figures for the semantic priming experiment report."""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FEATURES_CSV = PROJECT_ROOT / "results" / "features.csv"
STATS_SUMMARY = PROJECT_ROOT / "results" / "stats_summary.csv"
STATS_PAIRWISE = PROJECT_ROOT / "results" / "stats_pairwise.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["chatterbox", "xtts", "kokoro"]
MODEL_LABELS = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro",
}
MODEL_COLORS = {
    "chatterbox": "#2196F3",
    "xtts": "#FF9800",
    "kokoro": "#9CA3AF",
}
CONDITION_ORDER = ["cold", "primed_neutral", "primed_owl", "primed_death"]
CONDITION_LABELS = {
    "cold": "Cold",
    "primed_neutral": "Neutral\nPrime",
    "primed_owl": "Owl\nPrime",
    "primed_death": "Death\nPrime",
}
CONDITION_COLORS = {
    "cold": "#607D8B",
    "primed_neutral": "#78909C",
    "primed_owl": "#FF7043",
    "primed_death": "#7B1FA2",
}
CONDITION_HATCH = {
    "cold": "",
    "primed_neutral": "..",
    "primed_owl": "//",
    "primed_death": "\\\\",
}

LABEL_FEATURES = {
    "f0_mean": "F0 Mean (Hz)",
    "f0_std": "F0 Std (Hz)",
    "f0_range": "F0 Range (Hz)",
    "speech_rate": "Speech Rate (words/s)",
    "pause_count": "Pause Count",
    "pause_duration_mean": "Pause Duration (s)",
    "rms_energy": "RMS Energy",
    "spectral_centroid": "Spectral Centroid (Hz)",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "font.family": "sans-serif",
    }
)


def load_features_csv() -> dict[str, dict[str, list[dict]]]:
    """Return {model: {condition: [row, ...]}}."""
    with open(FEATURES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        data[row["model"]][row["condition"]].append(row)
    return data


def load_anova() -> list[dict]:
    """Return ANOVA rows from stats_summary.csv (condition=='ANOVA')."""
    with open(STATS_SUMMARY, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["condition"] == "ANOVA"]


def load_pairwise() -> list[dict]:
    with open(STATS_PAIRWISE, newline="") as f:
        return list(csv.DictReader(f))


# ── Figure 1: F0 Mean by Condition ──────────────────────────────────────────

def fig1_f0_mean(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        groups = []
        for cond in CONDITION_ORDER:
            rows = data[model].get(cond, [])
            vals = [float(r["f0_mean"]) for r in rows]
            groups.append(vals)

        positions = np.arange(len(CONDITION_ORDER))
        bp = ax.boxplot(
            groups,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="black", markersize=6),
            medianprops=dict(color="white", linewidth=1.5),
        )

        for patch, cond in zip(bp["boxes"], CONDITION_ORDER):
            patch.set_facecolor(CONDITION_COLORS[cond])
            patch.set_alpha(0.75)
            patch.set_hatch(CONDITION_HATCH[cond])

        # Overlay individual data points
        for i, (cond, vals) in enumerate(zip(CONDITION_ORDER, groups)):
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color="black", s=20, alpha=0.6, zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontweight="bold", color=MODEL_COLORS[model])
        ax.set_ylabel("F0 Mean (Hz)")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Target-Sentence F0 Mean by Prior Context", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "f0_mean.png", bbox_inches="tight")
    plt.close()
    print("  fig1: f0_mean.png")


# ── Figure 2: Speech Rate & Pause Count ─────────────────────────────────────

def fig2_speech_pause(data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey="row")

    for idx, model in enumerate(MODEL_ORDER):
        # Speech rate (top row)
        ax0 = axes[0][idx]
        for j, cond in enumerate(CONDITION_ORDER):
            rows = data[model].get(cond, [])
            vals = [float(r["speech_rate"]) for r in rows]
            mean = np.mean(vals) if vals else 0
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0
            ax0.bar(j, mean, 0.6, color=CONDITION_COLORS[cond], alpha=0.8,
                    yerr=sd, capsize=3, error_kw={"linewidth": 1.5})
        ax0.set_xticks(range(len(CONDITION_ORDER)))
        ax0.set_xticklabels([""] * len(CONDITION_ORDER))
        ax0.set_title(MODEL_LABELS[model], fontweight="bold", color=MODEL_COLORS[model])
        if idx == 0:
            ax0.set_ylabel("Speech Rate (words/s)")
        ax0.grid(axis="y", alpha=0.3)

        # Pause count (bottom row)
        ax1 = axes[1][idx]
        for j, cond in enumerate(CONDITION_ORDER):
            rows = data[model].get(cond, [])
            vals = [float(r["pause_count"]) for r in rows]
            mean = np.mean(vals) if vals else 0
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0
            ax1.bar(j, mean, 0.6, color=CONDITION_COLORS[cond], alpha=0.8,
                    yerr=sd, capsize=3, error_kw={"linewidth": 1.5})
        ax1.set_xticks(range(len(CONDITION_ORDER)))
        ax1.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8)
        if idx == 0:
            ax1.set_ylabel("Pause Count")
        ax1.grid(axis="y", alpha=0.3)

    plt.suptitle("Target-Sentence Speech Rate & Pause Count by Prior Context",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "speech_pause.png", bbox_inches="tight")
    plt.close()
    print("  fig2: speech_pause.png")


# ── Figure 3: Effect Size (η²) Summary ───────────────────────────────────────

def fig3_effects(anova_rows):
    features_plot = [
        "f0_mean", "f0_std", "f0_range",
        "speech_rate", "pause_count", "pause_duration_mean",
        "rms_energy", "spectral_centroid",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(features_plot))
    width = 0.25

    for j, model in enumerate(MODEL_ORDER):
        etas = []
        for feat in features_plot:
            match = [r for r in anova_rows if r["model"] == model and r["feature"] == feat]
            etas.append(float(match[0]["eta_squared"]) if match else 0)
        bars = ax.bar(x + j * width, etas, width,
                      label=MODEL_LABELS[model], color=MODEL_COLORS[model], alpha=0.85)

        # Annotate significant bars
        for i, (bar, feat) in enumerate(zip(bars, features_plot)):
            match = [r for r in anova_rows if r["model"] == model and r["feature"] == feat]
            if match and float(match[0]["sd"]) < 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        "✱", ha="center", fontsize=14, fontweight="bold", color="black")

    ax.set_xticks(x + width)
    ax.set_xticklabels([LABEL_FEATURES[f] for f in features_plot], fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("η² (Effect Size)")
    ax.set_title("ANOVA Effect Sizes by Feature and Model\n(✱ = p < 0.05)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(y=0.14, color="gray", linestyle="--", alpha=0.4, linewidth=0.7, label="large effect")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "effect_sizes.png", bbox_inches="tight")
    plt.close()
    print("  fig3: effect_sizes.png")


# ── Figure 4: Kokoro Detail ─────────────────────────────────────────────────

def fig4_kokoro_detail(data):
    """Kokoro shows strongest effect. Zoom into key features."""
    features = ["f0_mean", "speech_rate", "rms_energy", "spectral_centroid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        groups = []
        for cond in CONDITION_ORDER:
            rows = data["kokoro"].get(cond, [])
            vals = [float(r[feat]) for r in rows]
            groups.append(vals)

        positions = np.arange(len(CONDITION_ORDER))
        bp = ax.boxplot(
            groups, positions=positions, widths=0.5,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="black", markersize=6),
            medianprops=dict(color="white", linewidth=1.5),
        )
        for patch, cond in zip(bp["boxes"], CONDITION_ORDER):
            patch.set_facecolor(CONDITION_COLORS[cond])
            patch.set_alpha(0.75)

        for i, (cond, vals) in enumerate(zip(CONDITION_ORDER, groups)):
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color="black", s=20, alpha=0.6, zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER], fontsize=8)
        ax.set_ylabel(LABEL_FEATURES[feat], fontsize=9)
        ax.set_title(LABEL_FEATURES[feat], fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Kokoro: Prior-Context Effect Detail (all features p < 0.001)",
                 fontsize=14, fontweight="bold", color="#9CA3AF", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "kokoro_detail.png", bbox_inches="tight")
    plt.close()
    print("  fig4: kokoro_detail.png")


# ── Figure 5: Summary ───────────────────────────────────────────────────────

def fig5_summary(anova_rows, pairwise_rows):
    """Per-model summary: which pairwise comparisons are significant."""
    significant_pairs = defaultdict(list)
    for row in pairwise_rows:
        p_bonf = float(row["p_bonferroni_6"])
        if p_bonf < 0.05:
            significant_pairs[row["model"]].append(
                (row["feature"], row["condition_a"], row["condition_b"], p_bonf)
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: significance count per model per feature
    ax = axes[0]
    features_plot = [
        "f0_mean", "f0_std", "f0_range",
        "speech_rate", "pause_count", "pause_duration_mean",
    ]
    y_positions = np.arange(len(features_plot))
    width = 0.25

    for j, model in enumerate(MODEL_ORDER):
        counts = []
        for feat in features_plot:
            sig = [
                pair for pair in significant_pairs.get(model, [])
                if pair[0] == feat
            ]
            counts.append(len(sig))
        ax.barh(y_positions + j * width, counts, width,
                label=MODEL_LABELS[model], color=MODEL_COLORS[model], alpha=0.85)

    ax.set_yticks(y_positions + width)
    ax.set_yticklabels([LABEL_FEATURES[f] for f in features_plot], fontsize=9)
    ax.set_xlabel("Significant Pairwise Comparisons (out of 6)")
    ax.set_title("Significant Condition Pairs\n(Bonferroni-corrected p < 0.05)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 6)

    # Right: cold vs primed comparison (F0)
    ax2 = axes[1]
    for j, model in enumerate(MODEL_ORDER):
        pairwise_model = [r for r in pairwise_rows if r["model"] == model and r["feature"] == "f0_mean"]
        cold_vs = [(r["condition_b"], float(r["cohens_d"]), float(r["p_bonferroni_6"]))
                   for r in pairwise_model if r["condition_a"] == "cold"]
        for k, (cond, d, p_b) in enumerate(cold_vs):
            color = "#4CAF50" if p_b < 0.05 else "#BDBDBD"
            ax2.bar(j + k * 0.25, abs(d), 0.2, color=color, alpha=0.85,
                    label=f"{MODEL_LABELS[model]} vs {cond}" if j == 0 and k == 0 else "")

    ax2.set_xticks(np.arange(len(MODEL_ORDER)))
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
    ax2.set_ylabel("|Cohen's d| (cold vs primed)")
    ax2.set_title("F0 Mean: Effect Size vs Cold\n(green = significant after correction)", fontweight="bold")
    ax2.axhline(y=0.2, color="gray", linestyle="--", alpha=0.3, linewidth=0.7, label="small effect")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.7, label="medium effect")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Prior-Context Probe: Significance Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "significance.png", bbox_inches="tight")
    plt.close()
    print("  fig5: significance.png")


def main():
    data = load_features_csv()
    anova_rows = load_anova()
    pairwise_rows = load_pairwise()

    print("Generating figures...")
    fig1_f0_mean(data)
    fig2_speech_pause(data)
    fig3_effects(anova_rows)
    fig4_kokoro_detail(data)
    fig5_summary(anova_rows, pairwise_rows)

    print(f"\nAll figures saved to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
