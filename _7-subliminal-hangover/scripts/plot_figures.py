#!/usr/bin/env python3
"""Generate two-panel figure: f0_cv (primary) + speaking_rate (control).

Panel A: F0 Coefficient of Variation by model × condition
Panel B: Speaking Rate (syllables/sec) by model × condition

If Panel A shows a drop for numbers but Panel B is flat, the hangover is clean.
Saves to results/f0_variance_hangover.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

FEATURES_CSV = PROJECT / "features" / "features.csv"
FIG_PATH = PROJECT / "results" / "f0_variance_hangover.png"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

CONDITION_PALETTE = {
    "control": "#4CAF50",
    "noun": "#2196F3",
    "subliminal": "#FF5722",
}

CONDITION_ORDER = ["control", "noun", "subliminal"]
CONDITION_LABELS = {
    "control": "Nature (short)",
    "noun": "Nouns (length-matched)",
    "subliminal": "Numbers (robotic)",
}

MODEL_LABELS = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro",
}


def main():
    df = pd.read_csv(FEATURES_CSV)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["condition"] = pd.Categorical(
        df["condition"], categories=CONDITION_ORDER, ordered=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: f0_cv ─────────────────────────────────────────────
    sns.barplot(
        data=df,
        x="model_label",
        y="f0_cv",
        hue="condition",
        hue_order=CONDITION_ORDER,
        palette=CONDITION_PALETTE,
        ax=ax1,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        alpha=0.85,
    )
    sns.stripplot(
        data=df,
        x="model_label",
        y="f0_cv",
        hue="condition",
        hue_order=CONDITION_ORDER,
        palette=CONDITION_PALETTE,
        dodge=True,
        ax=ax1,
        size=5,
        alpha=0.4,
        jitter=True,
        legend=False,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("F0 Coefficient of Variation (f0_std / f0_mean)")
    ax1.set_title("A: Pitch Variance (Primary)")

    # ── Panel B: speaking_rate ─────────────────────────────────────
    sns.barplot(
        data=df,
        x="model_label",
        y="speaking_rate",
        hue="condition",
        hue_order=CONDITION_ORDER,
        palette=CONDITION_PALETTE,
        ax=ax2,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        alpha=0.85,
    )
    sns.stripplot(
        data=df,
        x="model_label",
        y="speaking_rate",
        hue="condition",
        hue_order=CONDITION_ORDER,
        palette=CONDITION_PALETTE,
        dodge=True,
        ax=ax2,
        size=5,
        alpha=0.4,
        jitter=True,
        legend=False,
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("Speaking Rate (syllables / sec)")
    ax2.set_title("B: Tempo (Control)")

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CONDITION_PALETTE["control"], alpha=0.85, label=CONDITION_LABELS["control"]),
        Patch(facecolor=CONDITION_PALETTE["noun"], alpha=0.85, label=CONDITION_LABELS["noun"]),
        Patch(facecolor=CONDITION_PALETTE["subliminal"], alpha=0.85, label=CONDITION_LABELS["subliminal"]),
    ]
    fig.legend(
        handles=legend_elements,
        title="Prime Condition",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
    )

    plt.tight_layout()
    # Adjust for legend
    plt.subplots_adjust(bottom=0.18)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_PATH), dpi=150, bbox_inches="tight")
    print(f"Figure saved to {FIG_PATH}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
