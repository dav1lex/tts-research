#!/usr/bin/env python3
"""Generate grouped bar chart: F0 StdDev per model × condition.

Saves to results/f0_variance_hangover.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

FEATURES_CSV = PROJECT / "features" / "features.csv"
FIG_PATH = PROJECT / "results" / "f0_variance_hangover.png"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_data():
    with open(FEATURES_CSV) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_data()
    if not rows:
        print("ERROR: no data")
        return 1

    # Build DataFrame-like structures
    data = []
    for r in rows:
        data.append({
            "model": r["model"],
            "condition": r["condition"],
            "f0_std": float(r["f0_std"]),
        })

    # Convert to grouped format for seaborn
    import pandas as pd
    df = pd.DataFrame(data)

    # Normalize model names
    model_labels = {
        "chatterbox": "Chatterbox",
        "xtts": "XTTS-v2",
        "kokoro": "Kokoro",
    }
    df["model_label"] = df["model"].map(model_labels)

    # Condition order
    df["condition"] = pd.Categorical(df["condition"], categories=["control", "subliminal"], ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="model_label",
        y="f0_std",
        hue="condition",
        palette={"control": "#4CAF50", "subliminal": "#FF5722"},
        ax=ax,
        capsize=0.1,
        errwidth=1.5,
        alpha=0.85,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("F0 Standard Deviation (Hz)")
    ax.set_title("F0 Variance Hangover: Control vs Subliminal Prime")

    # Add individual points
    sns.stripplot(
        data=df,
        x="model_label",
        y="f0_std",
        hue="condition",
        palette={"control": "#1B5E20", "subliminal": "#BF360C"},
        dodge=True,
        ax=ax,
        size=5,
        alpha=0.6,
        jitter=True,
        legend=False,
    )

    # Clean up duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates from stripplot
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Condition")

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_PATH), dpi=150)
    print(f"Figure saved to {FIG_PATH}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
