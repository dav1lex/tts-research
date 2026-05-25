#!/usr/bin/env python3
"""Plot identity drift results.

Plots:
1. Drift over time per model (line plot)
2. Per-feature drift heatmap
3. Early vs late drift bar chart

Kokoro is styled distinctly as a no-adaptation baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paths ---
PROJECT = Path("/home/davilex/tts-research/_4-identity-drift")
RESULTS = PROJECT / "results"
FIGURES_DIR = RESULTS / "figures"

FEATURE_COLS = [
    "f0_mean", "f0_std", "cpp_mean", "spectral_flatness",
    "spectral_tilt_ratio", "mfcc_mean", "rms_mean", "spectral_centroid",
]

MODELS = ["chatterbox", "xtts", "kokoro"]
COLORS = {"chatterbox": "#1f77b4", "xtts": "#ff7f0e", "kokoro": "#2ca02c"}
FEATURE_LABELS = {
    "f0_mean": "F0 Mean",
    "f0_std": "F0 Std",
    "cpp_mean": "CPP Mean",
    "spectral_flatness": "Spectral Flatness",
    "spectral_tilt_ratio": "Spectral Tilt Ratio",
    "mfcc_mean": "MFCC Mean",
    "rms_mean": "RMS Mean",
    "spectral_centroid": "Spectral Centroid",
}


def get_model_label(model: str) -> str:
    """Return display label for model, with baseline annotation for kokoro."""
    if model == "kokoro":
        return "kokoro (no-adaptation baseline)"
    return model


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    drift_path = RESULTS / "drift_by_window.csv"
    summary_path = RESULTS / "drift_summary.csv"

    if not drift_path.exists():
        print(f"ERROR: {drift_path} not found. Run measure_drift.py first.", file=sys.stderr)
        return 1

    drift_df = pd.read_csv(drift_path)
    print(f"Loaded {len(drift_df)} drift rows from {drift_path}")

    summary_df = None
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        print(f"Loaded summary from {summary_path}")

    # ── Plot 1: Drift over time ──
    fig, ax = plt.subplots(figsize=(12, 6))

    for model in MODELS:
        model_df = drift_df[drift_df["model"] == model].sort_values("window_idx")
        if len(model_df) == 0:
            continue

        time_minutes = model_df["time_start"] / 60.0
        drifts = model_df["drift_from_reference"].values

        linestyle = "--" if model == "kokoro" else "-"
        marker = "s" if model == "kokoro" else "o"
        markersize = 5 if model == "kokoro" else 4
        alpha = 0.7

        ax.plot(time_minutes, drifts,
                color=COLORS[model],
                linestyle=linestyle,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                label=get_model_label(model))

        # Add drift slope annotation if available
        if summary_df is not None:
            summary_row = summary_df[summary_df["model"] == model]
            if len(summary_row) > 0:
                slope = summary_row.iloc[0]["drift_slope"]
                if not np.isnan(slope):
                    ax.annotate(
                        f"slope={slope:.4f}",
                        xy=(time_minutes.iloc[-1], drifts[-1]),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=8, color=COLORS[model],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Drift from Reference (scaled abs delta)")
    ax.set_title("Identity Drift Over 5-Minute Monologue")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_over_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'drift_over_time.png'}")

    # ── Plot 2: Per-feature drift heatmap ──
    feature_drift_data = []
    for model in MODELS:
        model_df = drift_df[drift_df["model"] == model]
        if len(model_df) == 0:
            continue
        means = {}
        for col in FEATURE_COLS:
            col_name = f"drift_{col}"
            if col_name in model_df.columns:
                vals = model_df[col_name].dropna()
                means[col] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        feature_drift_data.append({**{"model": get_model_label(model)}, **means})

    heatmap_df = pd.DataFrame(feature_drift_data)
    if len(heatmap_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        heatmap_matrix = heatmap_df.set_index("model")[[c for c in FEATURE_COLS if c in heatmap_df.columns]]
        labels = [FEATURE_LABELS.get(c, c) for c in heatmap_matrix.columns]

        im = ax.imshow(heatmap_matrix.values, aspect="auto", cmap="YlOrRd")

        # Annotate cells
        for i in range(len(heatmap_matrix)):
            for j in range(len(heatmap_matrix.columns)):
                val = heatmap_matrix.iloc[i, j]
                text = f"{val:.3f}" if not np.isnan(val) else "nan"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="black" if (not np.isnan(val) and val < np.nanmax(heatmap_matrix.values) * 0.6) else "white")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap_matrix.index)))
        ax.set_yticklabels(heatmap_matrix.index)
        ax.set_title("Per-Feature Drift Heatmap (mean scaled abs delta)")
        fig.colorbar(im, ax=ax, label="Drift")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "drift_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIGURES_DIR / 'drift_heatmap.png'}")

    # ── Plot 3: Early vs Late drift bar chart ──
    if summary_df is not None and len(summary_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(summary_df))
        width = 0.35

        early_vals = summary_df["drift_early"].values
        late_vals = summary_df["drift_late"].values
        labels = [get_model_label(m) for m in summary_df["model"]]

        bars1 = ax.bar(x - width / 2, early_vals, width, label="Early (first windows)",
                       color="#4c72b0", edgecolor="black")
        bars2 = ax.bar(x + width / 2, late_vals, width, label="Late (last windows)",
                       color="#c44e52", edgecolor="black")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Model")
        ax.set_ylabel("Mean Drift from Reference")
        ax.set_title("Early vs Late Drift Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "early_vs_late_drift.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIGURES_DIR / 'early_vs_late_drift.png'}")

    print("\nAll plots generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
