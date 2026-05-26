#!/usr/bin/env python3
"""Generate figures for punctuation sensitivity report."""
import json
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    FEATURES_CSV,
    FIG_DIR,
    MODEL_ORDER,
    MODEL_LABELS,
    MODEL_COLORS,
    PUNCT_ORDER,
    PUNCT_LABELS,
    HIERARCHY_ORDER,
    HIERARCHY_LABELS,
    CATEGORIES,
    CATEGORY_LABELS,
    load_csv,
    safe_float,
)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "figure.dpi": 150})


def load_data():
    rows = list(load_csv(FEATURES_CSV))
    data = defaultdict(list)
    for r in rows:
        data[r["model"]].append(r)
    return data


def fig1_sentence_end(data):
    """F0 slope by terminal punctuation type, per model, with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    for j, punct in enumerate(PUNCT_ORDER):
        means, errs = [], []
        for model in MODEL_ORDER:
            se_items = [r for r in data[model] if r["category"] == "sentence_end" and r["punct_type"] == punct]
            slopes = [safe_float(r["terminal_f0_slope"]) for r in se_items]
            slopes = [s for s in slopes if s is not None]
            means.append(np.mean(slopes) if slopes else 0)
            errs.append(np.std(slopes) / np.sqrt(len(slopes)) if len(slopes) > 1 else 0)  # SEM
        bars = ax.bar(x + j * width, means, width, label=PUNCT_LABELS[punct],
                      color=["#2196F3", "#FF9800", "#F44336"][j], alpha=0.85,
                      yerr=errs, capsize=3, error_kw={"linewidth": 1.5})

    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Terminal F0 Slope (Hz/s)")
    ax.set_title("Sentence-End F0 Slope by Punctuation Type\n(bars = SEM)")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Subplot 2: question-vs-period difference with effect size annotation
    ax2 = axes[1]
    diffs, errs2 = [], []
    for model in MODEL_ORDER:
        se_items = [r for r in data[model] if r["category"] == "sentence_end"]
        period_slopes = [safe_float(r["terminal_f0_slope"]) for r in se_items if r["punct_type"] == "period"]
        question_slopes = [safe_float(r["terminal_f0_slope"]) for r in se_items if r["punct_type"] == "question"]
        period_slopes = [s for s in period_slopes if s is not None]
        question_slopes = [s for s in question_slopes if s is not None]
        diff = np.mean(question_slopes) - np.mean(period_slopes) if period_slopes and question_slopes else 0
        # Pooled SEM
        se_p = np.std(period_slopes) / np.sqrt(len(period_slopes)) if len(period_slopes) > 1 else 0
        se_q = np.std(question_slopes) / np.sqrt(len(question_slopes)) if len(question_slopes) > 1 else 0
        diffs.append(diff)
        errs2.append(np.sqrt(se_p**2 + se_q**2))

    colors = ["#4CAF50" if d > 20 else "#FFC107" if d > 0 else "#F44336" for d in diffs]
    bars = ax2.bar(range(len(MODEL_ORDER)), diffs, color=colors, alpha=0.85,
                   yerr=errs2, capsize=3, error_kw={"linewidth": 1.5})
    ax2.set_xticks(range(len(MODEL_ORDER)))
    ax2.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax2.set_ylabel("Question - Period F0 Slope (Hz/s)")
    ax2.set_title("Question-vs-Period Differentiation\n(positive = rising question intonation)")
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax2.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, diffs):
        offset = 5 if d >= 0 else -15
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f"{d:.0f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "sentence_end.png", bbox_inches="tight")
    plt.close()
    print("  fig1: sentence_end.png")


def fig2_pause_hierarchy(data):
    """Pause duration by punctuation type, per model (3-panel)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        ph_items = [r for r in data[model] if r["category"] == "pause_hierarchy"]
        by_punct = defaultdict(list)
        for r in ph_items:
            internal = json.loads(r["internal_pause_durations_ms"])
            by_punct[r["punct_type"]].extend(internal)

        positions, means = [], []
        for i, punct in enumerate(HIERARCHY_ORDER):
            durs = by_punct.get(punct, [])
            if durs:
                positions.append(i)
                means.append(np.mean(durs))
                bp = ax.boxplot(durs, positions=[i], widths=0.5, patch_artist=True,
                                boxprops=dict(facecolor=MODEL_COLORS[model], alpha=0.6),
                                medianprops=dict(color="black", linewidth=2),
                                flierprops=dict(marker="o", markersize=3, alpha=0.5))

        ax.set_xticks(range(len(HIERARCHY_ORDER)))
        ax.set_xticklabels(["Comma", "Semicolon", "Em-dash", "Ellipsis"], fontsize=9)
        ax.set_ylabel("Pause Duration (ms)")
        ax.set_title(MODEL_LABELS[model].split("\n")[0], fontweight="bold", color=MODEL_COLORS[model])
        ax.grid(axis="y", alpha=0.3)

        if means and len(means) == 4:
            correct = sum(1 for i in range(len(means)-1) if means[i] <= means[i+1])
            ax.text(0.98, 0.95, f"Hierarchy: {correct}/3", transform=ax.transAxes,
                    ha="right", va="top", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Pause Hierarchy: Duration by Punctuation Type", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "pause_hierarchy.png", bbox_inches="tight")
    plt.close()
    print("  fig2: pause_hierarchy.png")


def fig3_trailing(data):
    """Ellipsis vs period: F0 slope and pause comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # F0 slope comparison with error bars
    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    for j, (punct, label, hatch) in enumerate([("ellipsis", "Ellipsis (...)", "//"), ("period", "Period (.)", "")]):
        means, errs = [], []
        for model in MODEL_ORDER:
            t_items = [r for r in data[model] if r["category"] == "trailing" and r["punct_type"] == punct]
            slopes = [safe_float(r["terminal_f0_slope"]) for r in t_items]
            slopes = [s for s in slopes if s is not None]
            means.append(np.mean(slopes) if slopes else 0)
            errs.append(np.std(slopes) / np.sqrt(len(slopes)) if len(slopes) > 1 else 0)
        bars = ax.bar(x + j * width, means, width, label=label,
                      color=["#9C27B0", "#607D8B"][j], alpha=0.85, hatch=hatch,
                      yerr=errs, capsize=3, error_kw={"linewidth": 1.5})
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Terminal F0 Slope (Hz/s)")
    ax.set_title("Trailing: F0 Slope\n(ellipsis = trailing fade expected)")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    # Amplitude decay (symlog scale)
    ax2 = axes[1]
    for j, (punct, label, color) in enumerate([("ellipsis", "Ellipsis", "#9C27B0"), ("period", "Period", "#607D8B")]):
        means = []
        for model in MODEL_ORDER:
            t_items = [r for r in data[model] if r["category"] == "trailing" and r["punct_type"] == punct]
            decays = [safe_float(r["amplitude_decay_300ms"]) for r in t_items]
            decays = [d for d in decays if d is not None]
            means.append(np.mean(decays) * 1e6 if decays else 0)
        bars = ax2.bar(x + j * width, means, width, label=label,
                       color=color, alpha=0.85, linewidth=1.5, edgecolor="black")
        for bi, (bar, val) in enumerate(zip(bars, means)):
            bx = bar.get_x() + bar.get_width() / 2
            ax2.annotate(f"{val:.2f}", (bx, val if val != 0 else 1),
                         xytext=(0, 10 if val <= 0 else -14),
                         textcoords="offset points", fontsize=8, ha="center", color=color, fontweight="bold")
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax2.set_ylabel("Amplitude Decay (x10^-6)")
    ax2.set_title("Trailing: Amplitude Fade\n(more negative = stronger fade)")
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.grid(axis="y", alpha=0.3)

    # Pause duration comparison
    ax3 = axes[2]
    for j, (punct, label) in enumerate([("ellipsis", "Ellipsis"), ("period", "Period")]):
        means, errs = [], []
        for model in MODEL_ORDER:
            t_items = [r for r in data[model] if r["category"] == "trailing" and r["punct_type"] == punct]
            pauses = [safe_float(r["best_pause_ms"]) for r in t_items]
            pauses = [p for p in pauses if p is not None]
            means.append(np.mean(pauses) if pauses else 0)
            errs.append(np.std(pauses) / np.sqrt(len(pauses)) if len(pauses) > 1 else 0)
        ax3.bar(x + j * width, means, width, label=label,
                color=["#9C27B0", "#607D8B"][j], alpha=0.85,
                yerr=errs, capsize=3, error_kw={"linewidth": 1.5})
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax3.set_ylabel("Terminal Pause (ms)")
    ax3.set_title("Trailing: Pause Duration")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("Trailing Punctuation: Ellipsis vs Period", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trailing.png", bbox_inches="tight")
    plt.close()
    print("  fig3: trailing.png")


def fig4_quotation(data):
    """Quoted vs reported speech F0 range shift."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    for j, (ptype, label, color) in enumerate([("quoted", "Quoted speech", "#E91E63"), ("reported", "Reported speech", "#795548")]):
        means, errs = [], []
        for model in MODEL_ORDER:
            q_items = [r for r in data[model] if r["category"] == "quotation" and r["punct_type"] == ptype]
            ranges = [safe_float(r["f0_range"]) for r in q_items]
            ranges = [r for r in ranges if r is not None]
            means.append(np.mean(ranges) if ranges else 0)
            errs.append(np.std(ranges) / np.sqrt(len(ranges)) if len(ranges) > 1 else 0)
        ax.bar(x + j * width, means, width, label=label, color=color, alpha=0.85,
               yerr=errs, capsize=3, error_kw={"linewidth": 1.5})

    ax.set_xticks(x + width/2)
    ax.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("F0 Range (Hz)")
    ax.set_title("Quotation Sensitivity: F0 Range Shift\n(quoted speech should have wider F0 range)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "quotation.png", bbox_inches="tight")
    plt.close()
    print("  fig4: quotation.png")


def fig5_summary(data):
    """Summary scatter: model trade-offs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: scatter - question F0 diff vs pause hierarchy
    ax = axes[0]
    q_diffs, h_scores, labels, colors = [], [], [], []
    for model in MODEL_ORDER:
        se_items = [r for r in data[model] if r["category"] == "sentence_end"]
        period_slopes = [safe_float(r["terminal_f0_slope"]) for r in se_items if r["punct_type"] == "period"]
        question_slopes = [safe_float(r["terminal_f0_slope"]) for r in se_items if r["punct_type"] == "question"]
        period_slopes = [s for s in period_slopes if s is not None]
        question_slopes = [s for s in question_slopes if s is not None]
        qd = np.mean(question_slopes) - np.mean(period_slopes) if period_slopes and question_slopes else 0

        ph_items = [r for r in data[model] if r["category"] == "pause_hierarchy"]
        by_p = defaultdict(list)
        for r in ph_items:
            by_p[r["punct_type"]].extend(json.loads(r["internal_pause_durations_ms"]))
        means = [np.mean(by_p.get(p, [0])) for p in HIERARCHY_ORDER]
        hs = sum(1 for i in range(len(means)-1) if means[i] <= means[i+1]) / 3

        q_diffs.append(qd)
        h_scores.append(hs)
        labels.append(MODEL_LABELS[model].split(" ")[0])
        colors.append(MODEL_COLORS[model])

    for i in range(len(MODEL_ORDER)):
        ax.scatter(q_diffs[i], h_scores[i], s=300, c=colors[i], edgecolors="black",
                   linewidth=1.5, zorder=5, alpha=0.9)
        ax.annotate(labels[i], (q_diffs[i], h_scores[i]),
                    textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
    ax.set_xlabel("Question-vs-Period F0 Differentiation (Hz/s)")
    ax.set_ylabel("Pause Hierarchy Score (0-1)")
    ax.set_title("Model Comparison: F0 Cues vs Pause Timing\n(descriptive, n=28)", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.text(0.98, 0.98, "F0 + Pause\n(ideal)", transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.5)
    ax.text(0.02, 0.02, "Neither", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, alpha=0.5)

    # Right: overall pause duration distribution
    ax2 = axes[1]
    model_pauses = []
    for model in MODEL_ORDER:
        all_pauses = []
        for r in data[model]:
            if r["best_pause_ms"] and float(r["best_pause_ms"]) > 0:
                all_pauses.append(float(r["best_pause_ms"]))
        model_pauses.append(all_pauses)

    bp = ax2.boxplot(model_pauses, labels=[MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER],
                     patch_artist=True, widths=0.5)
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Best Pause Duration (ms)")
    ax2.set_title("Terminal Pause Duration Distribution", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=15, labelsize=9)

    plt.suptitle("Punctuation Sensitivity: Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "summary.png", bbox_inches="tight")
    plt.close()
    print("  fig5: summary.png")


def fig6_pause_count(data):
    """Number of pauses per utterance category, per model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.18

    for j, (cat, label) in enumerate(zip(CATEGORIES, [CATEGORY_LABELS[c] for c in CATEGORIES])):
        means, errs = [], []
        for model in MODEL_ORDER:
            items = [r for r in data[model] if r["category"] == cat]
            counts = [int(r["num_pauses"]) for r in items]
            means.append(np.mean(counts) if counts else 0)
            errs.append(np.std(counts) / np.sqrt(len(counts)) if len(counts) > 1 else 0)
        ax.bar(x + j * width, means, width, label=label, alpha=0.8,
               yerr=errs, capsize=2, error_kw={"linewidth": 1.0})

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([MODEL_LABELS[m].split(" ")[0] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Mean Number of Pauses per Utterance")
    ax.set_title("Pause Count by Utterance Category", fontweight="bold")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "pause_count.png", bbox_inches="tight")
    plt.close()
    print("  fig6: pause_count.png")


def main():
    data = load_data()
    print("Generating figures...")

    fig1_sentence_end(data)
    fig2_pause_hierarchy(data)
    fig3_trailing(data)
    fig4_quotation(data)
    fig5_summary(data)
    fig6_pause_count(data)

    print(f"\nAll figures saved to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())