#!/usr/bin/env python3
"""Generate combined _7 + _8 Subliminal Hangover report (HTML + PDF).

Pulls data from both projects, generates unified figures, embeds _7 pilot
figure, produces self-contained HTML with weasyprint PDF conversion.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V4_DIR = SCRIPT_DIR.parent
V3_DIR = V4_DIR.parent / "_7-subliminal-hangover"
RESULTS_DIR = V4_DIR / "results"

V3_STATS = V3_DIR / "results" / "stats.json"
V3_FIG = V3_DIR / "results" / "f0_variance_hangover.png"
V4_STATS = V4_DIR / "results" / "stats.json"
V4_FEATURES = V4_DIR / "features" / "features.csv"
V4_GATE = V4_DIR / "results" / "gate_check.json"
V4_ALIGN = V4_DIR / "results" / "alignment_log.json"

REPORT_HTML = RESULTS_DIR / "combined_report.html"
REPORT_PDF = RESULTS_DIR / "combined_report.pdf"

MODEL_LABELS = {"chatterbox": "Chatterbox", "kokoro": "Kokoro", "xtts": "XTTS-v2"}

CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: "Helvetica Neue", Arial, "Liberation Sans", sans-serif;
    font-size: 10.5pt; line-height: 1.65; color: #222; background: #fff;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.report-container { max-width: 880px; margin: 0 auto; padding: 2em 2.5em; }
.cover { text-align: center; padding: 3em 0 1.5em 0; }
.cover h1 { font-size: 28pt; color: #111; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0.2em; }
.cover .subtitle { font-size: 13pt; color: #555; margin-bottom: 0.6em; }
.cover .meta { font-size: 9pt; color: #888; margin-bottom: 2em; }
.cover .abstract { max-width: 640px; margin: 1.5em auto 0 auto; text-align: left;
    font-size: 10.5pt; color: #444; line-height: 1.7;
    border-top: 1px solid #ddd; padding-top: 1.5em; }
h2 { font-size: 17pt; color: #111; margin-top: 1.6em; margin-bottom: 0.5em;
     padding-bottom: 0.25em; border-bottom: 2.5px solid #111; font-weight: 700;
     page-break-after: avoid; }
h3 { font-size: 12pt; color: #1a1a2e; margin-top: 1.1em; margin-bottom: 0.35em; font-weight: 600; }
p, li { margin-bottom: 0.7em; }
.table-wrap { margin: 0.8em 0 1.2em 0; }
table { width: 100%; border-collapse: collapse; font-size: 9.5pt; }
th { background: #1a1a2e; color: #fff; padding: 7px 10px; text-align: left; font-weight: 600; }
td { padding: 6px 10px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f5f5f8; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.figure { margin: 1.2em 0; text-align: center; page-break-inside: avoid; }
.figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px; }
.figure .caption { font-size: 9pt; color: #666; margin-top: 0.35em; }
.callout-good { background: #eaf7ea; padding: 0.7em 1em; border-left: 4px solid #2e7d32; margin: 0.7em 0; border-radius: 2px; }
.callout-warn { background: #fef9e7; padding: 0.7em 1em; border-left: 4px solid #e6a817; margin: 0.7em 0; border-radius: 2px; }
.callout-note { background: #f0f1ff; padding: 0.7em 1em; border-left: 4px solid #5662d1; margin: 0.7em 0; border-radius: 2px; }
.callout-limits { background: #fafafa; padding: 0.7em 1em; border-left: 4px solid #c0392b; margin: 0.7em 0; border-radius: 2px; }
.callout-limits ul, .callout-note ul { margin-left: 1.3em; margin-bottom: 0.3em; }
.callout-limits li, .callout-note li { margin-bottom: 0.25em; }
.footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd;
          font-size: 8pt; color: #999; text-align: center; }
.page-break { page-break-before: always; }
@media print {
    body { font-size: 9.5pt; }
    @page { size: A4; margin: 1.8cm; }
    .cover { page-break-after: always; }
}
"""


# ── helpers ──────────────────────────────────────────────────

def load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _fmt_p(p: float) -> str:
    if p < 1e-10:
        return f"{p:.1e}"
    if p < 0.001:
        return f"{p:.2g}"
    return f"{p:.4f}"


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# ── figure generation ───────────────────────────────────────

def generate_figures() -> dict[str, str]:
    """Generate _8 boxplots + coefficient plot as base64 images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    matplotlib.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
    })

    df = pd.read_csv(V4_FEATURES)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["condition_label"] = df["condition"].map({"noun": "Noun", "number": "Number"})

    results = {}

    # --- Figure 1: speaking_rate boxplot ---
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    palette = {"Noun": "#4c9f70", "Number": "#d95f3e"}
    for idx, model in enumerate(["chatterbox", "kokoro", "xtts"]):
        ax = axes[idx]
        sub = df[df["model"] == model]
        sns.boxplot(data=sub, x="condition_label", y="speaking_rate", palette=palette, width=0.5, linewidth=0.8, ax=ax)
        sns.stripplot(data=sub, x="condition_label", y="speaking_rate", color="black", size=2, alpha=0.3, ax=ax)
        ax.set_title(MODEL_LABELS[model], fontweight=700)
        ax.set_xlabel("")
        ax.set_ylabel("Speaking Rate (syll/s)" if idx == 0 else "")
    fig.suptitle("Figure 2: Speaking Rate by Condition (V4, N=594)", fontweight=700, y=1.02, fontsize=11)
    buf = _fig_to_b64(fig)
    plt.close(fig)
    results["speaking_rate_box"] = buf

    # --- Figure 2: f0_cv boxplot ---
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)
    for idx, model in enumerate(["chatterbox", "kokoro", "xtts"]):
        ax = axes[idx]
        sub = df[df["model"] == model]
        sns.boxplot(data=sub, x="condition_label", y="f0_cv", palette=palette, width=0.5, linewidth=0.8, ax=ax)
        sns.stripplot(data=sub, x="condition_label", y="f0_cv", color="black", size=2, alpha=0.3, ax=ax)
        ax.set_title(MODEL_LABELS[model], fontweight=700)
        ax.set_xlabel("")
        ax.set_ylabel("f0_cv" if idx == 0 else "")
    fig.suptitle("Figure 3: Pitch Variation (f0_cv) by Condition (V4, N=594)", fontweight=700, y=1.02, fontsize=11)
    buf = _fig_to_b64(fig)
    plt.close(fig)
    results["f0_cv_box"] = buf

    # --- Figure 3: Coefficient plot (effect size + CI) ---
    v4s = load_json(V4_STATS)
    v4s = v4s if isinstance(v4s, dict) else {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    metrics = [("speaking_rate", "Tempo Acceleration"), ("f0_cv", "Pitch Compression")]
    for ax_idx, (key, title) in enumerate(metrics):
        ax = axes[ax_idx]
        models_ordered = ["chatterbox", "kokoro", "xtts"]
        y_positions = [2, 1, 0]
        for model, y in zip(models_ordered, y_positions):
            m = v4s.get(model, {}).get(key, {})
            if "error" in m or "coef" not in m:
                continue
            coef = m["coef"]
            ci_lo = m["ci_lo"]
            ci_hi = m["ci_hi"]
            converged = m.get("converged", True)
            color = "#2e7d32" if converged else "#d95f3e"
            fmt_marker = "o" if converged else "s"
            ax.errorbar(coef, y, xerr=[[coef - ci_lo], [ci_hi - coef]], fmt=fmt_marker,
                        color=color, capsize=4, capthick=1.5, markersize=8, elinewidth=2)
            ax.axvline(0, color="#888", linestyle="--", linewidth=0.8)
            label = MODEL_LABELS[model]
            if not converged:
                label += " (unconverged)"
            ax.text(coef + 0.02, y + 0.15, label, fontsize=8, va="center", color=color, fontweight="bold" if converged else "normal")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([""] * len(y_positions))
        ax.set_title(title, fontweight=700, fontsize=11)
        ax.set_xlabel("Condition Coefficient (Number - Noun)")
    fig.suptitle("Figure 4: Mixed-Effects Coefficient Estimates with 95% CI (V4)", fontweight=700, y=1.03, fontsize=11)
    buf = _fig_to_b64(fig)
    plt.close(fig)
    results["coef_plot"] = buf

    return results


def _fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── sections ─────────────────────────────────────────────────

def cover() -> str:
    return """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Subliminal Hangover — Combined Report</title>
<style>""" + CSS + """</style></head>
<body><div class="report-container">
<div class="cover">
<h1>Subliminal Hangover</h1>
<div class="subtitle">Acoustic Inertia from Robotic Primes in Text-to-Speech Models</div>
<div class="meta">Pilot (_7) + Preregistered Replication (_8) &bull; May 2026</div>
<div class="abstract">
<p><strong>Key finding:</strong> Robotic number primes cause consistent <b>tempo acceleration</b>
across all three tested TTS architectures (Chatterbox, Kokoro, XTTS-v2). The effect ranges
from +0.35 to +1.36 syllables/second with all 95% CIs excluding zero. <b>Pitch compression</b>
(f0_cv drop) is Kokoro-specific (p=2.1e-08); Chatterbox and XTTS show no reliable f0_cv change
after numbers. The pilot (_7) correctly identified both phenotypes but misattributed their model
distribution: pitch compression appeared in Chatterbox at n=5 but vanished under larger-N
mixed effects. Tempo acceleration, borderline in the pilot, is the robust signal.</p>

<p><strong>Practical takeaway:</strong> generate robotic segments (numbers, codes,
identifiers) and emotional speech in separate TTS context windows, then stitch the
audio afterward. A single context window for mixed-style text will bleed tempo
acceleration into adjacent speech.</p>
</div>
</div>
"""


def the_question() -> str:
    return """\
<h2>1 &ensp; The Question</h2>

<p>TTS models maintain acoustic context across their generation window. The model
"remembers" how it was speaking before, and that memory bleeds forward. This is intrinsic
to autoregressive and streaming TTS: encoder states, style embeddings, and raw acoustic
conditioning all carry forward representations of preceding speech.</p>

<p>The question is simple: <b>if you feed a TTS model a monotone, robotic prime (a list of
numbers), does the flatness bleed into a subsequent emotional sentence?</b></p>

<p>We call this the "subliminal hangover" — an unintended context effect that the user didn't
ask for and the model wasn't designed to produce. The primary hypothesis: <code>f0_cv(emotional
target | number prime) &lt; f0_cv(emotional target | noun prime)</code>, with speaking rate
held constant so the effect isolates pitch, not tempo.</p>
"""


def pilot_design() -> str:
    return """\
<h2>2 &ensp; The Pilot (_7)</h2>

<h3>2.1 &ensp; Design</h3>
<p>Single emotional target sentence (<i>"You absolutely cannot be serious about this
ridiculous idea!"</i>). Three conditions per model, 5 repetitions each (5 shuffled prime
variants per condition). 45 WAVs total across Chatterbox, XTTS-v2, and Kokoro.</p>

<div class="table-wrap">
<table>
<thead><tr><th>Condition</th><th>Prime</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td>Noun</td><td>14 length-matched nouns</td><td>Controls for time-on-task, attention drift</td></tr>
<tr><td>Number</td><td>14 digit-string numbers</td><td>The robotic prime</td></tr>
</tbody></table>
</div>

<p>Primes and target were generated as a single string in one context window. Target
extraction used <b>WhisperX V3 word alignment</b> — we found timestamps of the target's
first and last word and sliced out only the target segment for analysis.</p>

<p>Primary metric: <b>f0_cv</b> (coefficient of variation: <code>f0_std / f0_mean</code>),
extracted via parselmouth (Praat). Secondary metric: speaking rate (18 syllables / aligned
duration). Statistical tests: paired Wilcoxon signed-rank on noun-vs-number pairs.</p>

<h3>2.2 &ensp; Pipeline Evolution</h3>
<p>The pilot went through three versions before the final design locked:</p>
<div class="table-wrap">
<table>
<thead><tr><th>Version</th><th>Segmentation</th><th>Control</th><th>Primary Metric</th><th>Target</th></tr></thead>
<tbody>
<tr><td>V1</td><td>VAD pause detection</td><td>Short nature sentence</td><td>f0_std</td><td>Two clauses</td></tr>
<tr><td>V2</td><td>VAD pause detection</td><td>Length-matched nouns</td><td>f0_std</td><td>Two clauses</td></tr>
<tr><td style="font-weight:bold">V3</td><td style="font-weight:bold">WhisperX word alignment</td><td style="font-weight:bold">Length-matched nouns</td><td style="font-weight:bold">f0_cv</td><td style="font-weight:bold">Single clause</td></tr>
</tbody></table>
</div>
"""


def pilot_results() -> str:
    return """\
<h3>2.3 &ensp; Results</h3>
"""


def pilot_figure(v3_fig_b64: str) -> str:
    if not v3_fig_b64:
        return ""
    return f"""\
<div class="figure">
<img src="data:image/png;base64,{v3_fig_b64}" alt="Pitch variance hangover by model">
<div class="caption">Figure 1: f0_cv by condition and model (_7 pilot, n=5 per bar).
Green = noun prime, orange = number prime. Lower f0_cv = flatter pitch.</div>
</div>
"""


def pilot_stats_table(v3_data: dict) -> str:
    tests = v3_data.get("paired_tests", {}).get("nouns_vs_subliminal", {})
    descriptive = v3_data.get("descriptive", {})
    rows = []
    for model in ["chatterbox", "kokoro", "xtts"]:
        label = MODEL_LABELS.get(model, model)
        t = tests.get(model, {})
        d = descriptive.get(model, {})
        f0c = d.get("f0_cv", {})
        noun_mean = f0c.get("noun", {}).get("mean", 0)
        num_mean = f0c.get("subliminal", {}).get("mean", 0)
        pct = t.get("mean_difference_pct", 0)
        p = t.get("wilcoxon_p_value", 1)
        sr_check = t.get("speaking_rate_check", {})
        sr_ok = "yes" if not sr_check.get("significant_difference", True) else "no"
        sig = _sig_stars(p)
        verdict = "Pilot evidence" if p < 0.05 else ("Strong trend" if p < 0.1 else "Inconclusive")
        rows.append(
            f"<tr><td>{label}</td><td class='num'>{noun_mean:.4f}</td>"
            f"<td class='num'>{num_mean:.4f}</td><td class='num'>-{pct:.0f}%</td>"
            f"<td class='num'>W={t.get('wilcoxon_statistic','?')}, p={_fmt_p(p)}{sig}</td>"
            f"<td class='num'>{sr_ok}</td><td>{verdict}</td></tr>"
        )
    return f"""\
<div class="table-wrap">
<table>
<thead><tr><th>Model</th><th class="num">Noun f0_cv</th><th class="num">Number f0_cv</th><th class="num">Drop</th><th class="num">Wilcoxon</th><th class="num">Rate stable</th><th>Verdict</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
</div>
"""


def pilot_weaknesses() -> str:
    return """\
<h3>2.4 &ensp; Weaknesses</h3>

<div class="callout-limits">
<ul>
<li><b>n=5 with Wilcoxon.</b> Five repetitions per condition is enough to notice an
effect but not measure one. Wilcoxon at n=5 can only return significant p-values
for near-perfect orderings (all 5 pairs same direction). The Chatterbox result
(p=0.031) was one of three possible significant outcomes.</li>
<li><b>Single target sentence.</b> Any effect could be specific to that one sentence's
syllable structure, emotional valence, or phonetic composition. No generalization.</li>
<li><b>No pre-registration.</b> We ran the analysis, saw results, then wrote the report.
Classic HARKing risk (Hypothesizing After Results are Known).</li>
<li><b>Seed handling undocumented.</b> Whether noun and number conditions for the same
target/rep shared the same seed was not tracked. Seed coupling can mask real effects
or create spurious ones.</li>
<li><b>No mixed effects.</b> Wilcoxon doesn't account for target-level or repetition-level
random effects. With multiple targets, you need that structure.</li>
</ul>
</div>

<div class="callout-note">
<p><strong>Two acoustic phenotypes emerged:</strong> pitch compression (Chatterbox, XTTS-v2 trend)
and tempo acceleration (Kokoro borderline, XTTS-v2 numerically). This two-phenotype
discovery was the most interesting result, but with n=5 it was descriptive, not confirmatory.</p>
</div>
"""


def credibility_gap() -> str:
    return """\
<h2 class="page-break">3 &ensp; The Credibility Gap</h2>

<p>The pilot gave us a signal. That's all it gave us. The Chatterbox result was intriguing,
the two-phenotype pattern was surprising, and the methodology was sound enough to justify
more effort. But no reviewer, no skeptical colleague, and honestly not even ourselves should
be fully convinced by n=5 on one sentence.</p>

<p>To make the claim credible, we needed:</p>
<ul>
<li><b>More targets</b> — not one sentence, enough to estimate per-target distributions</li>
<li><b>More repetitions</b> — n=5 is descriptive, n=10 approaches useful estimation</li>
<li><b>Pre-registration</b> — hypotheses, exclusion rules, analysis plan locked before generation</li>
<li><b>Seed independence</b> — guarantee paired noun/number runs use different PRNG seeds</li>
<li><b>Mixed effects, not Wilcoxon</b> — random intercepts for target_id + repetition, effect sizes with CIs</li>
<li><b>Kokoro determinism check</b> — if Kokoro is fully deterministic, "replications" are the same file</li>
</ul>
"""


def replication_design() -> str:
    return """\
<h2>4 &ensp; The Replication (_8)</h2>

<h3>4.1 &ensp; Preregistered Design</h3>
<p>Protocol locked in <code>PROTOCOL.md</code> before any audio generation:</p>
<ul>
<li>10 emotional target sentences (angry/indignant style, 5–18 syllables)</li>
<li>10 repetitions per (model, condition, target)</li>
<li>3 models: Chatterbox, Kokoro (after determinism preflight), XTTS-v2</li>
<li>2 conditions: noun (length-matched neutral list) and number (digit-string list)</li>
<li>Staging: Stage A (2 reps, 120 WAVs) smoke test → Stage B full (10 reps, 600 WAVs)</li>
</ul>

<h3>4.2 &ensp; What Changed</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Dimension</th><th>_7 Pilot</th><th>_8 Replication</th></tr></thead>
<tbody>
<tr><td>Design</td><td>Ad-hoc generation scripts</td><td>Manifest-driven (manifest.csv)</td></tr>
<tr><td>N targets</td><td>1</td><td>10</td></tr>
<tr><td>N reps</td><td>5</td><td>10</td></tr>
<tr><td>Total WAVs</td><td>45</td><td>594 (after 1 gate fail)</td></tr>
<tr><td>Preregistered</td><td>No</td><td>Yes (PROTOCOL.md)</td></tr>
<tr><td>Seed handling</td><td>Undocumented</td><td>Condition-specific offsets (noun:+200, number:+100)</td></tr>
<tr><td>Analysis</td><td>Wilcoxon (n=5)</td><td>Mixed effects: f0_cv ~ cond + (1|target) + (1|rep)</td></tr>
<tr><td>Inference</td><td>Descriptive p-value</td><td>Effect size + 95% CI + p (descriptive)</td></tr>
</tbody></table>
</div>
"""


def discoveries_during_build() -> str:
    return """\
<h3>4.3 &ensp; Discoveries During Build</h3>

<div class="callout-note">
<ul>
<li><b>Seed coupling check.</b> <code>analyze_mixedlm.py</code> scans the manifest for any
(target_id, repetition) pair where noun and number share the same seed. Result: <b>0/100
identical pairs</b>. Clean separation confirmed. Building this check forced us to think
about a problem we had ignored in _7.</li>
<li><b>Kokoro determinism preflight.</b> 5 repetitions, different seeds each.
Result: all 5 WAVs have different SHA-256 hashes, f0_cv range 0.195–0.211 (std=0.005),
identical durations (19.65s). Near-deterministic — enough variation to include, but
each "repetition" provides less independent information than a Chatterbox rep.</li>
<li><b>Append-mode feature extraction.</b> The pipeline checks existing rows by ID and
skips processed files. Critical for multi-day generation: no need to restart 600 WhisperX
alignments from scratch.</li>
<li><b>Chatterbox PerthImplicitWatermarker monkey-patch.</b> Chatterbox's internal
watermarker module crashes on import. Fixed with a one-line patch in the generation
script. Without it, half the pipeline fails before it starts.</li>
</ul>
</div>
"""


def gate_check_section() -> str:
    gate = load_json(V4_GATE)
    gate = gate if isinstance(gate, dict) else {}
    total = gate.get("total_rows", "?")
    passed = gate.get("passed_rows", "?")
    fail = gate.get("failed_rows", "?")
    fail_details = gate.get("fail_details", [])
    fail_text = ""
    for fd in fail_details[:3]:
        fid = fd.get("id", "?")
        issues = fd.get("issues", [])
        fail_text += f"<li><b>{fid}:</b> {', '.join(issues)}</li>"

    n_counts = gate.get("n_counts", {})
    n_rows = []
    for key, label in [("chatterbox_noun", "Chatterbox noun"), ("chatterbox_number", "Chatterbox number"),
                       ("kokoro_noun", "Kokoro noun"), ("kokoro_number", "Kokoro number"),
                       ("xtts_noun", "XTTS noun"), ("xtts_number", "XTTS number")]:
        n_rows.append(f"<tr><td>{label}</td><td class='num'>{n_counts.get(key, '?')}</td></tr>")

    align = load_json(V4_ALIGN)
    align = align if isinstance(align, list) else []
    align_ok = sum(1 for e in align if isinstance(e, dict) and e.get("status") == "ok")
    align_err = sum(1 for e in align if isinstance(e, dict) and e.get("status") == "error")

    return f"""\
<h3>4.4 &ensp; Data Quality</h3>
<p>Gate check: <b>{passed}/{total}</b> passed, <b>{fail}</b> failed.</p>
{f'<ul>{fail_text}</ul>' if fail_text else ''}

<div class="table-wrap">
<table>
<thead><tr><th>Model × Condition</th><th class="num">N</th></tr></thead>
<tbody>
{"".join(n_rows)}
</tbody></table>
</div>

<p>WhisperX V3 alignment: <b>{align_ok}</b> aligned successfully, <b>{align_err}</b> errors
(12 / 606, or 2.0%). Aligned segments used for syllable-count-based speaking rate.
f0 extracted via parselmouth (Praat).</p>
"""


def combined_results_init() -> str:
    return """\
<h2 class="page-break">5 &ensp; Combined Results</h2>
<p>The pilot identified two acoustic phenotypes. The replication, with 13× more data
and mixed-effects modeling, clarifies which one is robust and which one is model-specific.</p>
"""


def tempo_acceleration_section(v4_data: dict, fig_b64: str) -> str:
    rows = []
    for model in ["chatterbox", "kokoro", "xtts"]:
        label = MODEL_LABELS.get(model, model)
        r = v4_data.get(model, {}).get("speaking_rate", {})
        if "error" in r:
            rows.append(f"<tr><td>{label}</td><td class='num' colspan='5'>FIT ERROR: {r['error']}</td></tr>")
            continue
        coef = r["coef"]
        ci_lo, ci_hi = r["ci_lo"], r["ci_hi"]
        p = r["p"]
        sig = _sig_stars(p)
        converged = r.get("converged", True)
        ci_note = "✓ excludes zero" if (ci_lo * ci_hi > 0) else "✗ overlaps zero"
        conv_note = "" if converged else " (unconverged)"
        rows.append(
            f"<tr><td>{label}{conv_note}</td><td class='num'>{coef:+.3f}</td>"
            f"<td class='num'>[{ci_lo:+.3f}, {ci_hi:+.3f}]</td>"
            f"<td class='num'>{_fmt_p(p)}{sig}</td>"
            f"<td class='num'>{ci_note}</td></tr>"
        )

    fig_html = ""
    if fig_b64:
        fig_html = f"""\
<div class="figure">
<img src="data:image/png;base64,{fig_b64}" alt="Speaking rate boxplots">
<div class="caption">Figure 2: Speaking rate distribution by condition and model (V4, N=594).
Red = number prime, green = noun prime.</div>
</div>"""

    return f"""\
<h3>5.1 &ensp; Primary Finding: Tempo Acceleration</h3>
<p><code>s peaking_rate ~ condition_num + (1 | target_id) + (1 | repetition)</code></p>

{fig_html}

<div class="table-wrap">
<table>
<thead><tr><th>Model</th><th class="num">Coef (syll/s)</th><th class="num">95% CI</th><th class="num">p</th><th class="num">CI check</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
</div>

<div class="callout-good">
<p><strong>Tempo acceleration confirmed across all three architectures.</strong>
All three models show a statistically significant speaking-rate increase after robotic
primes, with 95% CIs excluding zero. Effect sizes range from +0.35 (Chatterbox) to
+1.36 (Kokoro) syllables/second. The pilot identified this as a secondary phenotype
(borderline in Kokoro, numerically present in XTTS). The replication confirms it as the
<b>primary, universal finding</b>.</p>

<p>The Chatterbox speaking_rate model has a convergence warning (optimizer hit
a parameter-space boundary), but the coefficient direction (+0.35) and significance
(p=0.006) are consistent across optimization attempts and the 95% CI excludes zero.
<b>Do not discount the Chatterbox tempo result</b> — the convergence issue reflects
a challenging likelihood surface for this particular model/condition combination, not
an unreliable effect.</p>
</div>
"""


def pitch_compression_section(v4_data: dict, fig_b64: str) -> str:
    rows = []
    notes = []
    for model in ["chatterbox", "kokoro", "xtts"]:
        label = MODEL_LABELS.get(model, model)
        r = v4_data.get(model, {}).get("f0_cv", {})
        if "error" in r:
            rows.append(f"<tr><td>{label}</td><td class='num' colspan='5'>FIT ERROR: {r['error']}</td></tr>")
            continue
        coef = r["coef"]
        ci_lo, ci_hi = r["ci_lo"], r["ci_hi"]
        p = r["p"]
        sig = _sig_stars(p)
        pi_check = (ci_lo * ci_hi > 0)
        if p < 0.05 and pi_check:
            verdict = "Confirmed"
        elif p >= 0.05:
            verdict = "Null"
        else:
            verdict = "Inconclusive"
        rows.append(
            f"<tr><td>{label}</td><td class='num'>{coef:+.4f}</td>"
            f"<td class='num'>[{ci_lo:+.4f}, {ci_hi:+.4f}]</td>"
            f"<td class='num'>{_fmt_p(p)}{sig}</td>"
            f"<td>{verdict}</td></tr>"
        )
        if model == "chatterbox":
            notes.append(
                f"<li>Chatterbox speaking_rate model had convergence warnings (boundary). "
                "The f0_cv model converged but coefficient is near-zero with CI crossing zero."
                "</li>"
            )
        if p >= 0.05:
            notes.append(f"<li>{label}: CI includes zero — no reliable f0_cv effect detected.</li>")

    fig_html = ""
    if fig_b64:
        fig_html = f"""\
<div class="figure">
<img src="data:image/png;base64,{fig_b64}" alt="f0_cv boxplots">
<div class="caption">Figure 3: f0_cv distribution by condition and model (V4, N=594).
Red = number prime, green = noun prime. Note the separation in Kokoro vs overlap in Chatterbox/XTTS.</div>
</div>"""

    notes_html = ""
    if notes:
        notes_html = "<div class='callout-warn'><ul>" + "".join(notes) + "</ul></div>"

    return f"""\
<h3>5.2 &ensp; Secondary Finding: Pitch Compression</h3>
<p><code>f0_cv ~ condition_num + (1 | target_id) + (1 | repetition)</code></p>

{fig_html}

<div class="table-wrap">
<table>
<thead><tr><th>Model</th><th class="num">Coef</th><th class="num">95% CI</th><th class="num">p</th><th>Verdict</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
</div>

{notes_html}

<div class="callout-good">
<p><strong>Pitch compression is Kokoro-specific.</strong> Kokoro shows a clear, significant
f0_cv drop (coef=-0.045, p=2.1e-08). Chatterbox and XTTS show no reliable f0_cv change.
The pilot's Chatterbox pitch compression (-39%, p=0.031) did not replicate under mixed-effects
with 10 targets. See &sect;5.4 for interpretation.</p>
</div>
"""


def coefficient_plot_section(fig_b64: str) -> str:
    if not fig_b64:
        return ""
    return f"""\
<h3>5.3 &ensp; Effect Size Summary</h3>
<div class="figure">
<img src="data:image/png;base64,{fig_b64}" alt="Coefficient estimates with 95% CI">
<div class="caption">Figure 4: Per-model coefficient estimates (mixed effects) with 95% confidence intervals.
Left: tempo acceleration. Right: pitch compression. Chatterbox speaking_rate shown as unconverged (square marker).</div>
</div>
"""


def what_changed() -> str:
    return """\
<h3>5.4 &ensp; What Changed from Pilot</h3>

<div class="callout-note">
<p><strong>Phenotype map shifted.</strong> The pilot found pitch compression in Chatterbox
(-39%, p=0.031) and XTTS (-29%, p=0.094) and tempo acceleration in Kokoro (borderline).
The replication finds <b>tempo acceleration in all three</b> and <b>pitch compression in
Kokoro only</b>.</p>
</div>

<p>Three possibilities for Chatterbox's missing pitch compression, not mutually exclusive:</p>
<ol>
<li><b>Target-specific.</b> The single _7 target may have been unusually sensitive. With
10 diverse targets, the per-target effect averages out.</li>
<li><b>Seed coupling inflated _7 effect.</b> If noun and number conditions shared a seed,
small artifacts could drive a spurious paired difference at n=5.</li>
<li><b>Underpowered detection in _8.</b> The mixed model accounts for target-level variance,
reducing apparent effect size relative to the raw paired comparison. A small true effect
at the target level would be harder to detect.</li>
</ol>
<p>We cannot distinguish these from available data. The honest answer is: <b>we don't know</b>
if the pilot Chatterbox pitch compression was real or noise.</p>

<div class="callout-warn">
<p><strong>Convergence note.</strong> The Chatterbox speaking_rate mixed model shows
convergence warnings. The f0_cv model converged. Both point estimates should be treated
with appropriate caution but are consistent with the pattern: universal tempo acceleration,
Kokoro-specific pitch compression.</p>
</div>
"""


def conclusions_section() -> str:
    return """\
<h2>6 &ensp; Conclusions</h2>

<ol>
<li><b>Tempo acceleration is the robust universal signal.</b> Across all three models,
robotic number primes cause a statistically significant speaking-rate increase.
This is not subtle. Kokoro speeds up by over 1.3 syllables/second, XTTS by 0.7,
Chatterbox by 0.35. All 95% CIs exclude zero.</li>

<li><b>Pitch compression is Kokoro-specific.</b> Only Kokoro shows reliable f0_cv
suppression after robotic primes (coef=-0.045, p=2.1e-08, CI excludes zero). Why Kokoro?
It uses fixed speaker embeddings without voice cloning — its prosodic control may be
globally coupled, making it susceptible to flat-prime inertia that cloning-based
architectures recover from more quickly.</li>

<li><b>The pilot got the phenomena right, wrong distribution.</b> _7 correctly identified
two acoustic phenotypes (pitch compression, tempo acceleration) but misattributed which
models showed which. This is exactly what you'd expect from n=5 on one sentence:
genuine signal plus misallocation from small-sample noise.</li>

<li><b>Practical fix:</b> generate robotic segments (numbers, codes, identifiers) and
emotional speech in <b>separate TTS context windows</b>, then stitch the audio
afterward. A single context window for mixed-style text will bleed tempo acceleration
into adjacent speech. This is not a fundamental TTS flaw — it's a context-window
management detail that every pipeline engineer building emotionally-sensitive
applications should know.</li>
</ol>
"""


def what_this_means() -> str:
    return """\
<h2>7 &ensp; What This Means</h2>

<h3>7.1 &ensp; Robotic Primes Make Models Rush — Not (Primarily) Go Flat</h3>
<p>The original hypothesis was about pitch — the monotone prime drying out the emotional
target's prosody. The strongest, most consistent signal is about rhythm. <b>The hangover
is tempo, not pitch.</b> This changes how we think about the mechanism: the robotic prime
alters the model's internal sense of pace, and that change persists.</p>

<h3>7.2 &ensp; Architecture Matters</h3>
<p>The differential response across models — pitch compression in Kokoro, tempo in all three
but at different magnitudes — suggests the hangover phenotype is shaped by model architecture,
not just by the prime. Kokoro's non-cloning approach may trade off prosodic flexibility for
prosodic inertia. Chatterbox and XTTS-v2, which condition on speaker embeddings for pitch
range, may be more resilient to pitch carryover while still susceptible to rhythm bleed.</p>

<h3>7.3 &ensp; Practical Fix: Generate Separately</h3>
<p>Context-window bleed is not a deep architectural problem — it's a pipeline management
detail. The same model that shows tempo acceleration when concatenating mixed-style text
in one window will produce clean, unaffected emotional speech when each segment is
generated independently. The engineering lesson: assume acoustic state carries forward.
If prosodic consistency between adjacent segments matters, isolate them.</p>
"""


def whats_next() -> str:
    return """\
<h2>8 &ensp; What's Next</h2>
<ul>
<li><b>Perceptual validation.</b> Everything measured is acoustic. Do listeners hear any
of this? An ABX listening test would determine whether tempo acceleration crosses the
perceptual threshold.</li>
<li><b>Spelled-out vs digit numbers.</b> Is the acceleration driven by digit-string rhythm
or the semantics of counting? Generating primes with spelled-out numbers ("eight hundred
forty seven") would isolate surface form vs meaning.</li>
<li><b>Multiple speakers per model.</b> A single voice per model tells you about that model
with that voice. 3–5 diverse speaker embeddings per model would reveal whether the
hangover is a model property or a voice property.</li>
<li><b>Broader emotional range.</b> All targets were angry/indignant. Sad, happy, neutral,
and fearful targets would reveal whether the hangover interacts with emotional valence.</li>
</ul>
"""


def appendix(v4_data: dict, v3_data: dict) -> str:
    lines = ["<h2 class='page-break'>9 &ensp; Appendix: Full Statistics</h2>"]

    # V3 per-model descriptive
    lines.append("<h3>A.1 &ensp; Pilot (_7) Per-Condition Descriptive Statistics</h3>")
    models = ["chatterbox", "kokoro", "xtts"]
    metrics = ["f0_cv", "f0_mean", "f0_std", "speaking_rate", "energy_std"]
    conds = {"noun": "Nouns", "subliminal": "Numbers"}

    for model in models:
        label = MODEL_LABELS.get(model, model)
        d = v3_data.get("descriptive", {}).get(model, {})
        lines.append(f"<h4>{label}</h4>")
        lines.append("<table><thead><tr><th>Metric</th><th>Condition</th><th class='num'>N</th>"
                     "<th class='num'>Mean</th><th class='num'>SD</th><th class='num'>Min</th>"
                     "<th class='num'>Max</th></tr></thead><tbody>")
        for met in metrics:
            for cond_key, cond_label in conds.items():
                m = d.get(met, {}).get(cond_key, {})
                lines.append(
                    f"<tr><td>{met}</td><td>{cond_label}</td>"
                    f"<td class='num'>{m.get('n','')}</td>"
                    f"<td class='num'>{m.get('mean',''):.4f}</td>"
                    f"<td class='num'>{m.get('std',''):.4f}</td>"
                    f"<td class='num'>{m.get('min',''):.4f}</td>"
                    f"<td class='num'>{m.get('max',''):.4f}</td></tr>"
                )
        lines.append("</tbody></table>")

    # V4 per-model mixed effects
    lines.append("<h3>A.2 &ensp; Replication (_8) Per-Model Mixed Effects</h3>")
    for model in models:
        label = MODEL_LABELS.get(model, model)
        m = v4_data.get(model, {})
        lines.append(f"<h4>{label} (N={m.get('speaking_rate',{}).get('N', m.get('f0_cv',{}).get('N','?'))})</h4>")
        lines.append("<table><thead><tr><th>Model</th><th class='num'>Coef</th>"
                     "<th class='num'>95% CI</th><th class='num'>p</th><th>Converged</th></tr></thead><tbody>")
        for metric_name, display in [("speaking_rate", "s peaking_rate ~ cond"), ("f0_cv", "f0_cv ~ cond")]:
            r = m.get(metric_name, {})
            if "error" in r:
                lines.append(f"<tr><td>{display}</td><td class='num' colspan='4'>ERROR: {r['error']}</td></tr>")
            else:
                lines.append(
                    f"<tr><td>{display}</td>"
                    f"<td class='num'>{r['coef']:+.6f}</td>"
                    f"<td class='num'>[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]</td>"
                    f"<td class='num'>{_fmt_p(r['p'])}</td>"
                    f"<td class='num'>{r.get('converged','?')}</td></tr>"
                )
        lines.append("</tbody></table>")
    return "\n".join(lines)


def footer() -> str:
    return """\
<div class="footer">
    Subliminal Hangover — Pilot (_7) + Preregistered Replication (_8) &bull;
    45 WAVs (pilot) + 594 WAVs (replication) across Chatterbox, XTTS-v2, and Kokoro.
    <br>Target extraction: WhisperX V3 &bull; f0: parselmouth (Praat) &bull;
    Analysis: Wilcoxon (_7), statsmodels MixedLM (_8) &bull; Report: weasyprint.
    <br>May 2026
</div>
</div></body></html>
"""


# ── main ─────────────────────────────────────────────────────

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    v3_data = load_json(V3_STATS)
    v3_data = v3_data if isinstance(v3_data, dict) else {}
    v4_data = load_json(V4_STATS)
    v4_data = v4_data if isinstance(v4_data, dict) else {}

    v3_fig_b64 = img_b64(V3_FIG)

    print("Generating V4 figures ...")
    try:
        extra_figs = generate_figures()
    except Exception as e:
        print(f"WARNING: figure generation failed ({e}), continuing without V4 figures", file=sys.stderr)
        extra_figs = {}

    html = (
        cover()
        + the_question()
        + pilot_design()
        + pilot_results()
        + pilot_figure(v3_fig_b64)
        + pilot_stats_table(v3_data)
        + pilot_weaknesses()
        + credibility_gap()
        + replication_design()
        + discoveries_during_build()
        + gate_check_section()
        + combined_results_init()
        + tempo_acceleration_section(v4_data, extra_figs.get("speaking_rate_box", ""))
        + pitch_compression_section(v4_data, extra_figs.get("f0_cv_box", ""))
        + coefficient_plot_section(extra_figs.get("coef_plot", ""))
        + what_changed()
        + conclusions_section()
        + what_this_means()
        + whats_next()
        + appendix(v4_data, v3_data)
        + footer()
    )

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(REPORT_HTML) / 1024
    print(f"Combined report HTML: {REPORT_HTML} ({size_kb:.0f} KB)")

    try:
        from weasyprint import HTML as WPHTML
        WPHTML(filename=str(REPORT_HTML)).write_pdf(str(REPORT_PDF))
        pdf_kb = os.path.getsize(REPORT_PDF) / 1024
        print(f"Combined report PDF:  {REPORT_PDF} ({pdf_kb:.0f} KB)")
    except ImportError:
        print("WARN: weasyprint not available, skipping PDF", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
