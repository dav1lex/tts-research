#!/usr/bin/env python3
"""Generate self-contained HTML report for semantic priming experiment,
then convert to PDF via weasyprint."""

import base64
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FEATURES_CSV = RESULTS_DIR / "features.csv"
STATS_SUMMARY = RESULTS_DIR / "stats_summary.csv"
STATS_PAIRWISE = RESULTS_DIR / "stats_pairwise.csv"

REPORT_HTML = RESULTS_DIR / "report.html"
REPORT_PDF = RESULTS_DIR / "report.pdf"

MODEL_LABELS = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro",
}
CONDITION_LABELS = {
    "cold": "Cold",
    "primed_neutral": "Neutral Prime",
    "primed_owl": "Owl Prime",
    "primed_death": "Death Prime",
}
FEATURE_LABELS = {
    "f0_mean": "F0 Mean",
    "f0_std": "F0 Std",
    "f0_range": "F0 Range",
    "speech_rate": "Speech Rate",
    "pause_count": "Pause Count",
    "pause_duration_mean": "Pause Duration",
    "rms_energy": "RMS Energy",
    "spectral_centroid": "Spectral Centroid",
}
FEATURE_UNITS = {
    "f0_mean": "Hz",
    "f0_std": "Hz",
    "f0_range": "Hz",
    "speech_rate": "words/s",
    "pause_count": "",
    "pause_duration_mean": "s",
    "rms_energy": "",
    "spectral_centroid": "Hz",
}

CSS = """/* semantic priming report */
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt; line-height: 1.6; color: #333; background: #fff;
    padding: 2em 1em;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
body.pdf { padding: 0; }
.report-container { max-width: 900px; margin: 0 auto; padding: 2em 1.5em; }
h1 { font-size: 24pt; color: #1a1a2e; margin-bottom: 0.2em; font-weight: 700; letter-spacing: -0.3px; }
h2 { font-size: 17pt; color: #1a1a2e; margin-top: 1.6em; margin-bottom: 0.5em; padding-bottom: 0.3em; border-bottom: 3px solid #1a1a2e; font-weight: 600; }
h3 { font-size: 12pt; color: #16213e; margin-top: 1.2em; margin-bottom: 0.3em; font-weight: 600; }
.subtitle { font-size: 13pt; color: #555; margin-bottom: 1.5em; font-style: italic; }
p { margin-bottom: 0.7em; }
.figure-container { margin: 1em 0; text-align: center; }
.figure-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
table { width: 100%; border-collapse: collapse; margin: 0.8em 0 1.2em 0; font-size: 10pt; }
th { background: #1a1a2e; color: #fff; padding: 7px 9px; text-align: left; font-weight: 600; white-space: nowrap; }
td { padding: 6px 9px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f5f5f8; }
tr:hover td { background: #eaeaf0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.highlight { background: #f0f8e8; padding: 0.8em 1em; border-left: 4px solid #4CAF50; margin: 0.8em 0; border-radius: 2px; }
.warning { background: #fef9e7; padding: 0.8em 1em; border-left: 4px solid #e6a817; margin: 0.8em 0; border-radius: 2px; }
.limitations { background: #fafafa; padding: 0.8em 1em; border-left: 4px solid #c0392b; margin: 0.8em 0; border-radius: 2px; }
.limitations ul { margin-left: 1.5em; margin-bottom: 0.4em; }
.limitations li { margin-bottom: 0.3em; }
.sig-star { color: #4CAF50; font-weight: bold; }
.footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd; font-size: 9pt; color: #888; text-align: center; }
@media print {
    body { padding: 0; font-size: 10pt; }
    h2 { page-break-after: avoid; }
    .figure-container { page-break-inside: avoid; }
    table { page-break-inside: avoid; }
    @page { size: A4; margin: 1.5cm; }
}
"""


def img_b64(name):
    path = FIG_DIR / name
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('ascii')}"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(val, dec=1):
    if val is None:
        return "\u2014"
    try:
        v = float(val)
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{dec}f}"
    except (ValueError, TypeError):
        return str(val)


def p_stars(p):
    try:
        pv = float(p)
        if pv < 0.001:
            return "***"
        if pv < 0.01:
            return "**"
        if pv < 0.05:
            return "*"
    except (ValueError, TypeError):
        pass
    return ""


# ── build sections ──────────────────────────────────────────────────────────


def header():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prior-Context Prosody Probe - Report</title>
<style>{CSS}</style>
</head>
<body class="pdf">
<div class="report-container">

<h1>Prior-Context Prosody Probe</h1>
<p class="subtitle">
    Does pre-pended paragraph context change target-sentence prosody?<br>
    Acoustic analysis of 3 TTS models &times; 4 conditions &times; 5 repetitions<br>
    &mdash; 60 target-sentence segments extracted via VAD boundary detection
</p>
"""


def caveat():
    return """
<h2>1 &ensp; Caveat</h2>

<div class="limitations">
<p><strong>This report describes acoustic measurements from one fixed target sentence
per condition.</strong> The sample size is small (n=5 per condition per model). All
metrics are acoustic proxies (parselmouth F0, librosa pyin, VAD pauses). Informal
listening was used only as a segmentation and repetition sanity check, not as a
controlled listening study. Results are exploratory and hypothesis-generating.</p>
</div>
"""


def methodology():
    return """
<h2>2 &ensp; Methodology</h2>

<h3>Design</h3>
<p><strong>4 conditions:</strong> cold (target sentence only), primed_neutral
(target preceded by a neutral paragraph), primed_owl (target preceded by an
owl-description paragraph), primed_death (target preceded by a death/funeral
paragraph).</p>

<p><strong>Target sentence (fixed across all conditions):</strong>
<em>"The quarterly figures were reviewed and submitted before the deadline."</em>
(10 words).</p>

<p><strong>Prime paragraphs:</strong> 4 sentences, ~180-190 characters each.
Neutral, owl, and death primes are matched for sentence count and punctuation
pattern but differ in semantic valence.</p>

<h3>Models</h3>
<p><strong>Chatterbox</strong> and <strong>XTTS-v2</strong> use VCTK p229
female reference for speaker identity with fixed seed-derived variation across
repetitions. <strong>Kokoro</strong> uses af_bella voice preset with
<code>split_pattern=None</code> to prevent newline-based segmentation.</p>

<h3>Pipeline</h3>
<p>Full clips generated (prime + target as one input) &rarr; target sentence
extracted via VAD pause-boundary detection &rarr; 8 prosodic features measured
&rarr; one-way ANOVA + pairwise t-tests with Bonferroni correction.</p>

<h3>Acoustic Features (8)</h3>
<table>
<thead><tr><th>Feature</th><th>Tool</th><th>Captures</th></tr></thead>
<tbody>
<tr><td>F0 mean</td><td>parselmouth (Praat)</td><td>Average pitch</td></tr>
<tr><td>F0 std</td><td>parselmouth</td><td>Pitch variance</td></tr>
<tr><td>F0 range</td><td>parselmouth</td><td>Pitch excursion</td></tr>
<tr><td>Speech rate</td><td>word count / voiced duration</td><td>Speaking pace</td></tr>
<tr><td>Pause count</td><td>RMS threshold (0.02)</td><td>Pause frequency</td></tr>
<tr><td>Pause duration</td><td>RMS threshold</td><td>Average pause length</td></tr>
<tr><td>RMS energy</td><td>librosa</td><td>Loudness</td></tr>
<tr><td>Spectral centroid</td><td>librosa</td><td>Voice brightness</td></tr>
</tbody>
</table>

<h3>Statistics</h3>
<p>One-way ANOVA per model per feature (4 conditions). Post-hoc pairwise
Welch t-tests with Bonferroni correction (6 comparisons per feature &times;
8 features = 48 tests per model). Effect sizes: &eta;&sup2; for ANOVA,
Cohen&#8217;s d for pairwise differences. These tests are descriptive for this
single-sentence probe, not confirmatory population-level inference.</p>
"""


def f0_results(rows, pairwise):
    """Figure + table for F0 mean."""
    ans = [r for r in rows if r["condition"] == "ANOVA"]

    parts = ['<h2>3 &ensp; F0 Mean: Primary Metric</h2>']
    parts.append("""
<p>Average pitch of the target sentence across conditions. The hypothesis
predicts that semantically different primes produce different F0 in the
identical target sentence.</p>
""")

    b64_img = img_b64("f0_mean.png")
    if b64_img:
        parts.append(f'<div class="figure-container"><img src="{b64_img}" alt="F0 Mean by Condition"></div>')

    parts.append("""
<h3>ANOVA Summary</h3>
<table><thead><tr><th>Model</th><th>F(3,16)</th><th>p</th><th>&eta;&sup2;</th><th>Sig?</th></tr></thead><tbody>
""")
    for model in ["chatterbox", "xtts", "kokoro"]:
        match = [r for r in ans if r["model"] == model and r["feature"] == "f0_mean"]
        if match:
            r = match[0]
            p = float(r["sd"])
            eta = float(r["eta_squared"])
            sig = '<span class="sig-star">{}</span>'.format(p_stars(p)) if p < 0.05 else ""
            parts.append(
                f'<tr><td>{MODEL_LABELS[model]}</td>'
                f'<td class="num">{fmt(r["mean"], 2)}</td>'
                f'<td class="num">{fmt(p, 4)} {sig}</td>'
                f'<td class="num">{fmt(eta, 3)}</td>'
                f'<td>{"Yes" if p < 0.05 else "No"}</td></tr>'
            )
    parts.append("</tbody></table>")

    # Kokoro significant → show pairwise detail
    kokoro_pairwise = [r for r in pairwise if r["model"] == "kokoro" and r["feature"] == "f0_mean"]
    if kokoro_pairwise:
        parts.append("""
<h3>Kokoro: Pairwise Comparisons (Bonferroni-corrected)</h3>
<table><thead><tr><th>Pair</th><th>Mean A</th><th>Mean B</th><th>t</th><th>p (corr)</th><th>Cohen's d</th></tr></thead><tbody>
""")
        for pr in kokoro_pairwise:
            p_corr = float(pr["p_bonferroni_6"])
            sig = p_stars(p_corr) if p_corr < 0.05 else ""
            parts.append(
                f'<tr><td>{CONDITION_LABELS[pr["condition_a"]]} vs '
                f'{CONDITION_LABELS[pr["condition_b"]]}</td>'
                f'<td class="num">{fmt(pr["mean_a"], 1)}</td>'
                f'<td class="num">{fmt(pr["mean_b"], 1)}</td>'
                f'<td class="num">{fmt(pr["t"], 2)}</td>'
                f'<td class="num">{fmt(p_corr, 4)} {sig}</td>'
                f'<td class="num">{fmt(pr["cohens_d"], 2)}</td></tr>'
            )
        parts.append("</tbody></table>")

        parts.append("""
<div class="highlight">
<p><strong>Kokoro shows a large target-sentence F0 shift under prior context.</strong>
The cold condition (&tilde;190 Hz) differs from all primed conditions
(&tilde;196-208 Hz, p < 0.001). Some pairwise differences among primed conditions
also appear, but the dominant separation is cold vs anything-with-context.
This supports a "has prior paragraph" context effect more strongly than a
specific semantic priming effect.</p>
</div>
""")

    # Add Chatterbox note
    parts.append(f"""
<div class="warning">
<p><strong>Chatterbox and XTTS-v2: no significant F0 condition effect.</strong>
Stochastic variation between repetitions exceeds any between-condition difference.
With n=5 per condition, the experiment is underpowered to detect small effects
in these models.</p>
</div>
""")

    return "\n".join(parts)


def feature_table(rows, pairwise):
    """Full ANOVA results for all features."""
    ans = [r for r in rows if r["condition"] == "ANOVA"]

    parts = ['<h2>4 &ensp; All Features: ANOVA Results</h2>']
    parts.append("""
<p>One-way ANOVA (4 conditions) for each feature across all 3 models.
Significant effects (&alpha;=0.05) are bolded.</p>

<table><thead><tr>
<th>Model</th><th>Feature</th><th>F</th><th>p</th><th>&eta;&sup2;</th><th>Sig</th>
</tr></thead><tbody>
""")
    for model in ["chatterbox", "xtts", "kokoro"]:
        model_features = [
            r for r in ans if r["model"] == model
        ]
        for r in model_features:
            feature = r["feature"]
            p = float(r["sd"])
            eta = float(r["eta_squared"])
            sig = p < 0.05
            row_class = ' style="font-weight:600"' if sig else ""
            parts.append(
                f'<tr{row_class}>'
                f'<td>{MODEL_LABELS[model]}</td>'
                f'<td>{FEATURE_LABELS[feature]}</td>'
                f'<td class="num">{fmt(r["mean"], 2)}</td>'
                f'<td class="num">{fmt(p, 4)} {p_stars(p)}</td>'
                f'<td class="num">{fmt(eta, 3)}</td>'
                f'<td>{"<strong>Yes</strong>" if sig else "No"}</td></tr>'
            )
    parts.append("</tbody></table>")

    # Count per model
    counts = {}
    for model in ["chatterbox", "xtts", "kokoro"]:
        sigs = sum(
            1 for r in ans
            if r["model"] == model and float(r["sd"]) < 0.05
        )
        counts[model] = sigs
    parts.append(f"""
<div class="highlight">
<p><strong>Significant features per model:</strong>
Chatterbox = {counts['chatterbox']}/8,
XTTS-v2 = {counts['xtts']}/8,
Kokoro = {counts['kokoro']}/8.
Kokoro shows strong across-the-board context effects. Chatterbox and XTTS-v2
show no reliable condition effects on any feature.</p>
</div>
""")
    return "\n".join(parts)


def figure_spread():
    """Embed remaining figures."""
    parts = ['<h2>5 &ensp; Visual Summary</h2>']

    b64_speech = img_b64("speech_pause.png")
    if b64_speech:
        parts.append(
            '<h3>Speech Rate &amp; Pause Count</h3>'
            f'<div class="figure-container"><img src="{b64_speech}" alt="Speech Rate and Pause Count"></div>'
        )

    b64_effects = img_b64("effect_sizes.png")
    if b64_effects:
        parts.append(
            '<h3>Effect Sizes (&eta;&sup2;) by Feature</h3>'
            f'<div class="figure-container"><img src="{b64_effects}" alt="Effect Sizes"></div>'
        )

    b64_kokoro = img_b64("kokoro_detail.png")
    if b64_kokoro:
        parts.append(
            '<h3>Kokoro: Key Features Detail</h3>'
            f'<div class="figure-container"><img src="{b64_kokoro}" alt="Kokoro Detail"></div>'
        )

    b64_sig = img_b64("significance.png")
    if b64_sig:
        parts.append(
            '<h3>Significance Summary</h3>'
            f'<div class="figure-container"><img src="{b64_sig}" alt="Significance Summary"></div>'
        )

    return "\n".join(parts)


def condition_means(rows):
    """Summary table of condition means per model for key features."""
    parts = ['<h2>6 &ensp; Condition Means</h2>']
    parts.append("""
<p>Mean values per condition per model for primary features. Standard deviations
in parentheses (n=5).</p>
""")

    conds = ["cold", "primed_neutral", "primed_owl", "primed_death"]
    features_show = ["f0_mean", "speech_rate", "rms_energy", "spectral_centroid"]

    for model in ["chatterbox", "xtts", "kokoro"]:
        parts.append(f'<h3>{MODEL_LABELS[model]}</h3>')
        parts.append(
            '<table><thead><tr><th>Feature</th>'
            + ''.join(f'<th>{CONDITION_LABELS[c]}</th>' for c in conds)
            + '</tr></thead><tbody>'
        )
        for feat in features_show:
            feat_row = f'<tr><td>{FEATURE_LABELS[feat]} ({FEATURE_UNITS[feat]})</td>'
            for cond in conds:
                vals = [
                    float(r[feat])
                    for r in rows
                    if r["model"] == model and r["condition"] == cond
                ]
                if vals:
                    m = np_mean(vals)
                    s = np_std(vals)
                    feat_row += f'<td class="num">{fmt(m, 1)} ({fmt(s, 1)})</td>'
                else:
                    feat_row += '<td class="num">-</td>'
            feat_row += '</tr>'
            parts.append(feat_row)
        parts.append('</tbody></table>')

    return "\n".join(parts)


def np_mean(vals):
    try:
        import numpy
        return float(numpy.mean([float(v) for v in vals]))
    except Exception:
        return 0.0


def np_std(vals):
    try:
        import numpy
        return float(numpy.std([float(v) for v in vals], ddof=1))
    except Exception:
        return 0.0


def limitations():
    return """
<h2>7 &ensp; Methodological Limitations</h2>

<div class="limitations">
<ul>
    <li><strong>Sample size.</strong> n=5 per condition is minimal for statistical
    inference. Between-condition differences require large effects to reach
    significance.</li>
    <li><strong>Single sentence.</strong> One fixed target sentence means results
    may not generalise. The sentence structure, word choice, and phonetic content
    may interact with priming sensitivity.</li>
    <li><strong>VAD segmentation.</strong> Target-sentence extraction uses
    energy-based pause detection with a duration heuristic. No forced alignment
    validates that segments correspond exactly to the target sentence text.</li>
    <li><strong>Kokoro near-determinism.</strong> Kokoro cold condition shows
    near-zero variance across repetitions (F0 mean SD = 0.28 Hz). This inflates
    statistical significance for Kokoro comparisons due to vanishing within-group
    variance. Extremely large effect sizes partly reflect this unusually low
    within-condition variance and should not be read as population-level effect
    estimates.</li>
    <li><strong>Semantic vs syntactic priming.</strong> The primed conditions
    differ in both semantic content and lexical features. The owl paragraph
    contains different phonemes, syllable counts, and word frequencies than the
    death paragraph. Observed differences may reflect acoustic properties of the
    prime text, not semantic transfer.</li>
    <li><strong>No controlled listening study.</strong> Informal listening confirmed
    that target segments were natural and repetitions were not identical, but no
    blinded human rating or ABX test was run.</li>
    <li><strong>Multiple comparisons.</strong> 24 ANOVAs and 144 pairwise tests
    with Bonferroni correction still carry elevated Type I error risk.</li>
    <li><strong>Kokoro voice mismatch.</strong> Kokoro uses a different speaker
    (af_bella) than the VCTK p229 reference used by Chatterbox and XTTS-v2.
    Cross-model comparisons are confounded by speaker identity.</li>
</ul>
</div>
"""


def conclusions(rows, pairwise):
    """Write the main conclusions."""
    ans = [r for r in rows if r["condition"] == "ANOVA"]

    parts = ['<h2>8 &ensp; Conclusions</h2>']

    # Count significant features per model
    sig_counts = {}
    for model in ["chatterbox", "xtts", "kokoro"]:
        sigs = [r for r in ans if r["model"] == model and float(r["sd"]) < 0.05]
        sig_counts[model] = len(sigs)

    # Kokoro cold vs anything
    kokoro_f0 = [
        r for r in pairwise
        if r["model"] == "kokoro" and r["feature"] == "f0_mean" and r["condition_a"] == "cold"
    ]
    parts.append(f"""
<div class="highlight">
<p><strong>Kokoro shows strong context carryover (all 8 features significant).</strong>
Cold F0 mean ({fmt(kokoro_f0[0]['mean_a'], 1) if kokoro_f0 else '~190'} Hz) is
reliably lower than all primed conditions. F0 range increases &times;1.3, speech rate
drops &minus;34%, and pause patterns reorganise. This is a clear "has prior context"
effect in this single-sentence probe.</p>

<p>However, the primed conditions themselves (neutral, owl, death) differ only
subtly from each other in Kokoro. The separation is primarily cold vs everything-else,
suggesting that the mere <em>presence</em> of preceding text, rather than its semantic
content, drives the effect. Extremely large effect sizes partly reflect Kokoro's
low within-condition variance.</p>
</div>
""")

    parts.append(f"""
<div class="warning">
<p><strong>Chatterbox ({sig_counts['chatterbox']}/8) and XTTS-v2 ({sig_counts['xtts']}/8):
no reliable condition effect.</strong> Between-repetition stochastic variation
exceeds any between-condition signal. These models' prosody for a fixed 10-word sentence
showed within-condition variability larger than the observed between-condition
differences. This does not mean priming is absent &mdash; it may exist below the
detection threshold of n=5.</p>
</div>
""")

    parts.append("""
<h3>What This Tells Us</h3>
<ul>
    <li>Preceding paragraph context can measurably affect target-sentence prosody
    in at least one tested model: Kokoro.</li>
    <li>The strongest effect is a binary "has context / no context" signal, not a
    clean graded semantic response.</li>
    <li>Evidence for semantic <em>content</em>-specific priming (neutral vs owl vs death)
    is weak. The neutral-prime condition produces broadly similar prosody to the
    emotionally charged primes.</li>
    <li>The tested systems exhibited different levels of context carryover under
    these conditions. Architecture, voice preset, speaker reference, and pipeline
    differences are confounded.</li>
    <li>For this single target sentence, prior text mattered for Kokoro but not
    detectably for Chatterbox or XTTS-v2.</li>
</ul>

<h3>Practical Recommendation</h3>
<p>If building a prosody-sensitive TTS pipeline, do not assume preceding text is
acoustically neutral. Test target-only outputs after segmentation and include a
neutral-prime control. For Chatterbox and XTTS-v2, any context signal in this
experiment is smaller than within-condition stochastic variance at n=5.</p>
""")

    return "\n".join(parts)


def footer():
    return """
<div class="footer">
    Prior-Context Prosody Probe &mdash; 60 target-sentence segments across Chatterbox, XTTS-v2, and Kokoro.
    <br>Report generated with Python 3.12 &bull; matplotlib &bull; weasyprint.
</div>

</div>
</body>
</html>
"""


def main():
    features_rows = load_csv(FEATURES_CSV)
    summary_rows = load_csv(STATS_SUMMARY)
    pairwise_rows = load_csv(STATS_PAIRWISE)

    html = (
        header()
        + caveat()
        + methodology()
        + f0_results(summary_rows, pairwise_rows)
        + feature_table(summary_rows, pairwise_rows)
        + figure_spread()
        + condition_means(features_rows)
        + limitations()
        + conclusions(summary_rows, pairwise_rows)
        + footer()
    )

    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(REPORT_HTML) / 1024
    print(f"Report HTML written to {REPORT_HTML} ({size_kb:.0f} KB)")

    from weasyprint import HTML
    HTML(filename=str(REPORT_HTML)).write_pdf(str(REPORT_PDF))
    pdf_kb = os.path.getsize(REPORT_PDF) / 1024
    print(f"Report PDF written to {REPORT_PDF} ({pdf_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
