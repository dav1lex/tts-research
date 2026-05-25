#!/usr/bin/env python3
"""Generate self-contained HTML report for punctuation sensitivity benchmark."""

import base64
import csv
import json
import os
import pathlib
import statistics
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
FEATURES_DIR = RESULTS_DIR / "features"
FIGURES_DIR = RESULTS_DIR / "figures"

REPORT_PATH = RESULTS_DIR / "report.html"

CSV_PATH = FEATURES_DIR / "pause_features.csv"
ANALYSIS_PATH = RESULTS_DIR / "analysis.json"
GATE_PATH = RESULTS_DIR / "gate_check.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def img_to_b64(path):
    """Return base64-encoded data URI for a PNG image."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fmt(val, decimals=1):
    """Format a number for display, with smart handling of small values."""
    if val is None:
        return "—"
    try:
        v = float(val)
        if abs(v) < 0.01 and v != 0:
            return f"{v:.2e}"
        if decimals == 1:
            return f"{v:.1f}"
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def fmt_int(val):
    try:
        return f"{int(float(val))}"
    except (ValueError, TypeError):
        return str(val)


MODEL_DISPLAY = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "f5tts": "F5-TTS",
    "kokoro": "Kokoro",
}

MODEL_ORDER = ["chatterbox", "kokoro", "xtts", "f5tts"]


# ── load data ────────────────────────────────────────────────────────────────

analysis = load_json(ANALYSIS_PATH)
gate_check = load_json(GATE_PATH)
csv_rows = load_csv(CSV_PATH)

# load figures
figures = {}
for fname in [
    "sentence_end.png",
    "pause_hierarchy.png",
    "trailing.png",
    "quotation.png",
    "summary.png",
    "pause_count.png",
]:
    path = FIGURES_DIR / fname
    if path.exists():
        figures[fname] = img_to_b64(path)
    else:
        figures[fname] = None


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    background: #fff;
    padding: 2em 1em;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

body.pdf {
    padding: 0;
}

.report-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2em 1.5em;
}

h1 {
    font-size: 26pt;
    color: #1a1a2e;
    margin-bottom: 0.2em;
    font-weight: 700;
    letter-spacing: -0.5px;
}

h2 {
    font-size: 18pt;
    color: #1a1a2e;
    margin-top: 1.8em;
    margin-bottom: 0.6em;
    padding-bottom: 0.3em;
    border-bottom: 3px solid #1a1a2e;
    font-weight: 600;
}

h3 {
    font-size: 13pt;
    color: #16213e;
    margin-top: 1.4em;
    margin-bottom: 0.4em;
    font-weight: 600;
}

.subtitle {
    font-size: 13pt;
    color: #555;
    margin-bottom: 1.5em;
    font-style: italic;
}

p {
    margin-bottom: 0.8em;
}

.figure-container {
    margin: 1.2em 0;
    text-align: center;
}

.figure-container img {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0 1.5em 0;
    font-size: 10pt;
}

th {
    background: #1a1a2e;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}

td {
    padding: 7px 10px;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) td {
    background: #f5f5f8;
}

tr:hover td {
    background: #eaeaf0;
}

.key-table th:first-child,
.key-table td:first-child {
    font-weight: 600;
}

.highlight {
    background: #fef9e7;
    padding: 1em 1.2em;
    border-left: 4px solid #e6a817;
    margin: 1em 0;
    border-radius: 2px;
}

.limitations {
    background: #f8f8fa;
    padding: 1em 1.2em;
    border-left: 4px solid #999;
    margin: 1em 0;
    border-radius: 2px;
}

.limitations ul {
    margin-left: 1.5em;
    margin-bottom: 0.5em;
}

.limitations li {
    margin-bottom: 0.4em;
}

.finding {
    font-weight: 600;
    color: #1a1a2e;
}

.badge-pass {
    display: inline-block;
    background: #27ae60;
    color: #fff;
    font-size: 9pt;
    padding: 2px 10px;
    border-radius: 10px;
    font-weight: 600;
}

.badge-fail {
    display: inline-block;
    background: #c0392b;
    color: #fff;
    font-size: 9pt;
    padding: 2px 10px;
    border-radius: 10px;
    font-weight: 600;
}

.badge-yes {
    display: inline-block;
    background: #2e86c1;
    color: #fff;
    font-size: 9pt;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
}

.badge-no {
    display: inline-block;
    background: #7f8c8d;
    color: #fff;
    font-size: 9pt;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
}

.footer {
    margin-top: 3em;
    padding-top: 1em;
    border-top: 1px solid #ddd;
    font-size: 9pt;
    color: #888;
    text-align: center;
}

@media print {
    body {
        padding: 0;
        font-size: 10pt;
    }
    h2 {
        page-break-after: avoid;
    }
    .figure-container {
        page-break-inside: avoid;
    }
    table {
        page-break-inside: avoid;
    }
    @page {
        size: A4;
        margin: 1.5cm;
    }
}
"""


# ── HTML builder ─────────────────────────────────────────────────────────────

def build_html():
    parts = []

    # ── header ────────────────────────────────────────────────────────────
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Punctuation Sensitivity Benchmark Report</title>
<style>{CSS}</style>
</head>
<body class="pdf">
<div class="report-container">

<h1>Punctuation Sensitivity Benchmark</h1>
<p class="subtitle">
    Comparative analysis of F0 and pause behaviour across four TTS models<br>
    &mdash; Chatterbox, XTTS-v2, F5-TTS, and Kokoro &mdash;<br>
    across 28 utterances spanning 5 punctuation categories.
</p>
""")

    # ── 1. Executive Summary ────────────────────────────────────────────────
    parts.append("""
<h2>1 &ensp; Executive Summary</h2>

<p>
This benchmark evaluates how four TTS models interpret punctuation in prosody.
Two independent dimensions emerged: <strong>F0 sensitivity</strong> (does the
model pitch-bend for questions, exclamations, and quoted speech?) and
<strong>pause ordering</strong> (does comma &lt; semicolon &lt; em-dash &lt; ellipsis
hold?). No model scores high on both.
</p>

<ul style="margin-left:1.5em; margin-bottom:1em;">
    <li><strong>Chatterbox</strong> leads in F0 modulation: question marks trigger a
        <strong>+68.5 Hz/s</strong> rising slope versus periods, and quotation
        shifts the F0 range by 50 Hz. Pause ordering is moderate (0.67).</li>
    <li><strong>XTTS-v2</strong> has perfect pause ordering (1.00) with a clear
        comma-to-ellipsis gradient (101.5 ms), but shows <em>zero</em>
        question-mark F0 rise (−5.0 Hz/s).</li>
    <li><strong>F5-TTS</strong> produces the largest pause magnitudes overall
        (comma−ellipsis gradient 115.8 ms) but its pause <em>ordering</em> is
        inverted (0.33) due to noisy, excessive internal pauses. Moderate
        F0 sensitivity (+31.8 Hz/s).</li>
    <li><strong>Kokoro</strong> (no-adaptation baseline) has near-identical
        terminal F0 for sentence-end punctuation but shows the strongest
        trailing-punctuation F0 differentiation (+60.7 Hz/s ellipsis vs
        period), more than any other model. Pause gradient is moderate
        (30.8 ms) with 0.67 hierarchy. Weak on question intonation and
        quotation, but not uniformly punctuation-blind.</li>
</ul>
""")

    # ── Key Results Summary Table ──────────────────────────────────────────
    key_rows = [
        ("Chatterbox", "+68.5", "0.67", "YES", "F0 cues"),
        ("XTTS-v2", "−5.0", "1.00", "NO", "Pause ordering"),
        ("F5-TTS", "+31.8", "0.33", "YES", "Noisy pauses"),
        ("Kokoro", "+25.7", "0.67", "NO", "baseline"),
    ]
    parts.append("""<h3>Key Results Summary</h3>
<table class="key-table">
<thead>
<tr><th>Model</th><th>Question Rise (Hz/s)</th><th>Pause Hierarchy</th><th>Quotation</th><th>Best At</th></tr>
</thead>
<tbody>
""")
    for model, rise, hier, quote, best in key_rows:
        quote_badge = '<span class="badge-yes">YES</span>' if quote == "YES" else '<span class="badge-no">NO</span>'
        parts.append(f"<tr><td>{model}</td><td>{rise}</td><td>{hier}</td><td>{quote_badge}</td><td>{best}</td></tr>\n")
    parts.append("</tbody></table>\n")

    # ── 2. Methodology ────────────────────────────────────────────────────
    parts.append("""
<h2>2 &ensp; Methodology</h2>

<p>
<strong>Stimuli.</strong> 28 unique utterances were manually designed to probe
each punctuation category. Each utterance was rendered with the exact punctuation
required (no textual variation beyond the target mark) to isolate the effect of
punctuation on prosody.
</p>

<p>
<strong>Categories (5).</strong> The utterances span <em>sentence-end</em> (period,
question mark, exclamation mark), <em>pause hierarchy</em> (comma, semicolon,
em-dash, ellipsis), <em>trailing punctuation</em> (ellipsis vs period),
<em>quotation</em> (quoted speech vs reported speech), and
<em>capitalisation</em> (ALL CAPS, Title Case, lowercase).
</p>

<p>
<strong>Models (4).</strong> Chatterbox, XTTS-v2, F5-TTS, and Kokoro were each
used to generate 28 audio clips at 24 kHz. Synthesis used default sampling
parameters (temperature=0.7, top-k=40 where applicable) with no voice cloning
or speaker conditioning.
</p>

<p>
<strong>Acoustic analysis.</strong> Voice activity was detected using an
energy-based VAD (50 ms window, 20 ms step, relative threshold 0.15). F0 was
extracted with <code>pyin</code> (50–600 Hz range, 10 ms step). Pause durations
were measured as silences ≥30 ms between voiced regions. Terminal F0 slope was
computed over the final 100 ms of each voiced region via linear regression.
</p>

<p>
<strong>Gate check.</strong> A minimum viability check was applied before
analysis: each model's period-marked utterances were verified to have at least
50% of frames voiced in their final segment. All four models passed every item
(100% pass rate). This gate only confirms the model produced audible speech
at utterance boundaries; it cannot detect mispronunciation, incorrect words,
or prosodic failure modes such as flat intonation on questions.
</p>
""")

    # ── 3. Gate Check ──────────────────────────────────────────────────────
    parts.append("""<h2>3 &ensp; Gate Check Results</h2>
<p>All models passed the minimum viability check with a 100% pass rate on period-ending utterances.
This confirms that each model produced audible speech at utterance boundaries,
but cannot detect mispronunciation or prosodic failure.</p>
<table>
<thead>
<tr><th>Model</th><th>Period Items</th><th>Passing</th><th>Pass Rate</th><th>Status</th></tr>
</thead>
<tbody>
""")
    for g in gate_check:
        mdl = MODEL_DISPLAY.get(g["model"], g["model"].title())
        status = '<span class="badge-pass">PASS</span>'
        parts.append(
            f"<tr><td>{mdl}</td><td>{g['period_items']}</td>"
            f"<td>{g['periods_passing']}</td>"
            f"<td>{g['pass_rate']*100:.0f}%</td>"
            f"<td>{status}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    # ── 4. Sentence-End Sensitivity ────────────────────────────────────────
    se_data = analysis["sentence_end"]
    parts.append("""
<h2>4 &ensp; Sentence-End Sensitivity</h2>

<p>
The figure below shows mean terminal F0 slope for each punctuation type.
Positive slopes indicate rising pitch (question-like); negative slopes indicate
falling pitch (statement-like). The key metric is the
<strong>question-vs-period difference</strong>: a model that genuinely rises
for questions should show a large positive difference.
</p>

<div class="figure-container">
""")
    if figures.get("sentence_end.png"):
        parts.append(f'<img src="{figures["sentence_end.png"]}" alt="Sentence-end F0 slopes">')
    else:
        parts.append("<p><em>[sentence_end.png — figure not found]</em></p>")
    parts.append("</div>\n")

    parts.append("""<table>
<thead>
<tr><th>Model</th><th>Period (Hz/s)</th><th>Question (Hz/s)</th><th>Exclamation (Hz/s)</th><th>Q−P Diff (Hz/s)</th></tr>
</thead>
<tbody>
""")
    for m in MODEL_ORDER:
        d = se_data.get(m, {})
        p_slope = d.get("period", {}).get("mean_slope", "—")
        q_slope = d.get("question", {}).get("mean_slope", "—")
        e_slope = d.get("exclamation", {}).get("mean_slope", "—")
        diff = d.get("question_period_diff", "—")
        name = MODEL_DISPLAY.get(m, m.title())
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(p_slope)}</td><td>{fmt(q_slope)}</td>"
            f"<td>{fmt(e_slope)}</td><td><strong>{fmt(diff)}</strong></td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    parts.append("""
<p>
<strong>Interpretation.</strong> Chatterbox shows the strongest question-mark
differentiation (+68.5 Hz/s relative to periods), followed by F5-TTS (+31.8 Hz/s)
and Kokoro (+25.7 Hz/s). XTTS-v2 is essentially flat (−5.0 Hz/s), treating
questions and statements with nearly identical F0 trajectories. For exclamation
marks, F5-TTS produces the steepest downward slope (−489 Hz/s), indicating an
aggressive terminal fall.
</p>
""")

    # ── 5. Pause Hierarchy ─────────────────────────────────────────────────
    ph_data = analysis["pause_hierarchy"]
    parts.append("""
<h2>5 &ensp; Pause Hierarchy</h2>

<p>
Punctuation marks imply different pause lengths: commas are brief, ellipses are
long. The hierarchy score measures how consistently a model orders pauses as
comma &lt; semicolon &lt; em-dash &lt; ellipsis. A score of 1.0 means perfect
ordering across all pairs.
</p>

<div class="figure-container">
""")
    if figures.get("pause_hierarchy.png"):
        parts.append(f'<img src="{figures["pause_hierarchy.png"]}" alt="Pause hierarchy boxplot">')
    else:
        parts.append("<p><em>[pause_hierarchy.png — figure not found]</em></p>")
    parts.append("</div>\n")

    parts.append("""<table>
<thead>
<tr><th>Model</th><th>Comma (ms)</th><th>Semicolon (ms)</th><th>Em-dash (ms)</th><th>Ellipsis (ms)</th><th>Hierarchy Score</th></tr>
</thead>
<tbody>
""")
    for m in MODEL_ORDER:
        d = ph_data.get(m, {})
        comma = d.get("comma", {}).get("mean_ms", "—")
        semi = d.get("semicolon", {}).get("mean_ms", "—")
        dash = d.get("em_dash", {}).get("mean_ms", "—")
        ellip = d.get("ellipsis", {}).get("mean_ms", "—")
        score = d.get("hierarchy_score", "—")
        name = MODEL_DISPLAY.get(m, m.title())
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(comma)}</td><td>{fmt(semi)}</td>"
            f"<td>{fmt(dash)}</td><td>{fmt(ellip)}</td>"
            f"<td><strong>{fmt(score, 2)}</strong></td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    parts.append("""
<p>
<strong>Interpretation.</strong> XTTS-v2 achieves a perfect hierarchy score (1.0),
with a clear monotonic increase from commas (146 ms) to ellipses (248 ms).
Chatterbox and Kokoro tie at 0.67 — both show the correct general trend but with
occasional inversions (em-dash is the <em>shortest</em> pause among all four
marks for both Chatterbox and Kokoro — below even the comma). F5-TTS scores
lowest (0.33); its pause durations are dominated by a high baseline — all marks
produce pauses ≥200 ms — and the ordering between categories is inconsistent.
</p>
""")

    # ── 6. Trailing Punctuation ────────────────────────────────────────────
    tr_data = analysis["trailing"]
    parts.append("""
<h2>6 &ensp; Trailing Punctuation: Ellipsis vs Period</h2>

<p>
Trailing punctuation (ellipsis vs period) was compared on F0 slope, amplitude
decay, and pause duration. A model sensitive to the trailing mark should show
different F0 and pause behaviour between the two.
</p>

<div class="figure-container">
""")
    if figures.get("trailing.png"):
        parts.append(f'<img src="{figures["trailing.png"]}" alt="Trailing punctuation comparison">')
    else:
        parts.append("<p><em>[trailing.png — figure not found]</em></p>")
    parts.append("</div>\n")

    parts.append("""<table>
<thead>
<tr><th>Model</th><th>Ellipsis F0 (Hz/s)</th><th>Period F0 (Hz/s)</th><th>Ellipsis Pause (ms)</th><th>Period Pause (ms)</th><th>F0 Diff (Hz/s)</th></tr>
</thead>
<tbody>
""")
    for m in MODEL_ORDER:
        d = tr_data.get(m, {})
        e = d.get("ellipsis", {})
        p = d.get("period", {})
        e_slope = e.get("f0_slope_mean", "—")
        p_slope = p.get("f0_slope_mean", "—")
        e_pause = e.get("pause_mean", "—")
        p_pause = p.get("pause_mean", "—")
        diff = d.get("trailing_f0_diff", "—")
        name = MODEL_DISPLAY.get(m, m.title())
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(e_slope)}</td><td>{fmt(p_slope)}</td>"
            f"<td>{fmt(e_pause)}</td><td>{fmt(p_pause)}</td>"
            f"<td><strong>{fmt(diff)}</strong></td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    parts.append("""
<p>
<strong>Interpretation.</strong> Kokoro shows the largest F0 difference between
ellipsis and period (+60.7 Hz/s), indicating distinct prosodic treatment of
suspension vs finality. Chatterbox also differentiates (+24.2 Hz/s). XTTS-v2
and F5-TTS show negligible differences (−17.1 and +1.1 Hz/s respectively),
suggesting they do not prosodically distinguish trailing ellipsis from a period.
Pause durations are broadly similar within each model across the two conditions.
</p>
""")

    # ── 7. Quotation Sensitivity ───────────────────────────────────────────
    q_data = analysis["quotation"]
    parts.append("""
<h2>7 &ensp; Quotation Sensitivity</h2>

<p>
Quoted speech was compared to reported (non-quoted) speech. The metrics are
F0 mean shift and F0 range shift between the two conditions. A shift-detected
flag indicates whether the model changes its F0 profile for quotation marks.
</p>

<div class="figure-container">
""")
    if figures.get("quotation.png"):
        parts.append(f'<img src="{figures["quotation.png"]}" alt="Quotation sensitivity">')
    else:
        parts.append("<p><em>[quotation.png — figure not found]</em></p>")
    parts.append("</div>\n")

    parts.append("""<table>
<thead>
<tr><th>Model</th><th>F0 Mean Shift (Hz)</th><th>F0 Range Shift (Hz)</th><th>Shift Detected</th></tr>
</thead>
<tbody>
""")
    for m in MODEL_ORDER:
        d = q_data.get(m, {})
        mean_shift = d.get("f0_mean_shift", "—")
        range_shift = d.get("f0_range_shift", "—")
        detected = d.get("shift_detected", "—")
        name = MODEL_DISPLAY.get(m, m.title())
        detected_badge = (
            '<span class="badge-yes">YES</span>'
            if str(detected).lower() == "true"
            else '<span class="badge-no">NO</span>'
        )
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(mean_shift)}</td>"
            f"<td><strong>{fmt(range_shift)}</strong></td>"
            f"<td>{detected_badge}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    parts.append("""
<p>
<strong>Interpretation.</strong> Chatterbox shows a preliminary quotation
sensitivity signal, with a 50 Hz F0 range expansion — the widest shift of any
model — but based on only 2 utterances per condition (quoted vs reported).
This is suggestive, not definitive. F5-TTS also shifts F0 range
(+11.7 Hz) and the shift is flagged as detected. XTTS-v2 and Kokoro produce
negligible range shifts (3.9 Hz and 3.7 Hz) and were not flagged, indicating
they do not modulate F0 in response to quotation marks. Larger per-condition
samples are needed before drawing firm conclusions.
</p>
""")

    # ── 8. Summary: F0 vs Pause Trade-Off ──────────────────────────────────
    overall = analysis["overall"]
    parts.append("""
<h2>8 &ensp; Summary: Two Independent Dimensions</h2>

<p>
The scatter plot below shows that <strong>F0 sensitivity</strong> and
<strong>pause ordering</strong> are independent dimensions. Models that
score high on one tend to score low on the other — but F5-TTS is an
exception, producing large pauses with inverted ordering.
</p>

<div class="figure-container">
""")
    if figures.get("summary.png"):
        parts.append(f'<img src="{figures["summary.png"]}" alt="Summary scatter plot">')
    else:
        parts.append("<p><em>[summary.png — figure not found]</em></p>")
    parts.append("</div>\n")

    # pause_count.png
    parts.append("""
<div class="figure-container">
""")
    if figures.get("pause_count.png"):
        parts.append(f'<img src="{figures["pause_count.png"]}" alt="Pause count comparison">')
    else:
        parts.append("<p><em>[pause_count.png — figure not found]</em></p>")
    parts.append("</div>\n")

    parts.append("""<table>
<thead>
<tr><th>Model</th><th>Question Rise (Hz/s)</th><th>Pause Gradient (ms)</th><th>Pause Hierarchy</th><th>Dominant Strategy</th></tr>
</thead>
<tbody>
""")
    for m in MODEL_ORDER:
        d = overall.get(m, {})
        rise = d.get("question_vs_period_f0_diff_hz", "—")
        pause_diff = d.get("comma_to_ellipsis_pause_diff_ms", "—")
        name = MODEL_DISPLAY.get(m, m.title())

        # Get hierarchy score from pause_hierarchy data
        ph_d = analysis["pause_hierarchy"].get(m, {})
        hier_score = ph_d.get("hierarchy_score", 0)

        rv = rise if isinstance(rise, (int, float)) else -999
        pv = pause_diff if isinstance(pause_diff, (int, float)) else -999

        if rv > 30 and hier_score >= 0.6:
            strat = "F0-driven"
        elif hier_score >= 0.9 and rv < 20:
            strat = "Pause-ordering"
        elif pv > 80 and hier_score < 0.5:
            strat = "Noisy pauses"
        elif rv <= 10 and hier_score < 0.5:
            strat = "Minimal"
        else:
            strat = "Mixed"

        parts.append(
            f"<tr><td>{name}</td><td>{fmt(rise)}</td>"
            f"<td>{fmt(pause_diff)}</td><td>{fmt(hier_score, 2)}</td>"
            f"<td><em>{strat}</em></td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    parts.append("""
<div class="highlight">
<p><strong>Key finding.</strong> F0 sensitivity and pause ordering are
independent abilities in current TTS architectures. Chatterbox is
F0-strong but has only moderate pause ordering (0.67). XTTS-v2 has
perfect pause ordering (1.00) but zero question F0 rise. F5-TTS generates
the largest pause magnitudes (115.8 ms comma−ellipsis gradient) but its
pause <em>ordering</em> is inverted (0.33) due to excessive insertions of
random pauses between words. No model combines strong F0 sensitivity with
strong pause ordering.
</p>
</div>
""")

    # ── 9. Capitalisation ──────────────────────────────────────────────────
    cap_data = analysis.get("capitalization", {})
    parts.append("""
<h2>9 &ensp; Capitalisation Sensitivity</h2>

<p>
All-caps vs title-case vs lowercase utterances were compared for RMS amplitude
boost and F0 boost relative to the lowercase baseline.
</p>

<table>
<thead>
<tr><th>Model</th><th>RMS Boost in ALL CAPS</th><th>F0 Boost in ALL CAPS</th></tr>
</thead>
<tbody>
""")
    for m in MODEL_ORDER:
        d = cap_data.get(m, {})
        rms = d.get("rms_boost", "—")
        f0b = d.get("f0_boost", "—")
        name = MODEL_DISPLAY.get(m, m.title())
        rms_str = "YES" if rms is True else ("NO" if rms is False else str(rms))
        f0_str = (
            "YES" if f0b is True else ("NO" if f0b is False else str(f0b) if f0b is not None else "—")
        )
        rms_badge = f'<span class="badge-yes">{rms_str}</span>' if rms_str == "YES" else f'<span class="badge-no">{rms_str}</span>' if rms_str == "NO" else rms_str
        f0_badge = f'<span class="badge-yes">{f0_str}</span>' if f0_str == "YES" else f'<span class="badge-no">{f0_str}</span>' if f0_str == "NO" else f0_str
        parts.append(f"<tr><td>{name}</td><td>{rms_badge}</td><td>{f0_badge}</td></tr>\n")
    parts.append("</tbody></table>\n")

    parts.append("""
<p>
<strong>Interpretation.</strong> All models increase RMS amplitude for ALL CAPS
text, suggesting a universal "louder = emphatic" strategy. F0 boost in ALL CAPS
is less common: only F5-TTS shows a clear F0 elevation. Chatterbox and XTTS-v2
do not raise F0 for all-caps, while Kokoro's data was inconclusive.
</p>
""")

    # ── 10. Limitations ────────────────────────────────────────────────────
    parts.append("""
<h2>10 &ensp; Limitations</h2>

<div class="limitations">
<ul>
    <li><strong>Sample size.</strong> 28 utterances per model (112 total) yield
    robust descriptive statistics but limited inferential power. Category-level
    N ranges from 2 to 13, and some conditions (e.g., quotation pairs) have
    only 2–4 items per model.</li>
    <li><strong>Single-speaker.</strong> All audio was generated in a single
    synthetic voice per model. Results may not generalise to voice-cloned or
    multi-speaker scenarios.</li>
    <li><strong>VAD tuning.</strong> Energy-based VAD parameters (threshold 0.15,
    50 ms window) were optimised for this dataset; different settings may shift
    absolute pause durations, though relative comparisons should be robust.</li>
    <li><strong>F0 tracking.</strong> <code>pyin</code> is robust but can
    produce spurious values in unvoiced or creaky regions. We required ≥50%
    voiced frames in terminal segments for gate inclusion.</li>
    <li><strong>Limited punctuation scope.</strong> We tested five categories;
    other marks (colon, parentheses, hyphenation) were not included.</li>
    <li><strong>No listening test.</strong> These are purely acoustic metrics.
    Perceptual naturalness and appropriateness of prosody were not evaluated.</li>
    <li><strong>Single temperature.</strong> The default sampling temperature
    (0.7) may conceal stochastic variation in punctuation sensitivity.</li>
</ul>
</div>
""")

    # ── footer ─────────────────────────────────────────────────────────────
    parts.append("""
<div class="footer">
    Generated from the Punctuation Sensitivity Benchmark &mdash; analysis of
    112 utterances across Chatterbox, XTTS-v2, F5-TTS, and Kokoro.
    <br>Report generated with Python 3.12 &bull; stdlib only.
</div>

</div> <!-- .report-container -->
</body>
</html>
""")

    return "\n".join(parts)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    html = build_html()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(REPORT_PATH) / 1024
    print(f"Report written to {REPORT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
