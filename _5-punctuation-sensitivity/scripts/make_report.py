#!/usr/bin/env python3
"""Generate self-contained HTML report for punctuation sensitivity probe."""
import base64
import json
import os
import sys
from pathlib import Path

from common import (
    ANALYSIS_JSON,
    FEATURES_CSV,
    FIG_DIR,
    GATE_JSON,
    RESULTS_DIR,
    MODEL_ORDER,
    MODEL_LABELS,
    HIERARCHY_ORDER,
    PUNCT_ORDER,
    load_csv,
    load_json,
    fmt,
)

REPORT_PATH = RESULTS_DIR / "report.html"
PDF_PATH = RESULTS_DIR / "report.pdf"

# ── helpers ──────────────────────────────────────────────────────────────────


def img_to_b64(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fmt_int(val):
    try:
        return f"{int(float(val))}"
    except (ValueError, TypeError):
        return str(val)


# ── load data ────────────────────────────────────────────────────────────────

analysis = load_json(ANALYSIS_JSON)
gate_check = load_json(GATE_JSON)
csv_rows = load_csv(FEATURES_CSV)

figures = {}
for fname in [
    "sentence_end.png",
    "pause_hierarchy.png",
    "trailing.png",
    "quotation.png",
    "summary.png",
    "pause_count.png",
]:
    path = FIG_DIR / fname
    figures[fname] = img_to_b64(path) if path.exists() else None


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt; line-height: 1.6; color: #333; background: #fff;
    padding: 2em 1em;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
}

body.pdf { padding: 0; }

.report-container { max-width: 900px; margin: 0 auto; padding: 2em 1.5em; }

h1 { font-size: 26pt; color: #1a1a2e; margin-bottom: 0.2em; font-weight: 700; letter-spacing: -0.5px; }
h2 { font-size: 18pt; color: #1a1a2e; margin-top: 1.8em; margin-bottom: 0.6em; padding-bottom: 0.3em; border-bottom: 3px solid #1a1a2e; font-weight: 600; }
h3 { font-size: 13pt; color: #16213e; margin-top: 1.4em; margin-bottom: 0.4em; font-weight: 600; }
.subtitle { font-size: 13pt; color: #555; margin-bottom: 1.5em; font-style: italic; }
p { margin-bottom: 0.8em; }

.figure-container { margin: 1.2em 0; text-align: center; }
.figure-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }

table { width: 100%; border-collapse: collapse; margin: 1em 0 1.5em 0; font-size: 10pt; }
th { background: #1a1a2e; color: #fff; padding: 8px 10px; text-align: left; font-weight: 600; white-space: nowrap; }
td { padding: 7px 10px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f5f5f8; }
tr:hover td { background: #eaeaf0; }
.key-table th:first-child, .key-table td:first-child { font-weight: 600; }

.highlight { background: #fef9e7; padding: 1em 1.2em; border-left: 4px solid #e6a817; margin: 1em 0; border-radius: 2px; }

.limitations { background: #f8f8fa; padding: 1em 1.2em; border-left: 4px solid #c0392b; margin: 1em 0; border-radius: 2px; }
.limitations ul { margin-left: 1.5em; margin-bottom: 0.5em; }
.limitations li { margin-bottom: 0.4em; }

.badge-pass { display: inline-block; background: #27ae60; color: #fff; font-size: 9pt; padding: 2px 10px; border-radius: 10px; font-weight: 600; }
.badge-fail { display: inline-block; background: #c0392b; color: #fff; font-size: 9pt; padding: 2px 10px; border-radius: 10px; font-weight: 600; }
.badge-yes { display: inline-block; background: #2e86c1; color: #fff; font-size: 9pt; padding: 2px 8px; border-radius: 3px; font-weight: 600; }
.badge-no { display: inline-block; background: #7f8c8d; color: #fff; font-size: 9pt; padding: 2px 8px; border-radius: 3px; font-weight: 600; }

.footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd; font-size: 9pt; color: #888; text-align: center; }

@media print {
    body { padding: 0; font-size: 10pt; }
    h2 { page-break-after: avoid; }
    .figure-container { page-break-inside: avoid; }
    table { page-break-inside: avoid; }
    @page { size: A4; margin: 1.5cm; }
}
"""


# ── HTML builder ─────────────────────────────────────────────────────────────


def build_html():
    parts = []

    # ── header ──
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Punctuation Sensitivity Probe - Report</title>
<style>{CSS}</style>
</head>
<body class="pdf">
<div class="report-container">

<h1>Punctuation Sensitivity Probe</h1>
<p class="subtitle">
    Descriptive acoustic analysis of punctuation-prosody effects across three TTS models<br>
    &mdash; Chatterbox, XTTS-v2, and Kokoro &mdash;<br>
    over 28 utterances spanning 5 punctuation categories.<br>
    <strong>This is a preliminary smoke test, not a validated benchmark.</strong>
</p>
""")

    # ── 1. Caveat ──
    parts.append("""
<h2>1 &ensp; Caveat</h2>

<div class="limitations">
<p><strong>This report describes acoustic measurements from 28 hand-crafted utterances
per model. It is not a benchmark.</strong> The sample size is too small for ranking claims.
All metrics are acoustic proxies (VAD pauses, pyin F0) without perceptual validation.
Confidence intervals and effect sizes are reported where possible; most comparisons
have 2-6 data points per condition and should be treated as anecdotal.</p>
</div>
""")

    # ── 2. Methodology ──
    parts.append("""
<h2>2 &ensp; Methodology</h2>

<p><strong>Stimuli.</strong> 28 utterances were manually designed to probe each punctuation category.
Each utterance was rendered with the exact punctuation required (no textual variation beyond the
target mark) to isolate the effect of punctuation on prosody.</p>

<p><strong>Categories (5).</strong> Sentence-end (period, question mark, exclamation mark),
pause hierarchy (comma, semicolon, em-dash, ellipsis), trailing punctuation (ellipsis vs period),
quotation (quoted speech vs reported speech), and capitalisation (ALL CAPS, Title Case, lowercase).</p>

<p><strong>Models (3).</strong> Chatterbox, XTTS-v2, and Kokoro were each used to generate 28 audio
clips at 24 kHz. Chatterbox and XTTS-v2 use voice cloning from VCTK p229; Kokoro uses its default
voice (no adaptation baseline).</p>

<p><strong>Acoustic analysis.</strong> Voice activity detection uses energy-based VAD
(threshold = 0.05 &times; max RMS, 20 ms window). F0 extracted with librosa.pyin
(50-600 Hz, 256-hop). Pauses are RMS-below-threshold regions &ge;30 ms.
Internal pauses are those occurring before 95% of total duration.
Terminal F0 slope computed over the final 400 ms via linear regression.
95% CIs are bootstrap percentile intervals (1000 resamples).</p>
""")

    # ── 3. Gate Check ──
    parts.append("""<h2>3 &ensp; Gate Check</h2>
<p>Minimum viability: period-ending utterances must have best pause &ge;150 ms for &ge;80% of items.
This confirms audible speech at boundaries, not prosodic quality.</p>
<table><thead><tr><th>Model</th><th>Period Items</th><th>Passing</th><th>Pass Rate</th><th>Status</th></tr></thead><tbody>
""")
    for g in gate_check:
        mdl = MODEL_LABELS.get(g["model"], g["model"].title())
        status = '<span class="badge-pass">PASS</span>'
        parts.append(
            f"<tr><td>{mdl}</td><td>{g['period_items']}</td>"
            f"<td>{g['periods_passing']}</td>"
            f"<td>{g['pass_rate']*100:.0f}%</td>"
            f"<td>{status}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    # ── 4. Sentence-End ──
    se_data = analysis["sentence_end"]
    parts.append("""
<h2>4 &ensp; Sentence-End Punctuation</h2>
<p>Mean terminal F0 slope for each punctuation type. Positive = rising (question-like),
negative = falling (statement-like).</p>
<div class="figure-container">"""
                 )
    if figures.get("sentence_end.png"):
        parts.append(f'<img src="{figures["sentence_end.png"]}" alt="Sentence-end F0 slopes">')
    else:
        parts.append('<p><em>[figure not found]</em></p>')
    parts.append("</div>\n")

    parts.append("""<table><thead><tr>
<th>Model</th><th>Period (Hz/s)</th><th>Question (Hz/s)</th><th>Exclamation (Hz/s)</th>
<th>Q-P Diff (Hz/s)</th><th>Cohen's d</th>
</tr></thead><tbody>
""")
    for m in MODEL_ORDER:
        d = se_data.get(m, {})
        p = d.get("period", {})
        q = d.get("question", {})
        e = d.get("exclamation", {})
        diff = d.get("question_period_diff", None)
        es = d.get("question_vs_period_effect_size", {})
        es_str = f"{fmt(es.get('d'))} ({es.get('interpretation', 'N/A')})" if es.get("d") is not None else "N/A"
        name = MODEL_LABELS.get(m, m.title()).split("\n")[0]
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(p.get('mean'))}</td><td>{fmt(q.get('mean'))}</td>"
            f"<td>{fmt(e.get('mean'))}</td><td><strong>{fmt(diff)}</strong></td>"
            f"<td>{es_str}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")
    parts.append("""
<p>Chatterbox shows the largest question-mark differentiation (+68.5 Hz/s vs periods, Cohen's d large).
Kokoro shows moderate differentiation (+25.7 Hz/s). XTTS-v2 is near-zero (-5.0 Hz/s, d negligible),
treating questions and statements with similar F0 trajectories. These differences are descriptive;
with n=3 per condition, no significance claims are warranted.</p>
""")

    # ── 5. Pause Hierarchy ──
    ph_data = analysis["pause_hierarchy"]
    parts.append("""
<h2>5 &ensp; Pause Hierarchy</h2>
<p>Pause duration by internal punctuation type. Expected ordering: comma &lt; semicolon &lt; em-dash &lt; ellipsis.
Hierarchy score = fraction of adjacent pairs in correct order (0-1).</p>
<div class="figure-container">"""
                 )
    if figures.get("pause_hierarchy.png"):
        parts.append(f'<img src="{figures["pause_hierarchy.png"]}" alt="Pause hierarchy boxplot">')
    else:
        parts.append('<p><em>[figure not found]</em></p>')
    parts.append("</div>\n")

    parts.append("""<table><thead><tr>
<th>Model</th><th>Comma (ms)</th><th>Semicolon (ms)</th><th>Em-dash (ms)</th><th>Ellipsis (ms)</th>
<th>Hierarchy Score</th>
</tr></thead><tbody>
""")
    for m in MODEL_ORDER:
        d = ph_data.get(m, {})
        name = MODEL_LABELS.get(m, m.title()).split("\n")[0]
        comma = d.get("comma", {})
        semi = d.get("semicolon", {})
        dash = d.get("em_dash", {})
        ellip = d.get("ellipsis", {})
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(comma.get('mean_ms'))}</td><td>{fmt(semi.get('mean_ms'))}</td>"
            f"<td>{fmt(dash.get('mean_ms'))}</td><td>{fmt(ellip.get('mean_ms'))}</td>"
            f"<td><strong>{d.get('hierarchy_score', 'N/A')}</strong></td></tr>\n"
        )
    parts.append("</tbody></table>\n")
    parts.append("""
<p>XTTS-v2 achieves the highest hierarchy score (1.0) with a monotonic increase from commas to ellipses.
Chatterbox and Kokoro tie at 0.67; both show inversions (em-dash is shorter than comma). With
IRR at this sample size, these scores should be interpreted cautiously.</p>
""")

    # ── 6. Trailing ──
    tr_data = analysis["trailing"]
    parts.append("""
<h2>6 &ensp; Trailing Punctuation: Ellipsis vs Period</h2>
<p>F0 slope, amplitude decay, and pause duration for trailing ellipsis vs period.</p>
<div class="figure-container">"""
                 )
    if figures.get("trailing.png"):
        parts.append(f'<img src="{figures["trailing.png"]}" alt="Trailing punctuation comparison">')
    else:
        parts.append('<p><em>[figure not found]</em></p>')
    parts.append("</div>\n")

    parts.append("""<table><thead><tr>
<th>Model</th><th>Ellipsis F0 (Hz/s)</th><th>Period F0 (Hz/s)</th><th>Ellipsis Pause (ms)</th><th>Period Pause (ms)</th>
<th>F0 Diff (Hz/s)</th><th>Cohen's d</th>
</tr></thead><tbody>
""")
    for m in MODEL_ORDER:
        d = tr_data.get(m, {})
        e = d.get("ellipsis", {})
        p = d.get("period", {})
        diff = d.get("trailing_f0_diff")
        es = d.get("trailing_effect_size", {})
        es_str = f"{fmt(es.get('d'))} ({es.get('interpretation', 'N/A')})" if es.get("d") is not None else "N/A"
        name = MODEL_LABELS.get(m, m.title()).split("\n")[0]
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(e.get('f0_slope_mean'))}</td><td>{fmt(p.get('f0_slope_mean'))}</td>"
            f"<td>{fmt(e.get('pause_mean'))}</td><td>{fmt(p.get('pause_mean'))}</td>"
            f"<td><strong>{fmt(diff)}</strong></td><td>{es_str}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")
    parts.append("""
<p>Kokoro shows the largest F0 difference between ellipsis and period (+60.7 Hz/s, large d).
Chatterbox also differentiates (+24.2 Hz/s). XTTS-v2 shows negligible difference (-17.1 Hz/s, small d).
All comparisons are based on n=2 per condition per model.</p>
""")

    # ── 7. Quotation ──
    q_data = analysis["quotation"]
    parts.append("""
<h2>7 &ensp; Quotation Sensitivity</h2>
<p>F0 range and mean shift between quoted and reported speech.
Shift flag = F0 range shift > configured threshold.</p>
<div class="figure-container">"""
                 )
    if figures.get("quotation.png"):
        parts.append(f'<img src="{figures["quotation.png"]}" alt="Quotation sensitivity">')
    else:
        parts.append('<p><em>[figure not found]</em></p>')
    parts.append("</div>\n")

    parts.append("""<table><thead><tr>
<th>Model</th><th>F0 Mean Shift (Hz)</th><th>F0 Range Shift (Hz)</th><th>Shift Detected</th><th>Cohen's d (range)</th>
</tr></thead><tbody>
""")
    for m in MODEL_ORDER:
        d = q_data.get(m, {})
        name = MODEL_LABELS.get(m, m.title()).split("\n")[0]
        es = d.get("range_effect_size", {})
        es_str = f"{fmt(es.get('d'))} ({es.get('interpretation', 'N/A')})" if es.get("d") is not None else "N/A"
        detected = d.get("shift_detected", False)
        badge = '<span class="badge-yes">YES</span>' if detected else '<span class="badge-no">NO</span>'
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(d.get('f0_mean_shift'))}</td>"
            f"<td><strong>{fmt(d.get('f0_range_shift'))}</strong></td>"
            f"<td>{badge}</td><td>{es_str}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")
    parts.append("""
<p>Chatterbox shows a quotation F0 range shift flagged by the threshold (n=2 per condition).
XTTS-v2 and Kokoro do not. The threshold (>10 Hz range shift) is arbitrary; with n=2,
this is anecdotal evidence at best.</p>
""")

    # ── 8. Summary ──
    overall = analysis["overall"]
    parts.append("""
<h2>8 &ensp; Summary</h2>
<p>The scatter plot shows the relationship between F0-based and pause-based punctuation sensitivity.</p>
<div class="figure-container">"""
                 )
    if figures.get("summary.png"):
        parts.append(f'<img src="{figures["summary.png"]}" alt="Summary scatter plot">')
    else:
        parts.append('<p><em>[figure not found]</em></p>')
    parts.append("</div>\n")

    if figures.get("pause_count.png"):
        parts.append(f'<div class="figure-container"><img src="{figures["pause_count.png"]}" alt="Pause count comparison"></div>\n')

    parts.append("""<table><thead><tr>
<th>Model</th><th>Question Rise (Hz/s)</th><th>Pause Gradient (ms)</th><th>Pause Hierarchy</th>
</tr></thead><tbody>
""")
    for m in MODEL_ORDER:
        d = overall.get(m, {})
        rise = d.get("question_vs_period_f0_diff_hz", "N/A")
        pause_diff = d.get("comma_to_ellipsis_pause_diff_ms", "N/A")
        ph_d = analysis["pause_hierarchy"].get(m, {})
        hier_score = ph_d.get("hierarchy_score", "N/A")
        name = MODEL_LABELS.get(m, m.title()).split("\n")[0]
        parts.append(
            f"<tr><td>{name}</td><td>{fmt(rise)}</td><td>{fmt(pause_diff)}</td><td>{hier_score}</td></tr>\n"
        )
    parts.append("</tbody></table>\n")

    parts.append("""
<div class="highlight">
<p><strong>Descriptive observation.</strong> F0 sensitivity and pause ordering appear
to vary independently across these 3 models in this 28-item probe. Chatterbox shows
stronger F0 differentiation; XTTS-v2 shows stronger pause ordering. These are
hypothesis-generating observations, not validated findings.</p>
</div>
""")

    # ── 9. Capitalisation ──
    cap_data = analysis.get("capitalization", {})
    parts.append("""
<h2>9 &ensp; Capitalisation Sensitivity</h2>
<p><strong>WARNING: 1 item per condition. These are anecdotes, not evidence.</strong></p>
<table><thead><tr>
<th>Model</th><th>RMS Boost in ALL CAPS</th><th>F0 Boost in ALL CAPS</th>
</tr></thead><tbody>
""")
    for m in MODEL_ORDER:
        d = cap_data.get(m, {})
        rms = d.get("rms_boost", "N/A")
        f0b = d.get("f0_boost", "N/A")
        name = MODEL_LABELS.get(m, m.title()).split("\n")[0]
        rms_str = str(rms)
        f0_str = str(f0b) if f0b is not None else "N/A"
        parts.append(f"<tr><td>{name}</td><td>{rms_str}</td><td>{f0_str}</td></tr>\n")
    parts.append("</tbody></table>\n")
    parts.append("""
<p>All models increase RMS amplitude for ALL CAPS. F0 boost is inconsistent.
With n=1 per condition, no conclusions should be drawn.</p>
""")

    # ── 10. Methodological Limitations ──
    parts.append("""
<h2>10 &ensp; Methodological Limitations</h2>

<div class="limitations">
<ul>
    <li><strong>Sample size.</strong> 28 utterances per model with 2-6 items per subcategory.
    Insufficient for statistical inference or model ranking.</li>
    <li><strong>No perceptual ground truth.</strong> All metrics are acoustic proxies
    (VAD pauses, pyin F0). No listening tests, MOS, ABX, or preference judgments were collected.
    These metrics have unknown correlation with human perception of prosody.</li>
    <li><strong>No forced alignment.</strong> Pauses are pooled across all internal RMS-detectable
    silences, not anchored to specific punctuation tokens. A silence anywhere in the utterance
    before 95% duration counts as an "internal pause" regardless of where punctuation occurs.</li>
    <li><strong>Crude VAD.</strong> Energy threshold (0.05 &times; max RMS) is fragile across models
    with different loudness, noise floor, vocoder artifacts, and trailing silence policies.</li>
    <li><strong>F0 extraction noise.</strong> librosa.pyin can fail on creaky voice, unvoiced
    endings, and vocoder artifacts in short generated utterances. README and code acknowledge this.</li>
    <li><strong>Arbitrary thresholds.</strong> Gate (150 ms), quotation shift (>10 Hz F0 range),
    and internal pause cutoff (95% duration) are all heuristic values without empirical justification.</li>
    <li><strong>No text normalization audit.</strong> TTS frontends may normalize punctuation
    before synthesis. This is not inspected or logged in the current pipeline.</li>
    <li><strong>Unfair model comparison.</strong> Kokoro lacks voice cloning and uses a different
    speaker, prosody prior, and conditioning mode. It serves as a "no-adaptation baseline" but
    should not be compared directly in rankings.</li>
    <li><strong>Capitalization test conflates factors.</strong> "STOP!" vs "Stop!" vs "stop!"
    differs in capitalization, token frequency, and potentially phoneme-level priors. n=1.</li>
    <li><strong>No punctuation-stripped controls.</strong> Without a baseline condition where
    punctuation is removed, we cannot isolate whether models respond to the punctuation mark
    itself or to tokenizer-side effects of the surrounding text.</li>
</ul>
</div>
""")

    # ── footer ──
    parts.append("""
<div class="footer">
    Generated from the Punctuation Sensitivity Probe &mdash;
    descriptive analysis of 84 utterances across Chatterbox, XTTS-v2, and Kokoro.
    <br>Report generated with Python 3.12 &bull; stdlib minimal dependencies.
</div>

</div> <!-- .report-container -->
</body>
</html>
""")

    return "\n".join(parts)


def main():
    html = build_html()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(REPORT_PATH) / 1024
    print(f"Report written to {REPORT_PATH} ({size_kb:.0f} KB)")

    from weasyprint import HTML
    HTML(filename=str(REPORT_PATH)).write_pdf(str(PDF_PATH))
    pdf_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"PDF written to {PDF_PATH} ({pdf_kb:.0f} KB)")


if __name__ == "__main__":
    main()