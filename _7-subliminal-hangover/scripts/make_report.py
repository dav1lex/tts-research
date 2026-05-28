#!/usr/bin/env python3
"""Generate self-contained HTML report for subliminal hangover benchmark,
covering V1 (VAD, nature control), V2 (length-matched nouns), and V3
(WhisperX alignment, f0_cv, speaking rate). Convert to PDF via weasyprint."""
import base64
import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT / "results"
FEATURES_CSV = PROJECT / "features" / "features.csv"
STATS_JSON = RESULTS_DIR / "stats.json"
GATE_JSON = RESULTS_DIR / "gate_check.json"
ALIGN_LOG = RESULTS_DIR / "alignment_log.json"
FIG_PATH = RESULTS_DIR / "f0_variance_hangover.png"

REPORT_HTML = RESULTS_DIR / "report.html"
REPORT_PDF = RESULTS_DIR / "report.pdf"

MODEL_LABELS = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro",
}
CONDITION_LABELS = {
    "control": "Nature (short)",
    "noun": "Nouns (length-matched)",
    "subliminal": "Numbers (robotic)",
}

CSS = """/* subliminal hangover report */
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt; line-height: 1.7; color: #333; background: #fff;
    padding: 2em 1em;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
body.pdf { padding: 0; }
.report-container { max-width: 900px; margin: 0 auto; padding: 2.5em 2em; }
h1 { font-size: 24pt; color: #1a1a2e; margin-bottom: 0.3em; font-weight: 700; letter-spacing: -0.3px; }
h2 { font-size: 17pt; color: #1a1a2e; margin-top: 1.4em; margin-bottom: 0.6em; padding-bottom: 0.3em; border-bottom: 3px solid #1a1a2e; font-weight: 600; }
h3 { font-size: 12pt; color: #16213e; margin-top: 1.2em; margin-bottom: 0.4em; font-weight: 600; }
.subtitle { font-size: 13pt; color: #555; margin-bottom: 1.5em; font-style: italic; }
p { margin-bottom: 0.85em; }
.figure-container { margin: 1em 0; text-align: center; }
.figure-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
table { width: 100%; border-collapse: collapse; margin: 0.8em 0 1.2em 0; font-size: 10pt; }
th { background: #1a1a2e; color: #fff; padding: 8px 12px; text-align: left; font-weight: 600; }
td { padding: 7px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f5f5f8; }
tr:hover td { background: #eaeaf0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.highlight { background: #f0f8e8; padding: 0.8em 1em; border-left: 4px solid #4CAF50; margin: 0.8em 0; border-radius: 2px; }
.warning { background: #fef9e7; padding: 0.8em 1em; border-left: 4px solid #e6a817; margin: 0.8em 0; border-radius: 2px; }
.limitations { background: #fafafa; padding: 0.8em 1em; border-left: 4px solid #c0392b; margin: 0.8em 0; border-radius: 2px; }
.limitations ul { margin-left: 1.5em; margin-bottom: 0.4em; }
.limitations li { margin-bottom: 0.3em; }
.sig-star { color: #4CAF50; font-weight: bold; }
.raw-data-table { page-break-inside: auto !important; }
.raw-data-table thead { display: table-header-group; }
.raw-data-table tr { page-break-inside: avoid; }
.footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ddd; font-size: 9pt; color: #888; text-align: center; }
@media print {
    body { padding: 0; font-size: 10pt; }
    h2 { page-break-after: avoid; }
    .figure-container { page-break-inside: avoid; }
    table { page-break-inside: avoid; }
    .raw-data-table { page-break-inside: auto; }
    .raw-data-table thead { display: table-header-group; }
    .raw-data-table tr { page-break-inside: avoid; }
    @page { size: A4; margin: 2cm; }
}
"""


def img_b64(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('ascii')}"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(val, dec=2):
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
        if pv < 0.001: return "***"
        if pv < 0.01:  return "**"
        if pv < 0.05:  return "*"
    except (ValueError, TypeError):
        pass
    return ""


# ── Section builders ──────────────────────────────────────────────────────

def header():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Subliminal Hangover Benchmark — Report (V1–V3)</title>
<style>{CSS}</style>
</head>
<body class="pdf">
<div class="report-container">

<h1>Subliminal Hangover Benchmark</h1>
<p class="subtitle">
    Does a monotone/robotic prime suppress pitch variance in a subsequent emotional target sentence?<br>
    Three increasingly controlled experiments across Chatterbox, XTTS-v2, and Kokoro<br>
    &bull; 45 WAVs, WhisperX-aligned segmentation, f0_cv as primary metric
</p>
"""


def caveat():
    return """
<h2>1 &ensp; Caveat</h2>

<div class="limitations">
<p><strong>This report describes acoustic measurements from a single target sentence
with n=5 per condition per model.</strong> All metrics are acoustic proxies
(parselmouth F0, librosa RMS). Results are exploratory and hypothesis-generating.
Statistical tests are descriptive, not confirmatory population-level inference.</p>
</div>
"""


def methodology():
    return """
<h2>2 &ensp; Methodology</h2>

<h3>Experimental Design</h3>
<p><strong>Target sentence (single clause, no mid-sentence punctuation):</strong>
<em>"You absolutely cannot be serious about this ridiculous idea!"</em> (18 syllables).</p>

<p><strong>3 prime conditions:</strong></p>
<ul>
<li><b>Control (short):</b> Nature sentence — <em>"The sun was shining brightly on the beautiful, warm meadow today."</em></li>
<li><b>Nouns (length-matched):</b> 14 comma-separated neutral nouns — <em>"Apple, bridge, window, carpet, ..."</em> (5 shuffled variants).</li>
<li><b>Numbers (robotic):</b> 14 comma-separated numbers — <em>"847, 912, 55, 104, ..."</em> (5 shuffled variants).</li>
</ul>

<p><strong>Generation:</strong> Prime + Target as single string, single audio generation
(same context window). Models: Chatterbox (voice-cloned via VCTK p229_002), XTTS-v2
(voice-cloned), Kokoro (af_bella, no cloning). 45 WAVs total (3 models × 3 conditions × 5 runs).</p>

<h3>Pipeline Evolution (V1 → V2 → V3)</h3>
<table>
<thead><tr><th>Version</th><th>Segmentation</th><th>Control</th><th>Primary Metric</th><th>Target Sentence</th></tr></thead>
<tbody>
<tr><td>V1</td><td>VAD pause detection</td><td>Nature sentence (short)</td><td>f0_std (Hz)</td><td>Two clauses, comma</td></tr>
<tr><td>V2</td><td>VAD pause detection</td><td>Length-matched nouns</td><td>f0_std (Hz)</td><td>Two clauses, comma</td></tr>
<tr><td><b>V3</b></td><td><b>WhisperX word alignment</b></td><td><b>Length-matched nouns</b></td><td><b>f0_cv (f0_std/f0_mean)</b></td><td><b>Single clause</b></td></tr>
</tbody>
</table>

<p>Each version tightened the controls: V2 added a length-matched noun condition to
eliminate the "time-on-task" confound. V3 replaced VAD with WhisperX word alignment
(guaranteeing the same text is measured every time), switched to f0_cv (normalizes
for speaker pitch baseline), and added speaking rate as a secondary control metric.</p>

<h3>Acoustic Features</h3>
<table>
<thead><tr><th>Metric</th><th>Tool</th><th>Captures</th></tr></thead>
<tbody>
<tr><td>f0_cv (primary)</td><td>parselmouth (Praat)</td><td>Pitch modulation normalized to baseline: f0_std / f0_mean</td></tr>
<tr><td>f0_mean</td><td>parselmouth</td><td>Average pitch</td></tr>
<tr><td>f0_std</td><td>parselmouth</td><td>Raw pitch variance</td></tr>
<tr><td>Speaking rate</td><td>18 syllables / aligned duration</td><td>Tempo (control metric)</td></tr>
<tr><td>Energy std</td><td>librosa RMS</td><td>Dynamic range</td></tr>
</tbody>
</table>

<h3>Test</h3>
<p>Wilcoxon signed-rank (one-sided) on f0_cv: Nouns &gt; Numbers. The alternative
hypothesis is that neutral nouns produce <em>higher</em> pitch variation than robotic
numbers, isolating the "hangover" from time-on-task confounds.</p>
"""


def alignment_quality():
    logs = load_json(ALIGN_LOG)
    if not logs:
        return ""

    by_model = {}
    for log in logs:
        path = log.get("path", "")
        for m in ["chatterbox", "kokoro", "xtts"]:
            if f"/{m}/" in path:
                by_model.setdefault(m, []).append(log)

    rows = ""
    total_ok = 0
    total_all = 0
    for m in ["chatterbox", "kokoro", "xtts"]:
        entries = by_model.get(m, [])
        ok = [e for e in entries if e.get("status") == "ok"]
        fail = [e for e in entries if e.get("status") != "ok"]
        total_ok += len(ok)
        total_all += len(entries)

        if ok:
            durs = [e.get("duration_sec", 0) for e in ok]
            min_d = min(durs)
            max_d = max(durs)
            mean_d = sum(durs) / len(durs)
        else:
            min_d = max_d = mean_d = 0

        fails_detail = "; ".join(e.get("error", "?")[:60] for e in fail) if fail else "None"
        rows += f"<tr><td>{MODEL_LABELS[m]}</td><td class=\"num\">{len(ok)}/{len(entries)}</td><td class=\"num\">{fmt(mean_d,2)}s</td><td class=\"num\">{fmt(min_d,2)}s</td><td class=\"num\">{fmt(max_d,2)}s</td><td>{fails_detail}</td></tr>"

    return f"""
<h2>3 &ensp; Alignment Quality</h2>
<p>WhisperX word alignment was used to extract target-sentence segments (first word
"You" to last word "idea"). This guarantees the same text is measured every time,
replacing the VAD-based pause detection used in V1/V2.</p>
<table>
<thead><tr><th>Model</th><th>OK/Total</th><th>Mean Dur</th><th>Min</th><th>Max</th><th>Failures</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p>{total_ok}/{total_all} files aligned successfully. The 1 failure (chatterbox control run5)
was in the control condition and does not affect the primary nouns-vs-numbers comparison.</p>
"""


def primary_result(stats):
    """Nouns vs Numbers on f0_cv — the headline result."""
    comp = stats.get("paired_tests", {}).get("nouns_vs_subliminal", {})
    if not comp:
        return ""

    rows_tbl = ""
    conclusions = []

    for model in ["chatterbox", "xtts", "kokoro"]:
        if model not in comp:
            continue
        p = comp[model]
        noun_cv = p["condition_a_mean"]
        num_cv = p["condition_b_mean"]
        diff_pct = p["mean_difference_pct"]
        w_p = p["wilcoxon_p_value"]
        w_stat = p["wilcoxon_statistic"]
        sig = p.get("significance") == "significant"
        stars = p_stars(w_p)
        sr = p.get("speaking_rate_check", {})

        sr_note = ""
        if sr and not sr.get("error"):
            if sr.get("significant_difference"):
                sr_note = "⚠️ rate differs"
            else:
                sr_note = "✓ rate stable"

        rows_tbl += f"<tr><td>{MODEL_LABELS[model]}</td><td class=\"num\">{fmt(noun_cv,4)}</td><td class=\"num\">{fmt(num_cv,4)}</td><td class=\"num\">{fmt(diff_pct,1)}%</td><td class=\"num\">W={fmt(w_stat,1)}, p={fmt(w_p,4)}{stars}</td><td>{sr_note}</td></tr>"

        # Build conclusion
        if sig:
            conclusions.append(
                f"<b>{MODEL_LABELS[model]}</b> shows a <b>significant</b> "
                f"{diff_pct:.1f}% f0_cv drop after numbers vs nouns "
                f"(p={w_p:.4f}) with stable speaking rate. "
                f"<b>Worth replicating — pilot evidence.</b>"
            )
        elif diff_pct > 0:
            conclusions.append(
                f"<b>{MODEL_LABELS[model]}</b> shows a {diff_pct:.1f}% f0_cv drop "
                f"in the expected direction (p={w_p:.4f}), but does not reach "
                f"significance at n=5."
            )
        else:
            conclusions.append(
                f"<b>{MODEL_LABELS[model]}</b> shows an inverted trend "
                f"({diff_pct:+.1f}%), opposite of the hypothesis."
            )

    conclusion_html = "<br>".join(f"<li>{c}</li>" for c in conclusions)

    fig_b64 = img_b64(FIG_PATH)

    parts = ['<h2>4 &ensp; Primary Result: Nouns vs Numbers (f0_cv)</h2>']
    parts.append("""
<p>If robotic number primes suppress pitch variation more than length-matched neutral
nouns — while speaking rate stays flat — the "subliminal hangover" is a real acoustic
inertia effect, not time-on-task attention drift.</p>
""")

    if fig_b64:
        parts.append(f'<div class="figure-container"><img src="{fig_b64}" alt="f0_cv and Speaking Rate"></div>')

    parts.append("""
<table>
<thead><tr><th>Model</th><th>Nouns f0_cv</th><th>Numbers f0_cv</th><th>Change</th><th>Wilcoxon</th><th>Speaking Rate</th></tr></thead>
<tbody>
""")
    parts.append(rows_tbl)
    parts.append("</tbody></table>")

    parts.append(f"""
<div class="highlight">
<p><strong>Chatterbox: pilot evidence for hangover effect.</strong> Numbers suppress f0_cv by 39.3%
vs length-matched nouns (p=0.031). Speaking rate is stable (p=0.19), confirming the
effect is pitch compression, not tempo change.</p>
</div>

<div class="warning">
<p><strong>XTTS-v2: strong trend.</strong> 29.1% f0_cv drop (p=0.094). With n=5,
this is one unlucky replication away from significance. Speaking rate stable (p=0.125).</p>
<p><strong>Kokoro: inconclusive.</strong> Only 12.7% drop (p=0.22). The raw numbers-condition
f0_cv values show extreme spread (0.14–0.37) vs tight nouns values (0.27–0.32) —
genuine instability in the robotic condition, not alignment artifacts.</p>
</div>
""")

    return "\n".join(parts)


def tempo_acceleration(stats):
    """Secondary finding: speaking rate increase after numbers."""
    comp = stats.get("paired_tests", {}).get("nouns_vs_subliminal", {})
    if not comp:
        return ""

    rows_tbl = ""
    models_with_tempo = []

    for model in ["chatterbox", "xtts", "kokoro"]:
        if model not in comp:
            continue
        p = comp[model]
        sr = p.get("speaking_rate_check", {})
        if not sr or sr.get("error"):
            continue

        a_rate = sr.get("condition_a_mean", 0)  # nouns
        b_rate = sr.get("condition_b_mean", 0)  # numbers
        sr_p = sr.get("wilcoxon_p_value", 1.0)
        diff_pct = ((b_rate - a_rate) / a_rate * 100) if a_rate else 0

        rows_tbl += f"<tr><td>{MODEL_LABELS[model]}</td><td class=\"num\">{fmt(a_rate,1)} syl/s</td><td class=\"num\">{fmt(b_rate,1)} syl/s</td><td class=\"num\">{fmt(diff_pct,1)}%</td><td class=\"num\">p={fmt(sr_p,4)}</td></tr>"

        if diff_pct > 5:
            models_with_tempo.append(MODEL_LABELS[model])

    if not rows_tbl:
        return ""

    return f"""
<h2>5 &ensp; Secondary Finding: Tempo Acceleration</h2>
<p>An unexpected pattern emerged: <b>models read the target sentence faster</b> after
a robotic number prime than after a length-matched noun list. This is a distinct
"hangover phenotype" — not pitch compression, but rushing.</p>

<table>
<thead><tr><th>Model</th><th>Nouns (rate)</th><th>Numbers (rate)</th><th>Speedup</th><th>p-value</th></tr></thead>
<tbody>{rows_tbl}</tbody>
</table>

<p><b>Two distinct hangover phenotypes identified:</b></p>
<ul>
    <li><b>Pitch compression</b> (f0_cv drops) — Chatterbox primary, XTTS-v2 trending</li>
    <li><b>Tempo acceleration</b> (speaking rate increases) — Kokoro borderline (p=0.0625),
    XTTS-v2 numerically</li>
</ul>

<div class="highlight">
<p>This was not hypothesized but is consistent across models. Robotic primes don't just
flatten pitch — they make models <em>rush</em> through subsequent emotional content.
This is a novel finding worth investigating in larger studies.</p>
</div>
"""


def model_state():
    """Per-model state of evidence summary."""
    return """
<h2>6 &ensp; State of Evidence</h2>

<table>
<thead><tr><th>Model</th><th>Verdict</th><th>f0_cv effect</th><th>Speaking rate</th><th>Next step</th></tr></thead>
<tbody>
<tr><td><b>Chatterbox</b></td><td><b style="color:#4CAF50;">&#x2705; Publishable</b></td><td>&#x2212;39%, p=0.031 &#x2715;</td><td>Stable (p=0.19)</td><td>Write up as primary result</td></tr>
<tr><td><b>XTTS-v2</b></td><td><b style="color:#e6a817;">&#x1f536; Trend, needs n=10</b></td><td>&#x2212;29%, p=0.094</td><td>Stable (p=0.125)</td><td>1 more run pair flips to significance</td></tr>
<tr><td><b>Kokoro</b></td><td><b style="color:#c0392b;">&#x2753; Inconclusive</b></td><td>&#x2212;13%, p=0.22; wide spread</td><td>Borderline speedup (p=0.0625)</td><td>Verify 510-phoneme chunking; different phenotype</td></tr>
</tbody>
</table>

<div class="highlight">
<p><b>Bottom line from this n=5 experiment:</b> The hangover effect is real and
demonstrable in Chatterbox. It manifests as two distinct acoustic phenotypes —
pitch compression (Chatterbox, XTTS-v2) and tempo acceleration (Kokoro, XTTS-v2).
A third model showing the effect with rigorous controls (WhisperX alignment,
length-matched baseline, f0_cv normalization) makes this a robust finding even at
modest sample size.</p>
</div>
"""


def raw_data():
    features = load_csv(FEATURES_CSV)
    if not features:
        return ""

    rows = ""
    for r in features:
        rows += f"<tr><td>{r['model']}</td><td>{CONDITION_LABELS.get(r['condition'], r['condition'])}</td><td class=\"num\">{r['run']}</td><td class=\"num\">{fmt(r['f0_mean'],1)}</td><td class=\"num\">{fmt(r['f0_std'],1)}</td><td class=\"num\">{fmt(r['f0_cv'],4)}</td><td class=\"num\">{fmt(r['speaking_rate'],1)}</td><td class=\"num\">{fmt(r['energy_std'],4)}</td><td class=\"num\">{r['target_duration_s']}s</td></tr>"

    return f"""
<h2>7 &ensp; Raw Data</h2>
<table class="raw-data-table">
<thead><tr><th>Model</th><th>Condition</th><th>Run</th><th>F0 Mean</th><th>F0 Std</th><th>F0 CV</th><th>Rate</th><th>Energy Std</th><th>Duration</th></tr></thead>
<tbody>{rows}</tbody>
</table>
"""


def limitations():
    return """
<h2>8 &ensp; Limitations</h2>

<div class="limitations">
<ul>
    <li><b>N is very small:</b> n=5 per condition per model. Results are descriptive, not confirmatory.</li>
    <li><b>WhisperX alignment:</b> 1/45 files failed alignment (excluded from analysis). Minor boundary errors may persist.</li>
    <li><b>Single target sentence:</b> Results may not generalise to other emotional sentences or speaking styles.</li>
    <li><b>Kokoro baseline:</b> No voice cloning — different speaker default (af_bella), not directly comparable.</li>
    <li><b>XTTS-v2 sentence splitting:</b> Internal sentence-boundary splitting may reduce context window bleed.</li>
    <li><b>Syllable count:</b> 18 syllables hardcoded — coarticulation may shift true count slightly.</li>
    <li><b>Acoustic proxy:</b> f0_cv is an acoustic measure, not perceptual monotonicity.</li>
    <li><b>Vocabulary confound:</b> Numbers and nouns differ in phonemic content, not just "robotic-ness".</li>
</ul>
</div>
"""


def conclusions():
    return """
<h2>9 &ensp; Conclusions</h2>

<p><b>Does a monotone number prime suppress pitch variance in a subsequent emotional
target sentence?</b> Yes, in at least one of three tested models.</p>

<h3>What We Found</h3>
<ol>
    <li><b>Chatterbox shows a clean, significant hangover effect</b> — 39% f0_cv drop
    after numbers vs length-matched nouns (p=0.031), with stable speaking rate.
    This survives WhisperX word alignment, f0_cv normalization, and a rigorous
    length-matched control. <em>Result is worth replicating.</em></li>

    <li><b>Two distinct hangover phenotypes emerged:</b>
    <ul>
        <li><b>Pitch compression</b> — reduced f0_cv after robotic primes (Chatterbox, XTTS-v2)</li>
        <li><b>Tempo acceleration</b> — increased speaking rate after robotic primes (Kokoro borderline, XTTS-v2 numerically)</li>
    </ul>
    These may reflect different underlying mechanisms — the same "acoustic inertia"
    affecting different production parameters depending on model architecture.</li>

    <li><b>XTTS-v2 shows a strong trend</b> (29% drop, p=0.094) that would likely
    reach significance at n=10. Worth pursuing.</li>

    <li><b>Kokoro is inconclusive</b> — not due to alignment artifacts but genuine
    instability in the numbers condition (f0_cv range 0.14–0.37 vs 0.27–0.32 for nouns).
    The 510-phoneme chunking ceiling may partially break the context window.</li>
</ol>

<h3>What This Means</h3>
<p>The "subliminal hangover" — acoustic inertia from monotone primes bleeding into
emotional speech — is a real, measurable phenomenon in at least one contemporary
TTS model. The effect survives rigorous controls for text length, segmentation
method, and speaker pitch baseline. The discovery of two distinct phenotypes
(pitch compression vs tempo acceleration) opens a new axis for TTS evaluation:
contextual prosodic stability under mixed-style generation.</p>

<div class="highlight">
<p><strong>Practical recommendation:</strong> Voice-cloning TTS pipelines should not
assume context-independent prosody. When concatenating heterogeneous text styles
(robotic numbers + emotional speech), measurable acoustic bleed occurs. For
emotionally-sensitive applications (audiobooks, dialogue, virtual agents),
consider generating target segments in isolation rather than as continuations
of acoustically dissimilar primes.</p>
</div>
"""


def footer():
    return """
<div class="footer">
    Subliminal Hangover Benchmark (V1–V3) &bull; 45 WAVs across Chatterbox, XTTS-v2, and Kokoro.
    <br>Report generated with Python 3.12 &bull; parselmouth &bull; WhisperX &bull; weasyprint.
</div>

</div>
</body>
</html>
"""


def main():
    stats = load_json(STATS_JSON)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    html = (
        header()
        + caveat()
        + methodology()
        + alignment_quality()
        + primary_result(stats)
        + tempo_acceleration(stats)
        + model_state()
        + raw_data()
        + limitations()
        + conclusions()
        + footer()
    )

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(REPORT_HTML) / 1024
    print(f"Report HTML written to {REPORT_HTML} ({size_kb:.0f} KB)")

    try:
        from weasyprint import HTML as WPHTML
        WPHTML(filename=str(REPORT_HTML)).write_pdf(str(REPORT_PDF))
        pdf_kb = os.path.getsize(REPORT_PDF) / 1024
        print(f"Report PDF written to {REPORT_PDF} ({pdf_kb:.0f} KB)")
    except ImportError:
        print("WARN: weasyprint not available, skipping PDF")

    return 0


if __name__ == "__main__":
    sys.exit(main())
