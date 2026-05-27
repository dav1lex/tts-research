#!/usr/bin/env python3
"""Generate HTML report for the subliminal hangover benchmark.

Embeds the figure and results from stats.json.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent

STATS_JSON = PROJECT / "results" / "stats.json"
GATE_JSON = PROJECT / "results" / "gate_check.json"
FIG_PATH = PROJECT / "results" / "f0_variance_hangover.png"
REPORT_HTML = PROJECT / "results" / "report.html"
FEATURES_CSV = PROJECT / "features" / "features.csv"

MODEL_LABELS = {
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "kokoro": "Kokoro (no-adaptation baseline)",
}


def load_json(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_features_csv():
    import csv
    with open(FEATURES_CSV) as f:
        return list(csv.DictReader(f))


def fig_to_b64(path: Path) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_report():
    stats = load_json(STATS_JSON)
    gate = load_json(GATE_JSON)
    features = load_features_csv()

    paired = stats.get("paired_tests", {})
    descriptive = stats.get("descriptive", {})

    # ── Summary table ──────────────────────────────────────────────────
    summary_rows = ""
    for model in ["chatterbox", "xtts", "kokoro"]:
        if model not in paired:
            continue
        p = paired[model]
        label = MODEL_LABELS.get(model, model)
        sig_star = " *" if p.get("significance") == "significant" else ""
        summary_rows += f"""
        <tr>
            <td>{label}</td>
            <td>{p['control_mean']:.2f}</td>
            <td>{p['subliminal_mean']:.2f}</td>
            <td>{p['mean_difference_pct']:+.1f}%</td>
            <td>W={p['wilcoxon_statistic']:.1f}, p={p['wilcoxon_p_value']:.4f}{sig_star}</td>
        </tr>"""

    # ── Per-model detail tables ──────────────────────────────────────
    detail_html = ""
    for model in ["chatterbox", "xtts", "kokoro"]:
        if model not in descriptive:
            continue
        label = MODEL_LABELS.get(model, model)
        md = descriptive[model]
        detail_html += f"<h3>{label}</h3>"
        detail_html += """
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; margin-bottom:20px;">
        <tr><th>Metric</th><th>Condition</th><th>N</th><th>Mean</th><th>SD</th><th>SE</th><th>Min</th><th>Max</th></tr>
        """
        for metric in md:
            for cond in ["control", "subliminal"]:
                if cond not in md[metric]:
                    continue
                s = md[metric][cond]
                detail_html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{cond}</td>
                    <td>{s['n']}</td>
                    <td>{s['mean']}</td>
                    <td>{s['std']}</td>
                    <td>{s['se']}</td>
                    <td>{s['min']}</td>
                    <td>{s['max']}</td>
                </tr>"""
        detail_html += "</table>"

    # ── Raw data table ────────────────────────────────────────────────
    raw_rows = ""
    for r in features:
        raw_rows += f"""
        <tr>
            <td>{r['model']}</td>
            <td>{r['condition']}</td>
            <td>{r['run']}</td>
            <td>{r['f0_mean']}</td>
            <td>{r['f0_std']}</td>
            <td>{r['energy_std']}</td>
            <td>{r['target_duration_s']}</td>
        </tr>"""

    # ── Gate info ────────────────────────────────────────────────────
    gate_str = json.dumps(gate.get("summary", {}), indent=2) if gate else "N/A"

    # ── Conclusion builder ───────────────────────────────────────────
    conclusions = []
    for model in ["chatterbox", "xtts", "kokoro"]:
        if model not in paired:
            continue
        p = paired[model]
        label = MODEL_LABELS.get(model, model)
        diff_pct = p["mean_difference_pct"]
        if diff_pct < 0:
            conclusions.append(
                f"<b>{label}</b> showed a <b>{abs(diff_pct):.1f}% increase</b> in f0_std "
                f"(subliminal > control), opposite of the hangover hypothesis."
            )
        elif diff_pct > 0:
            conclusions.append(
                f"<b>{label}</b> showed a <b>{diff_pct:.1f}% drop</b> in f0_std "
                f"(subliminal < control)"
            )
            if p.get("significance") == "significant":
                conclusions[-1] += ", which was <b>statistically significant</b> (p&lt;0.05)."
            else:
                conclusions[-1] += ", but was <b>not statistically significant</b>."
        else:
            conclusions.append(
                f"<b>{label}</b> showed no change in f0_std."
            )

    conclusion_text = "<br>".join(conclusions)

    # Worst hangover model
    worst_model = None
    worst_diff = -999
    for model in ["chatterbox", "xtts", "kokoro"]:
        if model in paired:
            dp = paired[model].get("mean_difference_pct", 0)
            if dp > worst_diff:
                worst_diff = dp
                worst_model = MODEL_LABELS.get(model, model)

    # ── Figure ───────────────────────────────────────────────────────
    fig_b64 = fig_to_b64(FIG_PATH) if FIG_PATH.exists() else ""
    img_tag = f'<img src="data:image/png;base64,{fig_b64}" style="max-width:800px; width:100%;">' if fig_b64 else "<p>Figure not available.</p>"

    # ── HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Subliminal Hangover Benchmark — Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.6; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
h2 {{ color: #16213e; margin-top: 32px; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th {{ background: #1a1a2e; color: white; padding: 8px 12px; text-align: left; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
tr:hover td {{ background: #f5f5f5; }}
.fig-container {{ text-align: center; margin: 24px 0; }}
.gate-pass {{ color: #2e7d32; font-weight: bold; }}
.gate-fail {{ color: #c62828; font-weight: bold; }}
.sig {{ color: #2e7d32; }}
.ns {{ color: #888; }}
pre {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }}
</style>
</head>
<body>

<h1>🔬 Subliminal Hangover Benchmark</h1>
<p><em>Does robotic-number context bleed into emotional target prosody?</em></p>

<h2>Hypothesis</h2>
<p>
If forced context window processing carries acoustic inertia, then a monotone/robotic prime
(reading disconnected numbers) should suppress pitch variance (F0 StdDev) in a subsequent
highly emotional target sentence, compared to a neutral-but-natural prime (nature scene).
</p>

<h2>Methodology</h2>
<ol>
    <li><b>Target sentence (constant):</b> <code>"I can't believe you just did that! Get away from me right now!"</code></li>
    <li><b>Control prime:</b> <code>"The sun was shining brightly on the beautiful, warm meadow today."</code></li>
    <li><b>Subliminal prime:</b> Randomized number sequences (14 numbers per run, 5 unique shuffles).</li>
    <li><b>Generation:</b> Prime + Target concatenated into a single string → single audio generation (same context window).</li>
    <li><b>Segmentation:</b> VAD-based pause detection finds prime-target boundary; features extracted from target segment only.</li>
    <li><b>Primary metric:</b> F0 Standard Deviation (f0_std) — lower = more monotone / robotic.</li>
    <li><b>Test:</b> Wilcoxon signed-rank (one-sided, control > subliminal).</li>
    <li><b>Models:</b> Chatterbox (voice-cloned), XTTS-v2 (voice-cloned), Kokoro (no-adaptation baseline).</li>
    <li><b>Reference:</b> VCTK p229_002 (female, Southern England) for voice-cloning models.</li>
</ol>

<h2>Gate Check</h2>
<pre>{gate_str}</pre>
<p class="{'gate-pass' if gate.get('gate_passed') else 'gate-fail'}">
    Gate: {'PASSED' if gate.get('gate_passed') else 'PARTIAL FAIL'} ({gate.get('passed_rows', 0)}/{gate.get('total_rows', 0)} rows viable)
</p>

<h2>Results: F0 StdDev (Hz)</h2>

<table>
<tr><th>Model</th><th>Control</th><th>Subliminal</th><th>Change</th><th>Test</th></tr>
{summary_rows}
</table>

<div class="fig-container">
    {img_tag}
    <p><em>Figure: F0 Standard Deviation by Model and Condition (error bars = SE)</em></p>
</div>

<h2>Per-Model Descriptive Statistics</h2>
{detail_html}

<h2>Raw Data</h2>
<table>
<tr><th>Model</th><th>Condition</th><th>Run</th><th>F0 Mean</th><th>F0 Std</th><th>Energy Std</th><th>Duration (s)</th></tr>
{raw_rows}
</table>

<h2>Interpretation</h2>
<p>
{conclusion_text}
</p>
<p>
<b>Worst "hangover" candidate:</b> {worst_model} showed the largest f0_std suppression.
</p>

<h2>Caveats</h2>
<ul>
    <li><b>N is very small:</b> n=5 per condition per model. Results are descriptive, not conclusive.</li>
    <li><b>VAD boundary detection:</b> RMS-based pause detection is a proxy; boundary placement adds variance.</li>
    <li><b>No forced alignment:</b> Target segment boundaries are estimated, not anchored to phonemes.</li>
    <li><b>Kokoro baseline:</b> No voice cloning — different speaker default (af_bella), not directly comparable.</li>
    <li><b>XTTS-v2 sentence splitting:</b> XTTS internally splits text on sentence boundaries and processes independently — this may reduce contextual bleed compared to single-segment generation.</li>
    <li><b>Acoustic proxy:</b> F0 StdDev is an acoustic measure, not perceptual monotonicity.</li>
</ul>

<hr>
<p style="color:#888; font-size:0.85em;">
Generated by <code>make_report.py</code> | TTS Research Benchmark Suite
</p>

</body>
</html>"""

    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_HTML, "w") as f:
        f.write(html)
    print(f"Report saved to {REPORT_HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(build_report())
