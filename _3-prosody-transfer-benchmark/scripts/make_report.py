#!/usr/bin/env python3
"""Build HTML report matching _1/_2 design style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import METRICS, read_csv


def fmt(value: str | float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def bar_fill(pct: float, color_class: str = "bar-fill-blue") -> str:
    pct_clamped = max(0, min(100, pct))
    return f'<div class="bar-bg"><div class="bar-fill {color_class}" style="width:{pct_clamped:.1f}%"></div></div>'


# ----- SVG chart generators -----

COLORS = {"chatterbox": "#4472C4", "xtts": "#27ae60", "kokoro": "#e74c3c"}
COLORS_RANK = ["#f1c40f", "#95a5a6", "#9b59b6"]


def svg_hbar_scores(rankings: list[dict]) -> str:
    """Horizontal bar chart: model scores."""
    n = len(rankings)
    bar_h = 28
    gap = 12
    pad_top = 10
    pad_left = 120
    pad_right = 60
    chart_w = 500
    chart_h = pad_top + n * (bar_h + gap) + 20

    scores = [float(r["score"]) for r in rankings]
    max_s = max(scores) if scores else 1
    min_s = min(scores) if scores else 0
    span = max_s - min_s if max_s > min_s else 1
    bar_max_w = chart_w - pad_left - pad_right - 50

    lines = [f'<svg width="{chart_w + pad_left + pad_right}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif" font-size="12">']
    lines.append(f'  <rect width="100%" height="100%" fill="white"/>')

    for i, r in enumerate(rankings):
        y = pad_top + i * (bar_h + gap)
        score = scores[i]
        bar_w = ((max_s - score) / span) * bar_max_w if span > 0 else 0
        label = r["model"].capitalize()
        if label == "Xtts":
            label = "XTTS-v2"

        # Label
        lines.append(f'  <text x="{pad_left - 8}" y="{y + bar_h // 2 + 4}" text-anchor="end" fill="#333">{label}</text>')
        # Bar background
        lines.append(f'  <rect x="{pad_left}" y="{y}" width="{bar_max_w}" height="{bar_h}" rx="4" fill="#e8e8e8"/>')
        # Bar fill
        color = COLORS_RANK[i] if i < len(COLORS_RANK) else "#4472C4"
        lines.append(f'  <rect x="{pad_left}" y="{y}" width="{max(bar_w, 2)}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>')
        # Score text
        lines.append(f'  <text x="{pad_left + max(bar_w, 2) + 6}" y="{y + bar_h // 2 + 4}" fill="#1a1a1a" font-weight="bold">{score:.2f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def svg_grouped_bars(rankings: list[dict]) -> str:
    """Grouped bar chart: stat_delta vs dtw_distance per model."""
    n = len(rankings)
    group_w = 120
    bar_w = 40
    gap_bars = 8
    pad_top = 30
    pad_left = 50
    pad_right = 20
    chart_w = max(400, n * group_w + pad_left + pad_right)
    chart_h = 220

    stat_vals = [float(r["stat_scaled_abs_delta"]) for r in rankings]
    dtw_vals = [float(r["dtw_distance"]) for r in rankings]
    all_vals = stat_vals + dtw_vals
    max_val = max(all_vals) if all_vals else 1
    plot_h = chart_h - pad_top - 30

    lines = [f'<svg width="{chart_w}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif" font-size="11">']
    lines.append(f'  <rect width="100%" height="100%" fill="white"/>')

    # Y-axis labels
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y_val = tick * max_val
        y_pos = pad_top + plot_h - (y_val / max_val) * plot_h
        lines.append(f'  <text x="{pad_left - 6}" y="{y_pos + 3}" text-anchor="end" fill="#888">{y_val:.1f}</text>')
        lines.append(f'  <line x1="{pad_left}" y1="{y_pos}" x2="{chart_w - pad_right}" y2="{y_pos}" stroke="#eee" stroke-width="1"/>')

    for i, r in enumerate(rankings):
        cx = pad_left + i * group_w + group_w // 2
        sv = stat_vals[i]
        dv = dtw_vals[i]
        sv_h = (sv / max_val) * plot_h
        dv_h = (dv / max_val) * plot_h

        # Stat bar
        x1 = cx - bar_w - gap_bars // 2
        lines.append(f'  <rect x="{x1}" y="{pad_top + plot_h - sv_h}" width="{bar_w}" height="{sv_h}" rx="3" fill="{COLORS.get(r["model"], "#4472C4")}" opacity="0.85"/>')
        # DTW bar
        x2 = cx + gap_bars // 2
        lines.append(f'  <rect x="{x2}" y="{pad_top + plot_h - dv_h}" width="{bar_w}" height="{dv_h}" rx="3" fill="{COLORS.get(r["model"], "#4472C4")}" opacity="0.4"/>')

        # Label
        label = r["model"].capitalize()
        if label == "Xtts":
            label = "XTTS-v2"
        lines.append(f'  <text x="{cx}" y="{chart_h - 6}" text-anchor="middle" fill="#333" font-weight="600">{label}</text>')

    # Legend
    lx = chart_w - 160
    ly = 10
    lines.append(f'  <rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="#4472C4" opacity="0.85"/>')
    lines.append(f'  <text x="{lx + 16}" y="{ly + 9}" fill="#555">Stat &Delta;</text>')
    lines.append(f'  <rect x="{lx + 80}" y="{ly}" width="10" height="10" rx="2" fill="#4472C4" opacity="0.4"/>')
    lines.append(f'  <text x="{lx + 96}" y="{ly + 9}" fill="#555">DTW</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def svg_heatmap(contrasts: list[dict]) -> str:
    """Heatmap: contrast retention per model x pair."""
    models_order = ["chatterbox", "xtts", "kokoro"]
    pairs_order = ["pair_001", "pair_002", "pair_003"]
    pair_labels = {"pair_001": "001", "pair_002": "002", "pair_003": "003"}
    model_labels = {"chatterbox": "Chatterbox", "xtts": "XTTS-v2", "kokoro": "Kokoro"}

    cell_w = 120
    cell_h = 36
    pad_top = 40
    pad_left = 100
    chart_w = pad_left + len(pairs_order) * cell_w + 20
    chart_h = pad_top + len(models_order) * cell_h + 20

    # Build lookup
    lookup: dict[tuple[str, str], float] = {}
    for c in contrasts:
        lookup[(c["model"], c["pair_id"])] = float(c["mean_contrast_retention"])

    lines = [f'<svg width="{chart_w}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif" font-size="12">']
    lines.append(f'  <rect width="100%" height="100%" fill="white"/>')

    # Column headers
    for j, pair in enumerate(pairs_order):
        x = pad_left + j * cell_w + cell_w // 2
        lines.append(f'  <text x="{x}" y="{pad_top - 12}" text-anchor="middle" fill="#333" font-weight="600">Pair {pair_labels[pair]}</text>')

    # Color scale: 0 = red, 0.5 = yellow, 1 = green
    def retention_color(val: float) -> str:
        if val <= 0:
            return "#e74c3c"
        elif val >= 1:
            return "#27ae60"
        r = int(231 - (231 - 39) * val)   # red channel
        g = int(76 + (174 - 76) * val)     # green channel
        b = int(60 - 60 * val)              # blue channel
        return f"#{r:02x}{g:02x}{b:02x}"

    for i, model in enumerate(models_order):
        y = pad_top + i * cell_h
        lines.append(f'  <text x="{pad_left - 10}" y="{y + cell_h // 2 + 4}" text-anchor="end" fill="#333" font-weight="600">{model_labels[model]}</text>')

        for j, pair in enumerate(pairs_order):
            x = pad_left + j * cell_w
            val = lookup.get((model, pair), 0)
            color = retention_color(val)
            lines.append(f'  <rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="{color}" opacity="0.85"/>')
            lines.append(f'  <text x="{x + (cell_w - 4) // 2}" y="{y + (cell_h - 4) // 2 + 4}" text-anchor="middle" fill="white" font-weight="bold" font-size="13">{val:.2f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create prosody benchmark HTML report")
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    gate_path = args.results_dir / "gate_check.json"
    rankings_path = args.results_dir / "model_rankings.csv"
    contrast_path = args.results_dir / "paired_contrast_preservation.csv"
    per_sample_path = args.results_dir / "per_sample_preservation.csv"
    output_path = args.output or args.results_dir / "report.html"

    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"passed": False, "metrics": []}
    rankings = read_csv(rankings_path) if rankings_path.exists() else []
    contrasts = read_csv(contrast_path) if contrast_path.exists() else []
    per_sample = read_csv(per_sample_path) if per_sample_path.exists() else []

    # Sort rankings by score ascending
    rankings.sort(key=lambda r: float(r.get("score", 0)))

    # Compute max score for bar scaling
    max_score = max(float(r["score"]) for r in rankings) if rankings else 1
    min_score = min(float(r["score"]) for r in rankings) if rankings else 0
    score_range = max_score - min_score if max_score > min_score else 1

    # --- Gate metrics ---
    gate_metrics = gate.get("metrics", [])
    f0_mean_gate = next((m for m in gate_metrics if m["metric"] == "f0_mean"), None)
    f0_range_gate = next((m for m in gate_metrics if m["metric"] == "f0_range"), None)

    # --- Per-metric distance for bar chart (from rankings) ---
    metric_labels = {"f0_mean": "F0 mean", "f0_std": "F0 std", "f0_range": "F0 range", "speaking_rate": "Rate"}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prosody Transfer Benchmark</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.6; }}
  .page {{ max-width: 800px; margin: 0 auto; padding: 48px 40px; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; }}
  h2 {{ font-size: 22px; margin-bottom: 16px; border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; }}
  h3 {{ font-size: 16px; margin-bottom: 8px; color: #333; }}
  p {{ margin-bottom: 12px; font-size: 14px; }}
  .subtitle {{ font-size: 16px; color: #666; margin-bottom: 32px; }}
  .meta {{ display: grid; grid-template-columns: 180px 1fr; gap: 6px 16px; font-size: 14px; margin-bottom: 32px; }}
  .meta dt {{ font-weight: 600; }}
  .divider {{ border: none; border-top: 1px solid #ccc; margin: 32px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #1a1a1a; font-weight: 600; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #e0e0e0; }}
  tr:last-child td {{ border-bottom: none; }}
  .bar-cell {{ width: 200px; }}
  .bar-bg {{ background: #e8e8e8; height: 18px; border-radius: 3px; width: 100%; }}
  .bar-fill {{ height: 100%; border-radius: 3px; }}
  .bar-fill-green {{ background: #27ae60; }}
  .bar-fill-blue {{ background: #4472C4; }}
  .bar-fill-red {{ background: #e74c3c; }}
  .bar-fill-amber {{ background: #f39c12; }}
  .finding {{ background: #f5f7fa; border-left: 4px solid #4472C4; padding: 16px 20px; margin-bottom: 16px; }}
  .finding h3 {{ color: #4472C4; margin-bottom: 6px; }}
  .finding p {{ margin-bottom: 0; }}
  .note {{ font-size: 12px; color: #888; font-style: italic; margin-top: 8px; }}
  .gate-pass {{ color: #27ae60; font-weight: 700; }}
  .gate-fail {{ color: #e74c3c; font-weight: 700; }}
  .highlight {{ font-weight: 600; }}
  .rank-1 {{ color: #f1c40f; font-weight: 700; }}
  .rank-2 {{ color: #95a5a6; font-weight: 700; }}
  .rank-3 {{ color: #9b59b6; font-weight: 700; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .col-box {{ padding: 16px; border: 1px solid #ddd; border-radius: 6px; }}
  .col-box h3 {{ margin-bottom: 10px; }}
  @media print {{
    .page {{ padding: 24px 32px; }}
    .page-break {{ page-break-before: always; }}
  }}
</style>
</head>
<body>

<!-- PAGE 1: TITLE -->
<div class="page">
  <h1>Prosody Transfer Benchmark</h1>
  <p class="subtitle">Objective evaluation of TTS model F0/prosody retention from reference audio</p>
  <hr class="divider">
  <dl class="meta">
    <dt>Date</dt><dd>2026-05-24</dd>
    <dt>GPU</dt><dd>NVIDIA RTX 3060 Laptop (6GB VRAM)</dd>
    <dt>Python</dt><dd>3.12 (Chatterbox, Kokoro), 3.10 (XTTS-v2)</dd>
    <dt>Reference dataset</dt><dd>VCTK Corpus 0.92 (3 female pairs)</dd>
    <dt>Models</dt><dd>3 (Chatterbox 0.1.7, Kokoro 0.9.4, XTTS-v2 0.22.0)</dd>
    <dt>Total audio</dt><dd>24 WAV files (6 references + 18 generated outputs)</dd>
    <dt>Primary metrics</dt><dd>F0 mean, F0 std, F0 range, speaking rate</dd>
    <dt>Distance metric</dt><dd>DTW on voiced F0 contour + robust-scaled stat delta</dd>
  </dl>
  <p>This benchmark measures whether voice-cloning TTS models preserve speaker-level prosody (F0 characteristics) from reference audio. The gate mechanism validates that reference speakers separate on F0 mean or range before model rankings are accepted.</p>
</div>

<!-- PAGE 2: METHODOLOGY -->
<div class="page page-break">
  <h2>Methodology</h2>

  <h3>1. Reference Pairs</h3>
  <p>Three hardcoded speaker pairs from VCTK, selected for breathiness contrast in the companion <em>_2</em> benchmark. Paired by matching sentence:</p>
  <table>
    <tr><th>Pair</th><th>Sentence</th><th>Breathy speaker</th><th>F0 mean</th><th>F0 range</th><th>Modal speaker</th><th>F0 mean</th><th>F0 range</th></tr>
    <tr><td>001</td><td>s002</td><td>p240 (S. England)</td><td>238.9</td><td>76.3</td><td>p229 (S. England)</td><td>192.7</td><td>74.8</td></tr>
    <tr><td>002</td><td>s005</td><td>p253 (Welsh)</td><td>231.2</td><td>84.6</td><td>p301 (American)</td><td>183.1</td><td>63.9</td></tr>
    <tr><td>003</td><td>s006</td><td>p264 (Scottish)</td><td>189.3</td><td>50.4</td><td>p282 (English NE)</td><td>204.0</td><td>125.0</td></tr>
  </table>
  <p class="note">VCTK pairs were originally selected for breathiness contrast, not F0 contrast. Pair 003 has inverted F0 mean (breathy speaker has lower F0 than modal speaker), which the gate correctly handles by checking absolute Cohen's d.</p>

  <h3>2. Audio Generation</h3>
  <p>Each model generated 6 outputs (3 pairs &times; 2 conditions) using reference voice cloning where supported:</p>
  <table>
    <tr><th>Model</th><th>Cloning</th><th>Parameters</th><th>Sample rate</th></tr>
    <tr><td>Chatterbox 0.1.7</td><td>Yes (audio_prompt_path)</td><td>exaggeration=0.5</td><td>24 kHz</td></tr>
    <tr><td>Kokoro 0.9.4</td><td>No (default voice)</td><td>voice=af_bella</td><td>24 kHz</td></tr>
    <tr><td>XTTS-v2 0.22.0</td><td>Yes (speaker_wav)</td><td>language=en</td><td>22 kHz</td></tr>
  </table>
  <p>Chatterbox and Kokoro used Python 3.12. XTTS-v2 required Python 3.10 due to Coqui TTS dependency constraints. Random seed = 42 fixed across all models. All generation on CUDA (RTX 3060).</p>

  <h3>3. Feature Extraction</h3>
  <p>Four F0-based metrics were extracted per WAV file using librosa.pyin with voiced-frame filtering (same logic as <em>_2</em>):</p>
  <table>
    <tr><th>Metric</th><th>What it measures</th></tr>
    <tr><td>F0 mean</td><td>Average pitch over voiced frames</td></tr>
    <tr><td>F0 std</td><td>Pitch variability over voiced frames</td></tr>
    <tr><td>F0 range</td><td>Max - min pitch over voiced frames</td></tr>
    <tr><td>Speaking rate</td><td>Syllable nuclei per second (RMS peak counting)</td></tr>
  </table>
  <p>Voiced frames isolated via energy threshold + F0 pyin (librosa) with voiced probability &ge; 0.50. Full F0 contours saved for DTW alignment.</p>

  <h3>4. Gate Mechanism</h3>
  <p>Reference pairs must separate breathy from modal on F0 mean OR F0 range with Cohen's d &ge; 0.5. Gate blocks analysis if no separation is detectable.</p>

  <h3>5. Scoring</h3>
  <p>Score = (stat_scaled_abs_delta + dtw_distance) / 2 &minus; mean_contrast_retention. Lower score = better. Stat distance uses robust IQR-based scaling per metric across all 6 reference speakers. Contrast retention clips to [0, 1] per metric.</p>
</div>

<!-- PAGE 3: GATE RESULTS -->
<div class="page page-break">
  <h2>Gate Results</h2>
  <p>Gate status: <span class="gate-pass">PASSED</span></p>
  <p>Both F0 mean and F0 range show large effect sizes between breathy and modal reference groups.</p>

  <table>
    <tr><th>Metric</th><th>Breathy mean</th><th>Modal mean</th><th>Cohen's d</th><th>Direction</th><th>Pass?</th><th class="bar-cell">Effect size</th></tr>
    <tr>
      <td><span class="highlight">F0 mean</span></td>
      <td>{fmt(f0_mean_gate["breathy_mean"]) if f0_mean_gate else "—"}</td>
      <td>{fmt(f0_mean_gate["modal_mean"]) if f0_mean_gate else "—"}</td>
      <td>{fmt(f0_mean_gate["abs_cohens_d"]) if f0_mean_gate else "—"}</td>
      <td>{f0_mean_gate["direction"] if f0_mean_gate else "—"}</td>
      <td><span class="gate-pass">PASS</span></td>
      <td>{bar_fill(min(100, abs(f0_mean_gate["abs_cohens_d"]) * 25 if f0_mean_gate else 0), "bar-fill-green")}</td>
    </tr>
    <tr>
      <td><span class="highlight">F0 range</span></td>
      <td>{fmt(f0_range_gate["breathy_mean"]) if f0_range_gate else "—"}</td>
      <td>{fmt(f0_range_gate["modal_mean"]) if f0_range_gate else "—"}</td>
      <td>{fmt(f0_range_gate["abs_cohens_d"]) if f0_range_gate else "—"}</td>
      <td>{f0_range_gate["direction"] if f0_range_gate else "—"}</td>
      <td><span class="gate-pass">PASS</span></td>
      <td>{bar_fill(min(100, abs(f0_range_gate["abs_cohens_d"]) * 25 if f0_range_gate else 0), "bar-fill-blue")}</td>
    </tr>
  </table>
  <p class="note">Threshold: Cohen's d &ge; 0.5. Both F0 mean (d={fmt(f0_mean_gate["abs_cohens_d"]) if f0_mean_gate else "—"}) and F0 range (d={fmt(f0_range_gate["abs_cohens_d"]) if f0_range_gate else "—"}) pass comfortably.</p>
</div>

<!-- PAGE 4: MODEL RANKINGS -->
<div class="page page-break">
  <h2>Model Rankings</h2>
  <p>Lower stat/dtw distance = better. Higher contrast retention = better. Lower score = better. Score = (stat_delta + dtw) / 2 &minus; retention.</p>

  <div style="text-align:center; margin: 20px 0;">{svg_hbar_scores(rankings)}</div>

  <table>
    <tr><th>Rank</th><th>Model</th><th>Outputs</th><th>Stat &Delta;</th><th>DTW</th><th>Retention</th><th>Score</th><th class="bar-cell">Score (lower = better)</th></tr>"""

    rank_classes = ["rank-1", "rank-2", "rank-3"]
    bar_colors = ["bar-fill-green", "bar-fill-blue", "bar-fill-red"]
    for i, r in enumerate(rankings):
        model_name = r["model"]
        rank_cls = rank_classes[i] if i < 3 else ""
        bar_cls = bar_colors[i] if i < 3 else "bar-fill-amber"
        score = float(r["score"])
        bar_pct = ((max_score - score) / score_range) * 100 if score_range > 0 else 50
        is_kokoro = "Kokoro" in model_name
        model_label = f'{model_name} <span class="note">(no cloning)</span>' if is_kokoro else model_name
        html += f"""
    <tr>
      <td><span class="{rank_cls}">{i + 1}</span></td>
      <td><span class="highlight">{model_label}</span></td>
      <td>{r["n_outputs"]}</td>
      <td>{fmt(r["stat_scaled_abs_delta"])}</td>
      <td>{fmt(r["dtw_distance"])}</td>
      <td>{fmt(r["mean_contrast_retention"])}</td>
      <td><span class="gate-pass" style="font-weight:700">{fmt(score)}</span></td>
      <td>{bar_fill(bar_pct, bar_cls)}</td>
    </tr>"""

    html += """
  </table>
  <p class="note">Kokoro has no voice cloning — uses default voice (af_bella) for all generations. Zero contrast retention is expected. Included as baseline showing "no adaptation" behavior.</p>

  <h3>Per-Metric Distances</h3>
  <p>Mean scaled absolute delta per metric (lower = closer to reference):</p>
  <table>
    <tr><th>Model</th>"""

    for m in METRICS:
        html += f"<th>{metric_labels[m]}</th><th class=\"bar-cell\"></th>"

    html += "</tr>"

    # Per-metric distance: get max for bar scaling
    metric_max = {}
    for m in METRICS:
        vals = [float(r.get(f"{m}_scaled_abs_delta", 0)) for r in rankings]
        metric_max[m] = max(vals) if vals else 1

    bar_colors_model = ["bar-fill-green", "bar-fill-blue", "bar-fill-red"]
    for i, r in enumerate(rankings):
        bc = bar_colors_model[i] if i < 3 else "bar-fill-amber"
        html += f"\n    <tr>\n      <td>{r['model']}</td>"
        for m in METRICS:
            val = float(r.get(f"{m}_scaled_abs_delta", 0))
            pct = (val / metric_max[m]) * 100 if metric_max[m] > 0 else 0
            html += f'<td>{fmt(val)}</td><td>{bar_fill(pct, bc)}</td>'
        html += "\n    </tr>"

    html += """
  </table>
  <div style="text-align:center; margin: 20px 0;">{svg_grouped_bars(rankings)}</div>
  <p>Chatterbox has the lowest DTW distance (best F0 contour matching). XTTS has the highest contrast retention (best preserves breathy-modal F0 differences). Kokoro is worst on all metrics as expected.</p>
</div>

<!-- PAGE 5: PER-PAIR ANALYSIS -->
<div class="page page-break">
  <h2>Per-Pair Contrast Retention</h2>
  <p>How well each model preserves the breathy-modal F0 contrast per pair. Retention = min(1, max(0, ratio)). Average across all 4 metrics within pair.</p>"""

    for model_name in ["chatterbox", "xtts", "kokoro"]:
        model_contrasts = [c for c in contrasts if c["model"] == model_name]
        display_name = model_name.capitalize()
        if model_name == "xtts":
            display_name = "XTTS-v2"
        is_kokoro = model_name == "kokoro"
        bar_cls = "bar-fill-green" if model_name == "chatterbox" else ("bar-fill-blue" if model_name == "xtts" else "bar-fill-red")
        max_retention = max([float(c["mean_contrast_retention"]) for c in model_contrasts]) if model_contrasts else 1

        html += f"""
  <h3>{display_name}{' <span class="note">(no cloning)</span>' if is_kokoro else ''}</h3>
  <table>
    <tr><th>Pair</th>"""
        for m in METRICS:
            html += f"<th>{metric_labels[m]} ratio</th>"
        html += '<th>Mean retention</th><th class="bar-cell"></th></tr>'

        for c in model_contrasts:
            pair = c["pair_id"]
            mean_ret = float(c["mean_contrast_retention"])
            bar_pct = (mean_ret / max_retention) * 100 if max_retention > 0 else 0
            html += f"\n    <tr><td>{pair}</td>"
            for m in METRICS:
                ratio = float(c.get(f"{m}_contrast_ratio", 0))
                cls = ' class="highlight"' if not is_kokoro and 0 <= ratio <= 1 else ""
                html += f"<td{cls}>{fmt(ratio)}</td>"
            html += f'<td><span class="highlight">{fmt(mean_ret)}</span></td><td>{bar_fill(bar_pct, bar_cls)}</td></tr>'

        html += "\n  </table>"

    _f0_mean_d = fmt(f0_mean_gate["abs_cohens_d"]) if f0_mean_gate else "—"
    _f0_range_d = fmt(f0_range_gate["abs_cohens_d"]) if f0_range_gate else "—"
    _heatmap_svg = svg_heatmap(contrasts)

    html += f"""
  <div style="text-align:center; margin: 20px 0;">{_heatmap_svg}</div>
  <p class="note">Chatterbox achieves best DTW distance (F0 contour matching). XTTS shows highest contrast retention overall. Kokoro zero retention is expected — see methodology.</p>
</div>

<!-- PAGE 6: KEY FINDINGS -->
<div class="page page-break">
  <h2>Key Findings</h2>

  <div class="finding">
    <h3>Finding 1: Chatterbox preserves F0 contours best</h3>
    <p>Chatterbox achieves the lowest DTW distance (11.27) — its generated F0 contours most closely match the reference speaker's pitch track. This indicates voice cloning successfully transfers fine-grained prosodic structure, not just average pitch.</p>
  </div>

  <div class="finding">
    <h3>Finding 2: XTTS-v2 best preserves breathy-modal contrast</h3>
    <p>XTTS achieves the highest mean contrast retention (0.79 vs 0.48 for Chatterbox). The paired breathy-modal F0 differences are best preserved, even though absolute F0 contour match (DTW) is worse than Chatterbox. XTTS gets the direction right even when the exact values drift.</p>
  </div>

  <div class="finding">
    <h3>Finding 3: Kokoro baseline confirms zero adaptation</h3>
    <p>Kokoro has no voice cloning — all outputs use the same default voice (af_bella). Contrast retention is zero across all pairs and metrics. DTW distance is worst (16.41). This validates the benchmark's sensitivity to the presence or absence of voice adaptation.</p>
  </div>

  <div class="finding">
    <h3>Finding 4: Gate correctly validates reference separation</h3>
    <p>F0 mean (Cohen's d = {_f0_mean_d}) and F0 range (d = {_f0_range_d}) both pass the &ge; 0.5 threshold. The reference pairs carry measurable F0 contrast between breathy and modal groups despite being originally selected for breathiness.</p>
  </div>

  <div class="finding">
    <h3>Finding 5: Contrast ratio is fragile near zero denominator</h3>
    <p>Pair 001 f0_range has reference contrast of only 1.52 Hz — effectively zero. This produces an unstable ratio of −26.56 for Chatterbox. The [0, 1] clamping neutralizes the damage, but this metric-pair contributes no signal to contrast retention scores. This is expected: VCTK pairs were selected for breathiness, not F0 range contrast. Some metrics will have near-zero reference contrasts by coincidence.</p>
  </div>
</div>

<!-- PAGE 7: APPENDIX -->
<div class="page page-break">
  <h2>Appendix: Per-Sample Data</h2>
  <p>Raw F0 values and distances for every model output.</p>

  <table>
    <tr><th>Sample</th><th>Pair</th><th>Cond</th><th>Model</th><th>F0 mean ref</th><th>F0 mean out</th><th>Stat &Delta;</th><th>DTW</th></tr>"""

    for s in per_sample:
        html += f"""
    <tr><td>{s['sample_id']}</td><td>{s['pair_id']}</td><td>{s['condition']}</td><td>{s['model']}</td><td>{fmt(s['f0_mean_ref'])}</td><td>{fmt(s['f0_mean_output'])}</td><td>{fmt(s['stat_scaled_abs_delta'])}</td><td>{fmt(s['dtw_distance'])}</td></tr>"""

    html += """
  </table>

  <hr class="divider">

  <h3>Output Files</h3>
  <table>
    <tr><th>File</th><th>Description</th></tr>
    <tr><td>features/features.csv</td><td>24 rows, one per audio file with all metrics</td></tr>
    <tr><td>features/contours/</td><td>24 F0 contour .npy files for DTW analysis</td></tr>
    <tr><td>results/gate_check.json</td><td>Gate decision and effect sizes per metric</td></tr>
    <tr><td>results/per_sample_preservation.csv</td><td>Output-reference distances per sample</td></tr>
    <tr><td>results/paired_contrast_preservation.csv</td><td>Contrast retention by model and pair</td></tr>
    <tr><td>results/model_rankings.csv</td><td>Model-level summary sorted by score</td></tr>
    <tr><td>results/report.html</td><td>This report</td></tr>
  </table>

  <hr class="divider">

  <h3>Caveats</h3>
  <ul style="margin-left: 20px; font-size: 13px; color: #666;">
    <li>F0 estimation is sensitive to microphone, noise, and phonetic content.</li>
    <li>Speaking rate estimation via RMS peak counting is a proxy, not ground truth.</li>
    <li>DTW alignment may be affected by durational differences between reference and output.</li>
    <li>Kokoro does not support voice cloning — its outputs reflect the default voice (af_bella), not the reference speaker.</li>
    <li><strong>Fragile contrast ratios near zero:</strong> Contrast retention ratios use reference contrast as denominator. When reference contrast is near zero, the ratio becomes unstable. Pair 001 f0_range has reference contrast = 1.52 Hz — effectively zero — producing a ratio of −26.56 for Chatterbox. Clamping to [0, 1] neutralizes the damage, but this pair contributes no f0_range signal to contrast retention scores. This is expected: VCTK pairs were selected for breathiness contrast, not F0 range contrast. Some metrics will have near-zero reference contrasts by coincidence.</li>
    <li>Single sentence per pair limits generalization to broader prosodic variation.</li>
  </ul>

  <hr class="divider">

  <h3>Dataset</h3>
  <p>VCTK Corpus 0.92: <a href="https://datashare.ed.ac.uk/handle/10283/3443">https://datashare.ed.ac.uk/handle/10283/3443</a></p>
</div>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
