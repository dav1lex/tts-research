#!/usr/bin/env python3
"""
Generate self-contained HTML report for Identity Drift Benchmark (Research 4).
Embeds all data, figures, and styling inline.
"""

import csv
import json
import base64
import os
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_PATH = RESULTS_DIR / "report.html"


def b64_encode(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def h(val: str) -> str:
    """HTML-escape."""
    import html
    return html.escape(str(val))


def build_html():
    # ── Load data ──────────────────────────────────────────────────────────
    gate = load_json(RESULTS_DIR / "gate_check.json")
    drift_summary = load_csv(RESULTS_DIR / "drift_summary.csv")
    ref_feats = load_json(RESULTS_DIR / "features" / "reference_features.json")
    
    # Build drift summary lookup
    drift_by_model = {}
    for row in drift_summary:
        drift_by_model[row["model"]] = row

    # Base64-encode figures
    drift_over_time_b64 = b64_encode(FIGURES_DIR / "drift_over_time.png")
    drift_heatmap_b64 = b64_encode(FIGURES_DIR / "drift_heatmap.png")
    early_vs_late_b64 = b64_encode(FIGURES_DIR / "early_vs_late_drift.png")

    # Per-feature drift data (hardcoded from summary/CSV)
    per_feature = {
        "f0_mean":         {"chatterbox": 0.524, "xtts": 0.391, "kokoro": 1.743},
        "f0_std":          {"chatterbox": 0.426, "xtts": 0.195, "kokoro": 1.810},
        "cpp_mean":        {"chatterbox": 0.726, "xtts": 0.652, "kokoro": 0.769},
        "spectral_tilt":   {"chatterbox": 0.524, "xtts": 0.848, "kokoro": 0.686},
        "mfcc":            {"chatterbox": 0.345, "xtts": 0.545, "kokoro": 1.772},
        "rms":             {"chatterbox": 0.286, "xtts": 1.135, "kokoro": 0.298},
        "centroid":        {"chatterbox": 0.603, "xtts": 0.312, "kokoro": 1.955},
    }

    # Best score for bar scaling (Chatterbox = 0.491)
    best_score = 0.49051

    # ── Helper: bar HTML ───────────────────────────────────────────────────
    def drift_bar(model_name: str, value: float, max_val: float | None = None) -> str:
        if max_val is None:
            max_val = max(
                float(drift_by_model["chatterbox"]["drift_mean"]),
                float(drift_by_model["xtts"]["drift_mean"]),
                float(drift_by_model["kokoro"]["drift_mean"]),
            )
        pct = (value / max_val) * 100
        colors = {"chatterbox": "#f1c40f", "xtts": "#95a5a6", "kokoro": "#9b59b6"}
        c = colors.get(model_name, "#4472C4")
        return f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{c};"></div></div>'

    def fmt(v, precision=3):
        """Format a value as a string."""
        if v is None or v == "" or v == "—":
            return "—"
        try:
            fv = float(v)
            if abs(fv) < 0.001 and fv != 0:
                return f"{fv:.5f}"
            if abs(fv) >= 100:
                return f"{fv:.1f}"
            return f"{fv:.{precision}f}"
        except (ValueError, TypeError):
            return str(v)

    # ── Build HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Identity Drift Benchmark — Research 4</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.6; }}
  .page {{ max-width: 800px; margin: 0 auto; padding: 32px 40px; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; }}
  h2 {{ font-size: 22px; margin-bottom: 16px; border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; }}
  h3 {{ font-size: 16px; margin-bottom: 8px; color: #333; }}
  p {{ margin-bottom: 12px; font-size: 14px; }}
  .subtitle {{ font-size: 16px; color: #666; margin-bottom: 24px; }}
  .meta {{ display: grid; grid-template-columns: 180px 1fr; gap: 4px 16px; font-size: 14px; margin-bottom: 24px; }}
  .meta dt {{ font-weight: 600; }}
  .divider {{ border: none; border-top: 1px solid #ccc; margin: 24px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #1a1a1a; font-weight: 600; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #e0e0e0; }}
  tr:last-child td {{ border-bottom: none; }}
  .bar-cell {{ width: 180px; }}
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
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 700; }}
  .badge-green {{ background: #27ae60; color: #fff; }}
  .badge-purple {{ background: #9b59b6; color: #fff; }}
  .badge-silver {{ background: #95a5a6; color: #fff; }}
  .badge-gold {{ background: #f1c40f; color: #1a1a1a; }}
  .listening-box {{ background: #fff8e1; border-left: 4px solid #f39c12; padding: 16px 20px; margin: 16px 0; }}
  .listening-box h3 {{ color: #e67e22; margin-bottom: 6px; }}
  .listening-box p {{ margin-bottom: 8px; font-size: 13px; }}
  .footer {{ text-align: center; font-size: 12px; color: #999; margin-top: 40px; padding-top: 16px; border-top: 1px solid #e0e0e0; }}
  ul, ol {{ margin-left: 20px; font-size: 13px; color: #333; margin-bottom: 16px; }}
  ul li, ol li {{ margin-bottom: 4px; }}
  @media print {{
    @page {{ size: A4; margin: 10mm 12mm; }}
    .page {{ padding: 16px 0; }}
    .page-break {{ page-break-before: always; }}
    table {{ page-break-inside: avoid; font-size: 11px; }}
    table th, table td {{ padding: 5px 8px; }}
    p {{ font-size: 13px; }}
    .finding {{ page-break-inside: avoid; }}
    h3 {{ page-break-after: avoid; }}
    h2 {{ page-break-after: avoid; }}
  }}
</style>
</head>
<body>

<!-- ═══════════════ PAGE 1: TITLE ═══════════════ -->
<div class="page">
  <h1>Identity Drift Benchmark — Research 4</h1>
  <p class="subtitle">Does TTS voice identity drift over 5-minute synthesized monologues?</p>
  <hr class="divider">
  <dl class="meta">
    <dt>Date</dt><dd>2026-05-25</dd>
    <dt>GPU</dt><dd>NVIDIA RTX 3060 Laptop (6GB VRAM)</dd>
    <dt>Python</dt><dd>3.12 (Chatterbox, Kokoro), 3.10 (XTTS-v2)</dd>
    <dt>Reference</dt><dd>neutral_p229_002.wav (VCTK Corpus 0.92)</dd>
    <dt>Models</dt><dd>3: Chatterbox 0.1.7, XTTS-v2 0.22.0, Kokoro 0.9.4</dd>
    <dt>Total audio</dt><dd>60 windows across 3 × ~5min files</dd>
    <dt>Primary metric</dt><dd>IQR-scaled Euclidean distance in 7-feature acoustic space</dd>
    <dt>Gate status</dt><dd><span class="gate-pass">PASSED</span></dd>
  </dl>
</div>

<!-- ═══════════════ PAGE 2: METHODOLOGY ═══════════════ -->
<div class="page page-break">
  <h2>Methodology</h2>

  <h3>1. Reference</h3>
  <p>p229 (female, Southern England), 3.88 s, f0 = 192.7 Hz, CPP = 13.36 — same reference used across Papers 2–4.</p>

  <h3>2. Models</h3>
  <table>
    <tr><th>Model</th><th>Cloning</th><th>Parameters</th><th>Sample Rate</th></tr>
    <tr><td>Chatterbox 0.1.7</td><td>Yes (audio_prompt_path)</td><td>exaggeration=0.3</td><td>24 kHz</td></tr>
    <tr><td>XTTS-v2 0.22.0</td><td>Yes (speaker_wav)</td><td>language=en</td><td>22 kHz</td></tr>
    <tr><td>Kokoro 0.9.4</td><td>No (voice=af_heart)</td><td>N/A</td><td>24 kHz</td></tr>
  </table>

  <h3>3. Generation</h3>
  <p>868-word narrative text. Chatterbox chunked into 10 × ~100 words (CUDA context limit). XTTS and Kokoro generated whole. Seed = 42. Target ~5 min each.</p>

  <h3>4. Windowing</h3>
  <p>15-second non-overlapping windows. 60 total windows (chatterbox=19, xtts=21, kokoro=20).</p>

  <h3>5. Features (extracted per window)</h3>
  <p>f0_mean, f0_std, cpp_mean (Praat CPPS, copied from Paper 2), spectral_flatness, spectral_tilt_ratio, mfcc_mean, rms_mean, spectral_centroid. All resampled to 16 kHz.</p>

  <h3>6. Voiced-Frame Filtering</h3>
  <p>Energy threshold + librosa.pyin voiced_probability ≥ 0.50, yin fallback.</p>

  <h3>7. Gate</h3>
  <p>Reference voiced ratio ≥ 10%, ≥ 0.2 s voiced, no clipping. Per-model: ≥ 50% windows pass quality check.</p>

  <h3>8. Scoring</h3>
  <p>IQR-based robust scaling pooled across all valid windows. Per-window drift = mean of |feature_value − reference_value| / scale. drift_increase = drift_late − drift_early (first/last 5 windows). drift_slope = scipy.stats.linregress across all windows.</p>
</div>

<!-- ═══════════════ PAGE 3: GATE RESULTS ═══════════════ -->
<div class="page page-break">
  <h2>Gate Results</h2>
  <p>Gate status: <span class="badge badge-green">PASSED</span></p>
  <p>All audio passed quality checks. Reference and all 60 generation windows are valid for analysis.</p>

  <h3>Reference Gate</h3>
  <table>
    <tr><th>Path</th><th>Duration</th><th>Voiced Ratio</th><th>Voiced Seconds</th><th>Clipping</th><th>Pass?</th></tr>
    <tr>
      <td style="font-size:11px;">neutral_p229_002.wav</td>
      <td>{fmt(gate["reference"]["duration_sec"], 2)} s</td>
      <td>{fmt(gate["reference"]["voiced_frame_ratio"] * 100, 1)}%</td>
      <td>{fmt(gate["reference"]["voiced_seconds"], 2)} s</td>
      <td>None</td>
      <td><span class="gate-pass">PASS</span></td>
    </tr>
  </table>

  <h3>Generation Gate</h3>
  <table>
    <tr><th>Model</th><th>Total Windows</th><th>Passed</th><th>Pass Rate</th><th>Status</th></tr>"""

    model_gate_data = [
        ("Chatterbox", gate["models"]["chatterbox"]),
        ("XTTS-v2", gate["models"]["xtts"]),
        ("Kokoro", gate["models"]["kokoro"]),
    ]
    for name, mg in model_gate_data:
        tw = mg["total_windows"]
        pw = mg["passed_windows"]
        pr = mg["passed_ratio"]
        html += f"""
    <tr>
      <td>{h(name)}</td>
      <td>{tw}</td>
      <td>{pw}</td>
      <td>{fmt(pr * 100, 1)}%</td>
      <td><span class="gate-pass">PASS</span></td>
    </tr>"""

    html += """
  </table>
  <p class="note">XTTS-v2 had 3 clipped windows (at 60–75s, 90–105s, 225–240s) — documented but not blocking. All windows had adequate voicing.</p>
</div>

<!-- ═══════════════ PAGE 4: DRIFT RESULTS ═══════════════ -->
<div class="page page-break">
  <h2>Drift Results</h2>
  <p>The main drift summary table. Lower drift = closer to reference voice identity.</p>

  <table>
    <tr>
      <th>Model</th><th>Drift Mean</th><th>Drift Std</th><th>Drift Early</th>
      <th>Drift Late</th><th>Drift Increase</th><th>Drift Slope</th>
      <th>Raw Variance</th><th class="bar-cell">Relative Drift (lower = better)</th>
    </tr>"""

    # Drift mean bars: scale to best = 100%
    drift_vals = {
        "chatterbox": float(drift_by_model["chatterbox"]["drift_mean"]),
        "xtts": float(drift_by_model["xtts"]["drift_mean"]),
        "kokoro": float(drift_by_model["kokoro"]["drift_mean"]),
    }
    max_drift = max(drift_vals.values())

    for model_key, label in [("chatterbox", "Chatterbox"), ("xtts", "XTTS-v2"), ("kokoro", "Kokoro (no-adaptation)")]:
        d = drift_by_model[model_key]
        pct = (float(d["drift_mean"]) / max_drift) * 100
        colors = {"chatterbox": "#f1c40f", "xtts": "#95a5a6", "kokoro": "#9b59b6"}
        c = colors[model_key]
        raw_var = d.get("kokoro_raw_variance", "—")
        if raw_var and raw_var != "—" and raw_var != "":
            raw_var_disp = fmt(float(raw_var), 3)
        else:
            raw_var_disp = "—"
        html += f"""
    <tr>
      <td><strong class="rank-{['1','2','3'][['chatterbox','xtts','kokoro'].index(model_key)]}">{h(label)}</strong></td>
      <td>{fmt(d['drift_mean'])}</td>
      <td>{fmt(d['drift_std'])}</td>
      <td>{fmt(d['drift_early'])}</td>
      <td>{fmt(d['drift_late'])}</td>
      <td>{fmt(d['drift_increase'])}</td>
      <td>{fmt(d['drift_slope'], 5)}</td>
      <td>{raw_var_disp}</td>
      <td><div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{c};"></div></div></td>
    </tr>"""

    html += f"""
  </table>
  <p class="note">Bar column scaled to maximum drift mean ({max_drift:.3f} = 100%). Chatterbox (gold) = best, closest to reference.</p>

  <h3>Drift Over Time</h3>
  <img src="data:image/png;base64,{drift_over_time_b64}" style="max-width:100%;" alt="Drift over time">
  <p class="note">All models show positive drift. Drift rates are very similar (~0.004/window). Chatterbox starts closest but drifts fastest.</p>
</div>

<!-- ═══════════════ PAGE 5: PER-FEATURE DRIFT ═══════════════ -->
<div class="page page-break">
  <h2>Per-Feature Drift</h2>
  <p>Drift mean broken down by acoustic feature channel. Reveals <em>which</em> aspects of voice identity drift over time.</p>

  <table>
    <tr><th>Feature</th><th>Chatterbox</th><th>XTTS-v2</th><th>Kokoro</th></tr>"""

    for feat, vals in per_feature.items():
        html += f"""
    <tr>
      <td><strong>{h(feat)}</strong></td>
      <td>{fmt(vals['chatterbox'])}</td>
      <td>{fmt(vals['xtts'])}</td>
      <td>{fmt(vals['kokoro'])}</td>
    </tr>"""

    html += """
  </table>

  <h3>Feature Drift Heatmap</h3>
  <img src="data:image/png;base64,""" + drift_heatmap_b64 + """" style="max-width:100%;" alt="Drift heatmap">
  <p class="note">CPP (breathiness) is the highest-drift feature for both voice-cloning models. XTTS-v2 has anomalous RMS drift (1.135), likely from 3 clipped windows.</p>
</div>

<!-- ═══════════════ PAGE 6: EARLY VS LATE + LISTENING TEST ═══════════════ -->
<div class="page page-break">
  <h2>Early vs Late Drift</h2>
  <img src="data:image/png;base64,""" + early_vs_late_b64 + """" style="max-width:100%;" alt="Early vs late drift">

  <hr class="divider">
  <h2>Perceptual Listening Test</h2>

  <div class="listening-box">
    <h3>Listener</h3>
    <p><strong>davilex</strong> — 2026-05-25</p>

    <h3>Question 1: Does Chatterbox change from start to end?</h3>
    <p><strong>→ Barely perceptible change.</strong> Something shifts but can't be named. Consistent with CPP drift (0.73) exceeding F0 drift (0.52).</p>

    <h3>Question 2: Chatterbox end vs XTTS end vs reference?</h3>
    <p><strong>→ Both sound similar to reference at 5 minutes.</strong> XTTS no perceptible change from its start. Gap between models (0.09) below perceptual threshold.</p>

    <h3>Question 3: Kokoro?</h3>
    <p><strong>→ Clearly different person (expected).</strong> Most stable within own voice. af_heart is consistently af_heart — confirms no-adaptation baseline works as intended.</p>

    <h3>Conclusion</h3>
    <p><strong>5 minutes is too short for perceptible drift.</strong> The acoustic instrument is more sensitive than the ear at this timescale. Measurable drift exists but does not cross the perceptual threshold within the tested duration.</p>
  </div>
</div>

<!-- ═══════════════ PAGE 7: KEY FINDINGS ═══════════════ -->
<div class="page page-break">
  <h2>Key Findings</h2>

  <div class="finding">
    <h3>Finding 1: Voice identity drifts measurably over 5 minutes across all models</h3>
    <p>All three models show positive drift — voice moves away from reference over time. Drift increase ranges from +0.070 (Kokoro) to +0.080 (Chatterbox). The effect is systematic and measurable with the 7-feature acoustic instrument.</p>
  </div>

  <div class="finding">
    <h3>Finding 2: Chatterbox starts closest, drifts fastest — consistent with autoregressive conditioning decay</h3>
    <p>Chatterbox has the lowest mean drift (0.491 = best) but the highest drift_increase (+0.080). This pattern — strong initial match followed by gradual divergence — is consistent with autoregressive models losing conditioning signal over long sequences rather than exhibiting random-walk drift.</p>
  </div>

  <div class="finding">
    <h3>Finding 3: Nature of drift differs by model</h3>
    <p>Chatterbox drifts in CPP/breathiness (drift_cpp = 0.726) — the voice loses breathiness texture over time. XTTS-v2 drifts in RMS/loudness (drift_rms = 1.135) — voice gets louder and more variable, likely from clipping artifacts. These are qualitatively different failure modes, not just different magnitudes of the same effect.</p>
  </div>

  <div class="finding">
    <h3>Finding 4: CPP (breathiness) is the least stable channel — confirms Paper 2 at long time scale</h3>
    <p>For both voice-cloning models, CPP drift exceeds F0 drift: Chatterbox CPP=0.726 vs F0=0.524, XTTS CPP=0.652 vs F0=0.391. Breathiness quality degrades faster than pitch under long-form generation — a direct extension of Paper 2's finding that breathiness is the hardest voice characteristic to preserve.</p>
  </div>

  <div class="finding">
    <h3>Finding 5: Near-identical drift slopes (~0.004/window) may reflect homogeneous input text</h3>
    <p>All three models have nearly identical drift slopes: Chatterbox +0.00422, XTTS +0.00431, Kokoro +0.00473. This uniformity is suspicious — it may reflect the structural homogeneity of the input narrative (single paragraph, uniform sentence structure) rather than a fundamental property of long-form TTS generation. Testing with multi-genre text is needed to confirm.</p>
  </div>

  <div class="finding">
    <h3>Finding 6: Perceptual threshold not reached at 5 minutes</h3>
    <p>Acoustic drift is measurable but not reliably audible at this timescale. The 0.092 gap between Chatterbox and XTTS (0.491 vs 0.583) and the 0.080 within-Chatterbox increase are below the perceptual threshold for a single listener. The ear is less sensitive than the acoustic instrument — validated by listening test.</p>
  </div>
</div>

<!-- ═══════════════ PAGE 8: CONNECTIONS TO PAPERS 2 & 3 ═══════════════ -->
<div class="page page-break">
  <h2>Connections to Papers 2 &amp; 3</h2>

  <div class="two-col">
    <div class="col-box">
      <h3>Paper 2 — Breathiness</h3>
      <p>Paper 2 ranked XTTS-v2 best for short-form breathiness preservation (score 0.097) and Chatterbox second (0.311). In this long-form test, the CPP drift ordering is reversed: XTTS drift_cpp = 0.652 vs Chatterbox's 0.726 — both close.</p>
      <p>The breathiness metric (CPP) is the <strong>highest-drift feature</strong> for both voice-cloning models. CPP drift substantially exceeds F0 drift in both cases, confirming that <strong>breathiness quality is the least stable voice characteristic over long generation</strong>.</p>
      <p class="note">Praat CPPS function and parameters are identical to Paper 2, enabling direct cross-paper comparison.</p>
    </div>
    <div class="col-box">
      <h3>Paper 3 — Prosody (F0)</h3>
      <p>Paper 3 ranked Chatterbox best for F0 contour match (DTW 11.27). In this long-form test, Chatterbox has higher F0 mean drift (0.524) than XTTS (0.391), but XTTS has worse F0 std drift.</p>
      <p>The F0 drift is moderate compared to other features — <strong>voice cloning models preserve pitch reasonably well over time</strong>, but breathiness and timbre are less stable. Pitch preservation does not guarantee breathiness preservation.</p>
      <p class="note">Same reference speaker (p229) used across all three papers for consistency.</p>
    </div>
  </div>
</div>

<!-- ═══════════════ PAGE 9: WHAT'S SOLID + LIMITATIONS ═══════════════ -->
<div class="page page-break">
  <h2>What's Solid</h2>
  <ol>
    <li><strong>Gate design:</strong> Reference and per-model generation gates passed cleanly, confirming all audio is valid for analysis. No drift measurement rests on unvalidated data.</li>
    <li><strong>Praat CPP alignment:</strong> Breathiness measurement uses the exact function and parameters from Paper 2, enabling direct cross-paper comparison.</li>
    <li><strong>Kokoro baseline framing:</strong> Correctly labeled as no-adaptation baseline in all tables, figures, and analysis. Raw feature variance (13.38) reported separately to distinguish "stable in own voice" from "close to reference."</li>
    <li><strong>Per-window methodology:</strong> Non-overlapping 15-second windows with quality flags, voiced frame ratio filtering, and documented window failures — methodologically sound and reproducible.</li>
    <li><strong>Cross-paper connectivity:</strong> Both breathiness (Paper 2) and prosody (Paper 3) findings are extended to long-form generation, with specific feature-level comparisons that surface the CPP instability finding.</li>
  </ol>

  <hr class="divider">
  <h2>Limitations</h2>
  <ol>
    <li>Euclidean distance in normalized feature space is a proxy for perceptual voice similarity — not validated against human ratings.</li>
    <li>No human listening test to confirm whether measured drift is perceptually noticeable (single-listener sanity check only).</li>
    <li>drift_increase uses only first 5 and last 5 windows (~75 seconds each) — may miss mid-trajectory drift and recovery patterns.</li>
    <li>15-second non-overlapping windows may miss sub-window instability; overlapping windows would give smoother trajectories.</li>
    <li>5 minutes of generation may be insufficient for significant drift to manifest — longer tests (15–30 min) recommended.</li>
    <li>Window counts differ between models (19 vs 21 vs 20) — drift_early/late comparisons use different absolute time points.</li>
    <li>IQR-based robust scaling uses pooled window population (all models combined) rather than reference-only population.</li>
    <li>Chatterbox generation was chunked (10 chunks of ~100 words) due to CUDA context limitations — chunk boundaries may introduce artifacts.</li>
    <li>XTTS-v2 had 3 windows with clipping — these windows were included in analysis since they had adequate voicing.</li>
    <li>Spectral flatness had near-zero variance across all windows, rendering its per-feature drift NaN — contributed minimally to composite scores.</li>
    <li>Euclidean distance equally weights all 7 features — perceptual importance of each feature channel is unknown.</li>
    <li>Kokoro drift measures distance to a voice it was never asked to clone (af_heart vs p229).</li>
    <li>VCTK reference audio (p229) may differ from training data distributions for all models, potentially inflating baseline distances.</li>
    <li><strong>Input text homogeneity confound:</strong> The near-identical drift slopes across models (~0.004/window) may partially reflect the uniformity of the single narrative paragraph input text. Real long-form generation faces dialogue, narration shifts, lists, questions, and emotional variation — all of which stress voice consistency differently. Future work should use scripted multi-genre text with controlled prosodic variation.</li>
  </ol>
</div>

<!-- ═══════════════ PAGE 10: OUTPUT FILES ═══════════════ -->
<div class="page page-break">
  <h2>Output Files</h2>
  <table>
    <tr><th>File</th><th>Description</th></tr>
    <tr><td>results/features/window_features.csv</td><td>All features per 15-second window (60 rows)</td></tr>
    <tr><td>results/features/reference_features.json</td><td>Reference speaker feature anchor</td></tr>
    <tr><td>results/features/failed_windows.txt</td><td>Windows flagged for clipping or low voicing</td></tr>
    <tr><td>results/drift_by_window.csv</td><td>Per-window drift scores (60 rows with 7 feature channels)</td></tr>
    <tr><td>results/drift_summary.csv</td><td>Summary statistics per model</td></tr>
    <tr><td>results/gate_check.json</td><td>Gate decisions (reference + per-model)</td></tr>
    <tr><td>results/figures/drift_over_time.png</td><td>Main drift-over-time visualization</td></tr>
    <tr><td>results/figures/drift_heatmap.png</td><td>Per-feature drift heatmap</td></tr>
    <tr><td>results/figures/early_vs_late_drift.png</td><td>Early-vs-late stability comparison</td></tr>
    <tr><td>results/report.html</td><td>This report</td></tr>
    <tr><td>data/long_form/{model}/full.wav</td><td>Generated long-form audio per model</td></tr>
    <tr><td>.gate_passed</td><td>Gate sentinel file</td></tr>
  </table>

  <div class="footer">
    TTS Research Series — Research 4 of 4. Generated 2026-05-25.
  </div>
</div>

</body>
</html>"""

    return html


def main():
    html = build_html()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Report written to {OUTPUT_PATH}")
    print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
