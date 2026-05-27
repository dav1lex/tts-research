#!/usr/bin/env python3
"""Mixed-effects analysis for V4.

Fits, per model, two random-intercept mixed models:
  - f0_cv ~ condition_num + (1 | target_id) + (1 | repetition)
  - speaking_rate ~ condition_num + (1 | target_id) + (1 | repetition)

Where condition_num = 0 for noun, 1 for number.

Also checks XTTS seed coupling risk by scanning manifest.csv and reporting whether
noun/number seeds are identical for any (target_id, repetition) pair.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Optional model filter (comma-separated).")
    ap.add_argument("--no-seed-check", action="store_true", help="Skip XTTS seed coupling check.")
    args = ap.parse_args()

    features_csv = PROJECT_DIR / "features" / "features.csv"
    manifest_csv = PROJECT_DIR / "manifest.csv"

    if not features_csv.exists():
        raise SystemExit(f"missing {features_csv} (run extract_features.py first)")

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as e:
        raise SystemExit(
            f"missing dependency for analysis ({type(e).__name__}: {e}). "
            "Need pandas + statsmodels in the environment running this script."
        )

    df = pd.read_csv(features_csv)
    required_cols = {"model", "condition", "target_id", "repetition", "f0_cv", "speaking_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"features.csv missing columns: {sorted(missing)}")

    # Normalize types
    df["condition"] = df["condition"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["target_id"] = df["target_id"].astype(str).str.strip()
    df["repetition"] = df["repetition"].astype(str).str.strip()

    def cond_to_num(c: str) -> float:
        c = (c or "").strip()
        if c == "noun":
            return 0.0
        if c == "number":
            return 1.0
        return math.nan

    df["condition_num"] = df["condition"].map(cond_to_num)
    df = df[df["condition_num"].notna()].copy()

    # Optional model filtering
    models = sorted(set(df["model"].tolist()))
    if args.model.strip():
        filt = {m.strip() for m in args.model.split(",") if m.strip()}
        df = df[df["model"].isin(filt)].copy()
        models = sorted(set(df["model"].tolist()))

    if not models:
        raise SystemExit("no rows after filtering; nothing to analyze")

    if not args.no_seed_check:
        if not manifest_csv.exists():
            print(f"XTTS seed check: missing {manifest_csv} (skipping)", file=sys.stderr)
        else:
            _xtts_seed_coupling_check(manifest_csv)

    for model_name in models:
        sub = df[df["model"] == model_name].copy()
        # Ensure numeric
        for col in ("f0_cv", "speaking_rate"):
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=["f0_cv", "speaking_rate", "condition_num", "target_id", "repetition"])

        print(f"\n== {model_name} ==")
        print(f"N={len(sub)}")

        # Random intercept for target_id (groups) + random intercept for repetition via variance components
        # statsmodels MixedLM supports one groups arg plus vc_formula for additional components.
        vc = {"repetition": "0 + C(repetition)"}

        _fit_and_print(
            smf,
            sub,
            formula="f0_cv ~ condition_num",
            groups="target_id",
            vc_formula=vc,
            label="f0_cv",
        )
        _fit_and_print(
            smf,
            sub,
            formula="speaking_rate ~ condition_num",
            groups="target_id",
            vc_formula=vc,
            label="speaking_rate",
        )


def _fit_and_print(smf, df, formula: str, groups: str, vc_formula: dict, label: str) -> None:
    try:
        md = smf.mixedlm(formula, df, groups=df[groups], vc_formula=vc_formula, re_formula="1")
        res = md.fit(method="lbfgs", maxiter=200, disp=False)
    except Exception as e:
        print(f"{label}: FIT FAILED ({type(e).__name__}: {str(e)[:200]})")
        return

    term = "condition_num"
    if term not in res.params.index:
        print(f"{label}: missing term {term} in fit params")
        return

    coef = float(res.params[term])
    try:
        ci = res.conf_int().loc[term].tolist()
        ci_lo = float(ci[0])
        ci_hi = float(ci[1])
    except Exception:
        ci_lo = float("nan")
        ci_hi = float("nan")
    try:
        p = float(res.pvalues[term])
    except Exception:
        p = float("nan")

    print(f"{label}: coef={coef:+.6f}  95%CI=[{ci_lo:+.6f}, {ci_hi:+.6f}]  p={p:.6g}")


def _xtts_seed_coupling_check(manifest_csv: Path) -> None:
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    xtts = [r for r in rows if (r.get("model") or "").strip() == "xtts"]
    if not xtts:
        print("XTTS seed check: no xtts rows in manifest.csv")
        return

    # key -> {condition -> seed}
    by = defaultdict(dict)
    for r in xtts:
        key = (r.get("target_id", "").strip(), (r.get("repetition") or "").strip())
        cond = (r.get("condition") or "").strip()
        seed = (r.get("seed") or "").strip()
        if key[0] and key[1] and cond:
            by[key][cond] = seed

    identical = []
    incomplete = 0
    for key, m in by.items():
        s_n = m.get("noun")
        s_num = m.get("number")
        if not s_n or not s_num:
            incomplete += 1
            continue
        if s_n == s_num:
            identical.append((key[0], key[1], s_n))

    print("\nXTTS seed coupling check (manifest.csv)")
    print(f"pairs_total={len(by)} pairs_incomplete={incomplete} pairs_identical_seed={len(identical)}")
    for t, rep, seed in identical[:20]:
        print(f"IDENTICAL seed target_id={t} repetition={rep} seed={seed}")


if __name__ == "__main__":
    main()
