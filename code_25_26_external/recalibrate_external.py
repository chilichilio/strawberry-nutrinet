#!/usr/bin/env python3
# ============================================================
# recalibrate_external.py
# ------------------------------------------------------------
# Post-process an external-validation predictions CSV (from
# validate_external_25_26.py or validate_external_dl_25_26.py) and report
# RAW vs RECALIBRATED metrics for the held-out 25_26 season.
#
# WHY
#   A model trained on 24_25 can track 25_26 leaf-K in RANK (Pearson r) while
#   being biased / scale-shifted because the 25_26 target distribution sits at
#   a different level (label shift). R2 punishes that offset+slope harshly.
#   Recalibration = fit a simple correction  y_corr = a*y_pred + b  using a
#   SMALL labeled slice of 25_26, then score the REST. This separates
#   "does it rank correctly" (r, transferable) from "is it on the right
#   absolute scale this season" (needs a few local points).
#
# HONEST BY CONSTRUCTION
#   The correction is fit with GROUPED k-fold over Image_ID: for each fold the
#   line is fit on the OTHER plots and applied to the held-out plots, so no
#   sample ever calibrates itself and no plot leaks across the split. We report
#   the pooled out-of-fold recalibrated predictions.
#
#   Two correction forms are reported:
#     * offset : y_corr = y_pred + b        (removes only the season bias)
#     * linear : y_corr = a*y_pred + b       (removes bias AND slope compression)
#
# CEILING NOTE
#   Linear recalibration cannot raise R2 above r^2 (the squared Pearson
#   correlation). If r is small, the model has little transferable signal and
#   recalibration will NOT rescue it -- that is a modeling/inputs problem, not
#   a calibration problem. The report prints r^2 as the ceiling so this is
#   obvious.
#
# USAGE
#   python recalibrate_external.py \
#       --pred ../25_26/external_validation_M4/external_M4_predictions.csv
#   python recalibrate_external.py --pred <csv> --k 5 --min-cal 8
#
#   # score every external run at once:
#   python recalibrate_external.py --glob "../25_26/external_validation_*/external_*_predictions.csv"
#
# Writes <pred_dir>/<stem>_recalibrated.csv (per-sample) and prints a summary.
# Requires: numpy, pandas (scipy optional; falls back to numpy for r).
# ============================================================
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────
def _r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    if y.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float); p = np.asarray(p, float)
    err = p - y
    return {
        "R2": _r2(y, p),
        "r": _pearson(y, p),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),          # mean(pred - true); >0 = over-predict
    }


# ─────────────────────────────────────────────────────────────
# grouped k-fold recalibration
# ─────────────────────────────────────────────────────────────
def _fit_correction(y: np.ndarray, p: np.ndarray, form: str):
    """Return a function mapping y_pred -> y_corr fit on (y, p)."""
    if form == "offset":
        b = float(np.mean(y - p))
        return lambda x: x + b
    # linear least squares  y ~ a*p + b
    a, b = np.polyfit(p, y, 1)
    return lambda x: a * x + b


def grouped_kfold_recalibrate(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray,
    form: str, k: int, min_cal: int, seed: int = 0,
) -> Optional[np.ndarray]:
    """Out-of-fold recalibrated predictions. Fold assignment is by GROUP
    (Image_ID) so a plot's samples never split across calibration/eval.
    Returns None if there are too few groups to hold out a calibration set."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    uniq = np.array(sorted(pd.unique(groups)))
    if len(uniq) < 2:
        return None
    k = int(max(2, min(k, len(uniq))))
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    fold_of = {g: (i % k) for i, g in enumerate(order)}
    folds = np.array([fold_of[g] for g in groups])

    out = np.full_like(p, np.nan, dtype=float)
    for f in range(k):
        te = folds == f
        tr = ~te
        if te.sum() == 0 or tr.sum() < min_cal:
            return None  # not enough calibration data -> refuse (honest)
        corr = _fit_correction(y[tr], p[tr], form)
        out[te] = corr(p[te])
    return out


# ─────────────────────────────────────────────────────────────
# one file
# ─────────────────────────────────────────────────────────────
def process_file(path: str, k: int, min_cal: int,
                 true_col: str, pred_col: str, group_col: str) -> Dict:
    df = pd.read_csv(path)
    for c in (true_col, pred_col):
        if c not in df.columns:
            raise SystemExit(f"[{path}] missing column '{c}'. "
                             f"cols={list(df.columns)}")
    m = df[true_col].notna() & df[pred_col].notna()
    df = df[m].copy()
    y = df[true_col].to_numpy(float)
    p = df[pred_col].to_numpy(float)
    groups = (df[group_col].astype(str).to_numpy()
              if group_col in df.columns
              else np.arange(len(df)).astype(str))

    raw = _metrics(y, p)
    r = raw["r"]
    ceiling = r * r if np.isfinite(r) else float("nan")

    result = {"raw": raw, "r2_ceiling": ceiling, "n": int(len(y)),
              "n_groups": int(len(pd.unique(groups)))}

    for form in ("offset", "linear"):
        corr = grouped_kfold_recalibrate(y, p, groups, form, k, min_cal)
        if corr is None or np.isnan(corr).any():
            result[form] = None
            continue
        result[form] = _metrics(y, corr)
        df[f"y_pred_{form}_recal"] = corr

    out_csv = os.path.join(
        os.path.dirname(path),
        os.path.basename(path).replace(".csv", "_recalibrated.csv"))
    df.to_csv(out_csv, index=False)
    result["out_csv"] = out_csv
    result["file"] = path
    return result


def _fmt(m: Optional[Dict]) -> str:
    if m is None:
        return "  (skipped: too few plots/points to hold out a calibration set)"
    return (f"R2={m['R2']:+.3f}  r={m['r']:.3f}  RMSE={m['RMSE']:.3f}  "
            f"MAE={m['MAE']:.3f}  bias={m['bias']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="One predictions CSV.")
    ap.add_argument("--glob", help="Glob for many predictions CSVs "
                    "(quote it). Overrides --pred if given.")
    ap.add_argument("--k", type=int, default=5, help="Grouped folds (default 5).")
    ap.add_argument("--min-cal", type=int, default=8,
                    help="Min calibration samples per fold; below this the fold "
                         "is refused (default 8).")
    ap.add_argument("--true-col", default="y_true_reg")
    ap.add_argument("--pred-col", default="y_pred_reg")
    ap.add_argument("--group-col", default="Image_ID")
    ap.add_argument("--summary-csv", default=None,
                    help="If set, write one combined RAW+RECAL table (one row "
                         "per model) to this path.")
    args = ap.parse_args()

    files: List[str] = []
    if args.glob:
        files = sorted(glob.glob(args.glob))
    elif args.pred:
        files = [args.pred]
    if not files:
        raise SystemExit("Provide --pred <csv> or --glob '<pattern>'.")

    print("=" * 74)
    print("EXTERNAL RECALIBRATION  |  train 24_25 -> test 25_26")
    print("grouped k-fold over Image_ID; fold fit on other plots, applied to held-out")
    print("=" * 74)
    summary_rows = []

    def _model_tag(path: str) -> str:
        base = os.path.basename(path)
        for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7"):
            if f"_{m}_" in base or base.startswith(f"external_{m}"):
                return m
        d = os.path.basename(os.path.dirname(path))
        return d.replace("external_validation_", "") or base

    for f in files:
        res = process_file(f, args.k, args.min_cal,
                           args.true_col, args.pred_col, args.group_col)
        model = _model_tag(f)
        print(f"\n{model}   (n={res['n']}, plots={res['n_groups']})")
        print(f"  RAW      {_fmt(res['raw'])}")
        print(f"  OFFSET   {_fmt(res['offset'])}")
        print(f"  LINEAR   {_fmt(res['linear'])}")
        c = res["r2_ceiling"]
        print(f"  ceiling  linear-recal R2 cannot exceed r^2 = {c:.3f}")
        if np.isfinite(c) and c < 0.25:
            print("  NOTE     low r^2 -> little transferable signal; recalibration "
                  "will NOT rescue this run. Check inputs/DAP scale, not calibration.")
        print(f"  wrote    {res['out_csv']}")

        raw = res["raw"]; off = res["offset"]; lin = res["linear"]
        row = {"model": model, "n": res["n"], "plots": res["n_groups"],
               "r": round(raw["r"], 3), "r2_ceiling": round(c, 3),
               "R2_raw": round(raw["R2"], 3), "RMSE_raw": round(raw["RMSE"], 3),
               "bias_raw": round(raw["bias"], 3),
               "R2_offset": None if off is None else round(off["R2"], 3),
               "RMSE_offset": None if off is None else round(off["RMSE"], 3),
               "R2_linear": None if lin is None else round(lin["R2"], 3),
               "RMSE_linear": None if lin is None else round(lin["RMSE"], 3)}
        summary_rows.append(row)

    if args.summary_csv and summary_rows:
        order = {m: i for i, m in enumerate(["M1", "M2", "M3", "M4", "M5", "M6", "M7"])}
        summary_rows.sort(key=lambda x: order.get(x["model"], 99))
        pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
        print(f"\nCombined RAW+RECAL summary -> {args.summary_csv}")
    print()


if __name__ == "__main__":
    main()
