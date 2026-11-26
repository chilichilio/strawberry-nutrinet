#!/usr/bin/env python3
"""
paired_tests.py: paired fold-level model comparisons (Supplementary Tables S3, S12, S13).

For each pair of runs evaluated on the same five plot-grouped outer folds, the
fold-wise difference d_k = metric(A)_k - metric(B)_k is tested with the
Nadeau-Bengio corrected resampled t-test:

    se = sqrt((1/k + n_test/n_train) * var(d)),   t = mean(d) / se,   df = k - 1

with k = 5 and n_test/n_train taken from the season size (N: 360/1439,
K: 252/1007). The corrected 95% CI is mean(d) +/- t_{0.975, 4} * se, and
Cohen's dz = mean(d) / sd(d). p-values are adjusted with Benjamini-Hochberg within
each nutrient x metric x comparison family:

    S3   temporal image encoding of M7 (3 comparisons)
    S12  progressive model comparisons (8 prespecified comparisons; M5 vs M6 is
         reported descriptively, outside the adjusted family)
    S13  input ablations (12 comparisons)

Fold metrics are read from the run folders written by run_ml_v4.py / run_dl_v4.py:
    classical runs : <run>/<M?_RF|M?_PLSR>_fold_metrics.csv
    deep runs      : <run>/cv_results_<model_tag>_reg.csv
For N, the classical models are read from outputs_dlfold/ (re-run on the deep-learning
outer folds via V4_FOLD_MANIFEST), so every N comparison is paired on identical folds.
K classical and deep-learning folds are identical by construction.

    python paired_tests.py [--out-root outputs] [--dlfold-root outputs_dlfold] [--csv out.csv]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N_OBS = {"N": 1799, "K": 1259}

S = "__sampEnd"
M6 = "M6__seqfix_sampEnd_nested"
M7 = "M7__seqfix_sampEnd"
LABEL = {
    "M1" + S: "M1", "M2" + S: "M2", "M3" + S: "M3", "M4" + S: "M4", "M5" + S: "M5", M6: "M6", M7: "M7",
    "M2__sampEnd_noUAV": "M2 without UAV descriptors",
    "M4__sampEnd_noUAV": "M4 without UAV descriptors",
    "M6__seqfix_sampEnd_noUAV_nested": "M6 without UAV descriptors",
    "M7__seqfix_sampEnd_noUAV": "M7 without imagery",
    "M2__sampEnd_noXY": "M2 without plot coordinates",
    "M4__sampEnd_noXY": "M4 without plot coordinates",
    "M6__seqfix_sampEnd_noXY_nested": "M6 without plot coordinates",
    "M7__seqfix_sampEnd_noXY": "M7 without plot coordinates",
    "M6__seqfix_sampEnd_noW_nested": "M6 without daily weather sequence",
    "M7__seqfix_sampEnd_noW": "M7 without daily weather sequence",
    "M7__seqfix_sampEnd_noF": "M7 without fertilizer and weather inputs",
    "M7__seqfix_sampEnd_noUAV_noF": "M7 without fertilizer, weather and imagery",
    "M7__seqfix_sampEnd_single": "M7, matched acquisition only",
    "M7__seqfix_sampEnd_lstm": "M7, BiLSTM image-sequence encoder",
    "M7__seqfix_sampEnd_noTemp": "M7, no temporal image encoder",
}
FAMILIES = {
    "S3": [(M7, "M7__seqfix_sampEnd_single"), (M7, "M7__seqfix_sampEnd_lstm"),
           (M7, "M7__seqfix_sampEnd_noTemp")],
    "S12": [("M2" + S, "M1" + S), ("M3" + S, "M1" + S), ("M4" + S, "M2" + S), ("M5" + S, "M4" + S),
            (M6, "M4" + S), (M7, "M5" + S), (M7, M6), (M7, "M4" + S)],
    "S13": [("M2" + S, "M2__sampEnd_noUAV"), ("M4" + S, "M4__sampEnd_noUAV"),
            (M6, "M6__seqfix_sampEnd_noUAV_nested"), (M7, "M7__seqfix_sampEnd_noUAV"),
            ("M2" + S, "M2__sampEnd_noXY"), ("M4" + S, "M4__sampEnd_noXY"),
            (M6, "M6__seqfix_sampEnd_noXY_nested"), (M7, "M7__seqfix_sampEnd_noXY"),
            (M6, "M6__seqfix_sampEnd_noW_nested"), (M7, "M7__seqfix_sampEnd_noW"),
            (M7, "M7__seqfix_sampEnd_noF"), ("M7__seqfix_sampEnd_noF", "M7__seqfix_sampEnd_noUAV_noF")],
}
DESCRIPTIVE = {"S12": [("M5" + S, M6)]}


def fold_metrics(nut, run, out_root, dlfold_root):
    """Five-row table (fold, R2, RMSE) for one run."""
    deep = run.startswith(("M5", "M7"))
    roots = [out_root] if deep or nut == "K" else [dlfold_root, out_root]
    for root in roots:
        d = os.path.join(root, nut, run)
        files = glob.glob(os.path.join(d, "cv_results_*_reg.csv")) if deep else \
            glob.glob(os.path.join(d, "*_fold_metrics.csv"))
        if files:
            f = pd.read_csv(sorted(files)[0])
            f = f[pd.to_numeric(f["fold"], errors="coerce").notna()].copy()
            f["fold"] = f["fold"].astype(int)
            f = f.set_index("fold").sort_index().loc[[1, 2, 3, 4, 5]]
            return f[["R2", "RMSE"]], os.path.relpath(sorted(files)[0], ROOT)
    return None, None


def corrected_test(a, b, metric, nut):
    d = (a[metric] - b[metric]).to_numpy(float)
    k = len(d)
    n_test = round(N_OBS[nut] / k)
    ratio = n_test / (N_OBS[nut] - n_test)
    sd = d.std(ddof=1)
    se = np.sqrt((1 / k + ratio) * sd ** 2)
    t = d.mean() / se
    tc = stats.t.ppf(0.975, k - 1)
    favour = (d > 0) if metric == "R2" else (d < 0)
    return dict(mean_A=a[metric].mean(), mean_B=b[metric].mean(), delta=d.mean(),
                ci_lo=d.mean() - tc * se, ci_hi=d.mean() + tc * se, t_corrected=t,
                p=2 * stats.t.sf(abs(t), k - 1), cohens_dz=d.mean() / sd,
                folds_favouring_A=int(favour.sum()))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.environ.get("NUTRINET_OUT", os.path.join(ROOT, "outputs")))
    ap.add_argument("--dlfold-root", default=None, help="default: <out-root>_dlfold")
    ap.add_argument("--csv", default=None, help="default: <out-root>/paired_tests.csv")
    a = ap.parse_args()
    dlfold = a.dlfold_root or a.out_root.rstrip(os.sep) + "_dlfold"
    rows, missing = [], set()
    for fam, pairs in FAMILIES.items():
        for nut in ("N", "K"):
            for metric in ("R2", "RMSE"):
                block = []
                for A, B in pairs + DESCRIPTIVE.get(fam, []):
                    fa, sa = fold_metrics(nut, A, a.out_root, dlfold)
                    fb, sb = fold_metrics(nut, B, a.out_root, dlfold)
                    if fa is None or fb is None:
                        missing.update(f"{nut}/{r}" for r, f in ((A, fa), (B, fb)) if f is None)
                        continue
                    r = dict(table=fam, nutrient=nut, metric=metric, A=LABEL[A], B=LABEL[B],
                             in_family=(A, B) in pairs, source_A=sa, source_B=sb,
                             **corrected_test(fa, fb, metric, nut))
                    block.append(r)
                fam_rows = [r for r in block if r["in_family"]]
                if fam_rows:
                    for r, q in zip(fam_rows, bh([r["p"] for r in fam_rows])):
                        r["q_BH"] = q
                rows += block
    out = pd.DataFrame(rows)
    csv = a.csv or os.path.join(a.out_root, "paired_tests.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv)), exist_ok=True)
    out.to_csv(csv, index=False)
    if missing:
        print("runs not found (skipped):", ", ".join(sorted(missing)))
    if len(out):
        show = out[["table", "nutrient", "metric", "A", "B", "delta", "ci_lo", "ci_hi", "p", "q_BH"]]
        with pd.option_context("display.width", 220, "display.max_rows", 500):
            print(show.round(2).to_string(index=False))
    print("wrote", csv)


if __name__ == "__main__":
    main()
