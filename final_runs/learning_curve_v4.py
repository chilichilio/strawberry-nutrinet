#!/usr/bin/env python
"""
learning_curve_v4.py: calibration learning curve for cross-season K recalibration
(Supplementary Figure S10), standalone replacement for the missing
code_25_26_external/recalibrate_learning_curve.py.

For every external run in outputs/K_ext2526/<run>/external_<run>_predictions.csv
(columns Image_ID, y_true_reg, y_pred_reg):
  for each calibration size k (number of PLOTS) and each of R random draws:
    * draw k plots from the 2025-26 plots -> calibration set (all their sampling dates)
    * fit on the calibration set
        offset : y = p + b             (b = mean(y - p))
        linear : y = a + s * p         (OLS)
    * apply to the remaining plots only and score R2 / RMSE there
      (raw transfer is scored on the same held-out plots for reference)
The same plot draws are used for every run (seeded per draw), so runs are compared
on identical calibration sets. Reads nothing but the prediction files; writes only
to --out-dir.

Outputs
  recalibration_learning_curve_draws.csv   one row per run x k x draw x method
  recalibration_learning_curve.csv         mean, SD, 2.5/97.5 percentiles per run x k x method
  recalibration_learning_curve.png         optional plot (--runs selects the lines)
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def find_runs(root):
    """run name -> prediction file. Only per-run names (contain '__'), which skips the
    base pipeline-style duplicates such as external_M5_predictions.csv."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "external_*_predictions.csv"))):
        m = re.match(r"external_(.+)_predictions\.csv$", os.path.basename(f))
        if m and "__" in m.group(1):
            runs[m.group(1)] = f
    return runs


def load(f):
    d = pd.read_csv(f)
    need = {"Image_ID", "y_true_reg", "y_pred_reg"}
    miss = need - set(d.columns)
    if miss:
        raise KeyError(f"{f}: missing columns {sorted(miss)}")
    d = d[np.isfinite(d["y_true_reg"]) & np.isfinite(d["y_pred_reg"])]
    return d[["Image_ID", "y_true_reg", "y_pred_reg"]].reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="outputs/K_ext2526")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(range(5, 45, 5)))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None, help="default: --root")
    ap.add_argument("--runs", nargs="*", default=["M4__sampEnd", "M7__seqfix_sampEnd"],
                    help="runs drawn in the PNG (all runs go into the CSVs)")
    a = ap.parse_args(argv)
    out = a.out_dir or a.root
    os.makedirs(out, exist_ok=True)

    runs = find_runs(a.root)
    if not runs:
        raise SystemExit(f"no external_*__*_predictions.csv under {a.root}")
    data = {r: load(f) for r, f in runs.items()}

    # identical plot universe across runs (they should all be the same 180 plots)
    plot_sets = {r: set(d["Image_ID"]) for r, d in data.items()}
    common = sorted(set.intersection(*plot_sets.values()))
    for r, s in plot_sets.items():
        if s != set(common):
            print(f"[warn] {r}: {len(s)} plots, {len(common)} in common; using the common set")
    plots = np.array(common)
    data = {r: d[d["Image_ID"].isin(common)].reset_index(drop=True) for r, d in data.items()}
    print(f"[lc] {len(runs)} runs, {len(plots)} plots, sizes {a.sizes}, {a.reps} draws")

    rows = []
    for k in a.sizes:
        if k >= len(plots):
            print(f"[skip] k={k} >= number of plots")
            continue
        for rep in range(a.reps):
            rng = np.random.default_rng([a.seed, k, rep])      # same draw for every run
            cal = set(rng.choice(plots, size=k, replace=False))
            for run, d in data.items():
                is_cal = d["Image_ID"].isin(cal).to_numpy()
                y, p = d["y_true_reg"].to_numpy(), d["y_pred_reg"].to_numpy()
                y_c, p_c, y_t, p_t = y[is_cal], p[is_cal], y[~is_cal], p[~is_cal]
                b = float(np.mean(y_c - p_c))
                if len(p_c) >= 2 and np.ptp(p_c) > 0:
                    s, a0 = np.polyfit(p_c, y_c, 1)
                else:                                          # degenerate: fall back to offset
                    s, a0 = 1.0, b
                for meth, pt in (("raw", p_t), ("offset", p_t + b), ("linear", a0 + s * p_t)):
                    rows.append({"run": run, "n_cal_plots": k, "draw": rep, "method": meth,
                                 "n_cal_obs": int(is_cal.sum()), "n_test_obs": int((~is_cal).sum()),
                                 "R2": r2(y_t, pt), "RMSE": rmse(y_t, pt),
                                 "offset": b if meth == "offset" else np.nan,
                                 "slope": s if meth == "linear" else np.nan})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "recalibration_learning_curve_draws.csv"), index=False)

    S = (D.groupby(["run", "method", "n_cal_plots"])
          .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
               R2_p025=("R2", lambda x: np.nanpercentile(x, 2.5)),
               R2_p975=("R2", lambda x: np.nanpercentile(x, 97.5)),
               RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
               n_draws=("R2", "size"))
          .reset_index())
    S.to_csv(os.path.join(out, "recalibration_learning_curve.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        show = S[S["method"] != "raw"]
        print(show.pivot_table(index=["run", "method"], columns="n_cal_plots",
                               values="R2_mean").round(3))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        styles = ["-", "--", ":", "-."]
        colors = {"offset": "tab:blue", "linear": "tab:orange"}
        for i, run in enumerate([r for r in a.runs if r in data]):
            for meth in ("offset", "linear"):
                s_ = S[(S["run"] == run) & (S["method"] == meth)]
                ax.plot(s_["n_cal_plots"], s_["R2_mean"], styles[i % 4], color=colors[meth],
                        label=f"{run.split('__')[0]} {meth}")
                ax.fill_between(s_["n_cal_plots"], s_["R2_mean"] - s_["R2_sd"],
                                s_["R2_mean"] + s_["R2_sd"], color=colors[meth], alpha=0.12)
        ax.set_xlabel("Calibration plots (2025-26)")
        ax.set_ylabel("R² on held-out plots")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "recalibration_learning_curve.png"), dpi=300)
    except Exception as e:                                      # plotting is optional
        print(f"[lc] plot skipped: {e}")
    print(f"[lc] wrote {out}/recalibration_learning_curve*.csv")


if __name__ == "__main__":
    main()
