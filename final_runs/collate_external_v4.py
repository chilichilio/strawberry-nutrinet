#!/usr/bin/env python3
"""
collate_external_v4.py: Table 4 for the v4 external-season runs.
Per run in outputs/K_ext2526/<run>/:
  zero-label transfer   external_<run>_transfer.json   (R2, RMSE, Pearson r, bias, slope)
  label-informed recal  external_<run>_predictions_recalibrated.csv, written by base pipeline
                        recalibrate_external.py (grouped 5-fold cross-calibration;
                        columns y_pred_offset_recal = bias-corrected, y_pred_linear_recal)
(The base pipeline summary CSV labels every M7 variant "M7", so it is not used here.)
    python collate_external_v4.py [--root outputs/K_ext2526]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def r2_rmse(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return np.nan, np.nan
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), float(np.sqrt(np.mean((y - p) ** 2)))


ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(
    os.environ.get("NUTRINET_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")),
    "K_ext2526"))
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.root, "*", "external_*_transfer.json"))):
    run = re.search(r"external_(.+)_transfer\.json$", os.path.basename(f)).group(1)
    m = json.load(open(f))
    row = {"run": run, "n": m.get("n"), "pearson_r": m.get("pearson_r"),
           "slope": m.get("slope_measured_on_pred"), "raw_R2": m.get("R2"),
           "raw_RMSE": m.get("RMSE"), "raw_bias": m.get("bias")}
    rc = os.path.join(os.path.dirname(f), f"external_{run}_predictions_recalibrated.csv")
    if os.path.exists(rc):
        d = pd.read_csv(rc)
        y = d["y_true_reg"].to_numpy(float)
        for form, lab in (("offset", "bias_corrected"), ("linear", "linear_recal")):
            c = f"y_pred_{form}_recal"
            if c in d:
                row[f"{lab}_R2"], row[f"{lab}_RMSE"] = r2_rmse(y, d[c].to_numpy(float))
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(a.root, "table4_v4.csv"), index=False)
print(T.round(3).to_string(index=False))
