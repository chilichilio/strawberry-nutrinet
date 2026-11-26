#!/usr/bin/env python3
"""
aggregate_seeds_v4.py: seed-to-seed variability of the new M7.

Seed 42 = the main run (outputs/<nut>/M7__<tag>); seeds 7, 13, 2024, 99 =
outputs/<nut>_multiseed/M7__<tag>__seed<s> (run_dl_v4.py --seed). Same folds for
every seed, so fold-to-fold and seed-to-seed spread can be compared directly
(manuscript: "Seed-to-seed variability was smaller than fold-to-fold variability").

    python aggregate_seeds_v4.py [--root outputs] [--tag seqfix_sampEnd]
Writes <root>/multiseed_summary.csv and prints a table.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def fold_metrics(d):
    f = glob.glob(os.path.join(d, "cv_results_*_reg.csv"))
    if not f:
        return None
    x = pd.read_csv(f[0])
    return x[pd.to_numeric(x["fold"], errors="coerce").notna()].astype({"fold": int})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("NUTRINET_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")))
    ap.add_argument("--model", default="M7")
    ap.add_argument("--tag", default="seqfix_sampEnd")
    a = ap.parse_args()
    rows, summ = [], []
    for nut in ("N", "K"):
        runs = {42: os.path.join(a.root, nut, f"{a.model}__{a.tag}")}
        for d in glob.glob(os.path.join(a.root, f"{nut}_multiseed", f"{a.model}__{a.tag}__seed*")):
            runs[int(re.search(r"seed(\d+)$", d).group(1))] = d
        per_seed = {}
        for s, d in sorted(runs.items()):
            fm = fold_metrics(d)
            if fm is None:
                print(f"[skip] {nut} seed {s}: no cv_results in {d}")
                continue
            per_seed[s] = fm.set_index("fold")
            rows.append({"nutrient": nut, "seed": s, **{k: fm[k].mean() for k in ("R2", "RMSE", "MAE")}})
        if not per_seed:
            continue
        R2 = pd.DataFrame({s: m["R2"] for s, m in per_seed.items()})     # folds x seeds
        seed_means = R2.mean(0)
        summ.append({"nutrient": nut, "n_seeds": R2.shape[1],
                     "R2_mean_over_seeds": seed_means.mean(),
                     "R2_SD_between_seeds": seed_means.std(ddof=1) if len(seed_means) > 1 else np.nan,
                     "R2_SD_between_folds_mean": R2.std(0, ddof=1).mean(),
                     "R2_min_seed": seed_means.min(), "R2_max_seed": seed_means.max()})
    pd.DataFrame(rows).to_csv(os.path.join(a.root, "multiseed_per_seed.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(a.root, "multiseed_summary.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print()
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
