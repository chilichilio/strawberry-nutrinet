#!/usr/bin/env python3
"""
explain_v4.py: explanations for the final M7 (Figure 7, Supplementary Figure S7, Tables S8-S9) with
the base pipeline explainer chain, unchanged:
    M7_explainability.py           IG (images, DEM, static, fertilizer and weather
                                   sequences), SHAP (static branch), attention, bands
    M7_explainability_stratified   the 50-sample status-balanced explanation set
    M7_explainability_fixed        IG baseline fix, signed Grad-CAM, per-element shares,
                                   IG completeness check
    M7_explainability_shapfix      frozen parameters during SHAP (no OOM)
plus three changes, all applied in-process:

 1. The model and data are the NEW M7 (v4 TCN fix, sampling-date inputs, corrected
    weather summaries, the N image window), loaded from
    outputs/<nut>/M7__<tag>/model_M7_Transformer_reg_fold<k>.pt (run_dl_v4.py
    --save-checkpoints).
 2. The explainer's inputs now match training exactly. base pipeline recomputed the fold
    statistics from ALL outer-training rows and z-scored the weather over the whole
    season. Training used the fitting subset after the early-stopping split and
    fold-wise fertilizer/weather scalers (dl_core._fold_seq_norm). Both are rebuilt here.
 3. SHAP background = the 'typical plot' set: the 100
    training-fold samples closest to the training median of the (standardized) static
    covariates. base pipeline used 10 random samples from the validation fold. A SHAP value
    now reads as how a covariate moves the prediction away from a typical training plot.

Run (GPU, one job per nutrient; see run_all.sh for the environment variables):
    python explain_v4.py --nutrient N --seqfix --sampEnd
Output: outputs/<nut>/M7__<tag>/explain_v4/
Any other argument is passed to M7_explainability.py (e.g. --fold 3, --skip_ig).
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

import v4_common as v4


def main():
    v4.use_fork_workers()
    ap = argparse.ArgumentParser(add_help=False)
    v4.add_variant_args(ap)
    ap.add_argument("--bg-central", type=int, default=v4.N_BACKGROUND,
                    help="size of the typical-plot SHAP background (default 100)")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="default: outputs/<nut>/M7__<tag>")
    args, passthrough = ap.parse_known_args()
    v = v4.Variant.from_args(args)
    if v.noUAV or v.noF or v.lstm or v.noTemp or v.nested or v.treat:
        raise SystemExit("explain_v4 explains the full M7 (image + sequence model) only")
    nut = args.nutrient

    dp, _, dc = v4.setup(nut, need_dl=True)
    v4.patch_load_samples(dp, nut, v)
    v4.patch_dl_samples(dc, dp, nut, v.single)
    if v.seqfix:
        v4.patch_seqfix(dc)

    ckpt = args.checkpoint_dir or os.path.join(v4.OUT_ROOT, nut, f"M7__{v.tag()}")
    if not os.path.isdir(ckpt):
        raise SystemExit(f"checkpoint dir not found: {ckpt}")
    out = os.path.join(ckpt, "explain_v4")
    os.makedirs(out, exist_ok=True)

    # explainer loads weather once for all folds; keep it RAW so each fold can be
    # normalized exactly as in training (patched build_fold_stats below)
    _orig_lw = dc.load_weather_array

    def _raw_weather(path, *a, **k):
        k["normalize"] = False
        return _orig_lw(path, *a, **k)
    dc.load_weather_array = _raw_weather

    # base pipeline explainer chain (imports patch `mod` in place; fixed reads its flags from argv)
    sys.argv = [sys.argv[0]] + passthrough
    mod = importlib.import_module("M7_explainability")
    importlib.import_module("M7_explainability_fixed")      # also imports _stratified
    importlib.import_module("M7_explainability_shapfix")

    # ── 2. fold statistics exactly as in training ──
    _orig_bfs = mod.build_fold_stats

    def build_fold_stats_v4(df_s, tr_idx, feature_cols, fert_lookup, fert_mean, fert_std,
                            weather_array, weather_transplant, crop_hw):
        tr_idx = np.asarray(tr_idx)
        fit_idx, _ = dc._inner_earlystop_split(df_s, tr_idx)        # = run_cv
        st = _orig_bfs(df_s, fit_idx, feature_cols, fert_lookup, fert_mean, fert_std,
                       weather_array, weather_transplant, crop_hw)
        fm, fs, wn = dc._fold_seq_norm(df_s, fit_idx, fert_lookup, weather_array)
        st.update(fert_mean=fm, fert_std=fs, weather_array=wn)
        st["_train_idx"] = tr_idx
        return st
    mod.build_fold_stats = build_fold_stats_v4

    # ── 3. typical-plot SHAP background from the training fold ──
    BG = {"feats": None}
    bg_log = []
    _prev_run = mod.run_explainability_for_fold

    def run_fold_v4(model, df_s, va_idx, stats, device, fold_id, task, base_dir, a,
                    explain_idx=None):
        cols = stats["feature_cols"]
        tr = stats["_train_idx"]
        X = np.nan_to_num(df_s.iloc[tr][cols].apply(pd.to_numeric, errors="coerce")
                          .astype(np.float32).values)
        Z = np.nan_to_num((X - stats["feat_mean"]) / stats["feat_std"])   # = model input scaling
        bi = v4.central_background(Z, args.bg_central)
        BG["feats"] = Z[bi].astype(np.float32)
        b = df_s.iloc[tr[bi]][["Image_ID", "conc_date", "value"]].copy()
        b["fold"] = fold_id
        bg_log.append(b)
        print(f"[v4 explain] fold {fold_id}: SHAP background = {len(bi)} typical training plots",
              flush=True)
        return _prev_run(model, df_s, va_idx, stats, device, fold_id, task, base_dir, a,
                         explain_idx=explain_idx)
    mod.run_explainability_for_fold = run_fold_v4

    _prev_shap = mod.run_shap_tabular

    def run_shap_tabular_v4(model, device, x_ortho, x_dem, seq_len, bg_feats_np, x_feat_np,
                            *a, **k):
        return _prev_shap(model, device, x_ortho, x_dem, seq_len, BG["feats"], x_feat_np, *a, **k)
    mod.run_shap_tabular = run_shap_tabular_v4

    # ── run ──
    drv = importlib.import_module("M7_Transformer")
    cfg = drv.CFG
    cli = ["--checkpoint_dir", ckpt, "--output_dir", out, "--task", "reg",
           "--root_images", cfg.root_images, "--excel_features_path", cfg.excel_features_path,
           "--fert_path", cfg.fert_path, "--weather_path", cfg.weather_path,
           "--explain_select", "balanced", "--explain_n_total", "50",
           "--ig_steps", "32", "--bg_n", str(args.bg_central), "--max_dates_viz", "99",
           "--skip_occlusion"]
    given = {x for x in sys.argv[1:] if x.startswith("--")}
    i = 0
    while i < len(cli):                         # passthrough options override the defaults above
        if cli[i] in given:
            del cli[i:i + 2]
        else:
            i += 2
    sys.argv = [sys.argv[0]] + cli + sys.argv[1:]
    with open(os.path.join(out, "v4_explain_config.json"), "w") as f:
        json.dump({"nutrient": nut, "variant": v.tag(), "checkpoint_dir": ckpt,
                   "bg_central": args.bg_central, "argv": sys.argv[1:],
                   "EXPLAIN_TAKE_ALL_BELOW": os.environ.get("EXPLAIN_TAKE_ALL_BELOW")}, f, indent=2)
    print(f"[v4 explain] {nut} {v.tag()} -> {out}", flush=True)
    if args.dry_run:
        return
    try:
        mod.main()
    finally:
        if bg_log:
            pd.concat(bg_log).to_csv(os.path.join(out, "shap_background_samples.csv"), index=False)


if __name__ == "__main__":
    main()
