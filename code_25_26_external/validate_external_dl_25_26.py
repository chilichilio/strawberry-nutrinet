#!/usr/bin/env python3
# ============================================================
# validate_external_dl_25_26.py
# ------------------------------------------------------------
# INDEPENDENT-SEASON (external) validation of a DEEP-LEARNING 24_25 leaf-K (%)
# model (M5 CNN-fusion / M7 temporal Transformer) on the 2025-2026 season.
#
# This is the deep-learning counterpart to validate_external_25_26.py (which
# covers the classical M1..M4). Your best 24_25 K model was M7 (R2=0.82), so
# this is the one you'll most likely want.
#
# WHAT IT DOES
#   1. Trains the chosen DL config on 2024-2025 (reusing dl_core.train_one_fold
#      exactly as the in-season models do — same backbones, same normalization,
#      fit on 24_25 only, early-stopping on an internal 24_25 hold-out split).
#   2. Runs inference on 2025-2026 with dl_core._infer_val, but with the
#      2025-2026 image tiles + (for M7) the 2025-2026 daily fertilizer/weather
#      sequences swapped into the stats dict. All *normalization* statistics
#      (image band stats, DEM stats, scalar feat mean/std, target y mean/std,
#      crop size) stay fixed from the 24_25 training fold — i.e. train-fit
#      normalization applied to the held-out season, which is the correct
#      external-validation protocol.
#   3. Writes regression + stage-aware classification metrics with per-stage
#      and per-experiment (SB2/SB3/SB4) breakdowns, plus per-sample predictions
#      — same output shape as the classical harness.
#
# WHY A SEPARATE INFERENCE STATS DICT (important)
#   The fertilizer lookup in dl_core is keyed by Image_ID, and 25_26 reuses the
#   SAME plot IDs as 24_25 (SB2_01, ...). A single train_one_fold call therefore
#   cannot carry both seasons' sequences without key collisions. Training and
#   inference are run as two separate steps so each season uses its own image
#   root and its own fert/weather sources, with zero collision. This does NOT
#   modify dl_core.py.
#
# WHAT YOU MUST EDIT
#   The two path blocks below: TRAIN (24_25) and TEST (25_26). The 24_25 block
#   should match the paths already in code_24_25_K/M7_Transformer.py. The
#   25_26 block points at the processed 25_26 UAV tiles + the 25_26 daily
#   fert/weather files. Only the UAV tile roots need a real path (see below).
#
# RUN (single GPU):   python validate_external_dl_25_26.py --config M7
#      (DDP is unnecessary here: one train + one inference pass, no CV loop.)
#
# Requires the same environment as M5/M7 (torch, torchvision, dl_core deps).
# ============================================================
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Import the UNMODIFIED 24_25 DL code (dl_core.py + data_prep.py).
# ─────────────────────────────────────────────────────────────
def _import_dl(code_dir: str):
    code_dir = os.path.abspath(code_dir)
    if not os.path.isdir(code_dir):
        raise SystemExit(f"[FATAL] --code-dir not found: {code_dir}")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import dl_core as dc          # noqa: E402
    import data_prep as dp        # noqa: E402
    return dc, dp


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

# ─────────────────────────────────────────────────────────────
# ============  TRAIN / TEST INPUTS  ============
# TRAIN season = 2024-2025 (should match code_24_25_K/M7_Transformer.py)
# ─────────────────────────────────────────────────────────────
TRAIN = dict(
    root_images=os.path.join(BASE, "data", "24_25_K"),   # season folder of the data deposit (tables + UAV tiles)
    excel_features_path=os.path.join(BASE, "data/24_25_K/trait_K_weather_yield_combined_conc.xlsx"),
    fert_path=os.path.join(BASE, "data/24_25_K/fert_by_plot_daily_lb2kg.xlsx"),
    weather_path=os.path.join(BASE, "data/24_25_K/FAWN_report_metric_24_25.xlsx"),
)

# TEST season = 2025-2026 (processed UAV tiles + 25_26 daily sequence files)
TEST = dict(
    root_images=os.path.join(BASE, "data", "25_26_K"),   # season folder of the data deposit (tables + UAV tiles)
    excel_features_path=os.path.join(BASE, "data/25_26_K/trait_K_weather_yield_combined_conc.xlsx"),
    fert_path=os.path.join(BASE, "data/25_26_K/fert_by_plot_daily_lb2kg.xlsx"),
    weather_path=os.path.join(BASE, "data/25_26_K/FAWN_report_metric_25_26.xlsx"),
)
ORTHO_FILTER = "ortho_by_plot"
DEM_FILTER = "dem_by_plot"
# ================================================================


def _make_cfg(dc, model: str, out_dir: str):
    """DLRunConfig mirroring code_24_25_K/M5 & M7, but single-pass."""
    if model == "M5":
        return dc.DLRunConfig(
            model_tag="M5_CNN_fusion_ext2526", config_name="M5",
            image_config="closest_date", use_image_temporal=False,
            use_sequences=False, task_modes=("reg",),
            root_images=TRAIN["root_images"], ortho_filter=ORTHO_FILTER,
            dem_filter=DEM_FILTER, excel_features_path=TRAIN["excel_features_path"],
            output_dir=out_dir, epochs=80, batch_size=4,
            ortho_backbone="resnet18", dem_backbone="convnext_tiny",
            freeze_backbone_epochs=5, use_amp=True)
    if model == "M7":
        return dc.DLRunConfig(
            model_tag="M7_Transformer_ext2526", config_name="M7",
            image_config="on_before_closest", use_image_temporal=True,
            use_sequences=True, task_modes=("reg",),
            root_images=TRAIN["root_images"], ortho_filter=ORTHO_FILTER,
            dem_filter=DEM_FILTER, excel_features_path=TRAIN["excel_features_path"],
            fert_path=TRAIN["fert_path"], weather_path=TRAIN["weather_path"],
            output_dir=out_dir, epochs=80, batch_size=2,
            ortho_backbone="resnet18", dem_backbone="convnext_tiny",
            freeze_backbone_epochs=5, use_amp=True,
            num_workers=2, persistent_workers=False, pin_memory=False)
    raise SystemExit(f"[UNSUPPORTED] --config {model}. This DL harness handles "
                     f"M5 or M7. (M6 = RF + Eseq: use the classical harness with "
                     f"a train-fold TCN encoder, or ask for an M6 driver.)")


def _build_samples(dc, dp, cfg, root_images, excel_path):
    """Scan a season's image tiles + labeled table -> the df_s sample frame
    (one row per unique conc-date target, with ortho_paths/dem_paths lists),
    exactly as dl_core.run_cv builds it."""
    df_imgs = dc.find_all_paired_tifs(root_images, cfg.ortho_filter, cfg.dem_filter)
    df_imgs["Image_ID"] = (df_imgs["exp_ID"].astype(str) + "_"
                           + df_imgs["plot_num"].astype(str))
    df_all = dp.load_samples(excel_path, require_target=True)
    df_all["Image_ID"] = df_all["Image_ID"].astype(str)

    if cfg.image_config == "closest_date":
        df_s = dc.build_closest_date_samples(df_imgs, df_all)
    elif cfg.image_config == "on_before_closest":
        df_s = dc.build_on_before_closest_samples(df_imgs, df_all, dc.MIN_IMAGES_PER_SAMPLE)
    else:
        df_s = dc.build_all_dates_samples(df_imgs, df_all, dc.MIN_IMAGES_PER_SAMPLE)

    # NaN-safe: coerce target first so missing-value placeholders (e.g. the
    # string ".") become real NaN and get dropped, instead of surviving
    # notna() and crashing the float() cast
    df_s[dc.TARGET_COL] = pd.to_numeric(df_s[dc.TARGET_COL], errors="coerce")
    df_s = df_s[df_s[dc.TARGET_COL].notna()].copy()
    df_s["value"] = df_s[dc.TARGET_COL].astype(float)
    if "sampling_stage" not in df_s.columns:
        df_s["sampling_stage"] = [dc.infer_sampling_stage(dc._safe_to_date(r["conc_date"]))
                                  for _, r in df_s.iterrows()]

    # ── SEASON-RELATIVE DAP (critical for cross-season validation) ──
    # The source tables store `day` as a GLOBAL running counter from the
    # 2023-10-16 season start (24_25 -> 381..507, 25_26 -> 743..864), so the
    # M7 timing scalar and the fert/weather sequence-slice length (dl_core
    # line ~768: fert_lookup[iid][:dap], weather_array[:dap+1]) are on
    # incompatible scales across seasons -> the model extrapolates far outside
    # its training range and the sequences are sliced with the wrong window.
    # Recompute `day` as days since EACH season's OWN transplant date (via the
    # season-aware dc._transplant_of), overwriting the global counter. Applied
    # identically to the 24_25 train frame and the 25_26 test frame so both sit
    # on the same 0..~150 DAP scale. Does not modify dl_core.py.
    def _season_dap(cd):
        d = dc._safe_to_date(cd)
        return int((d - dc._transplant_of(d)).days) if d is not None else 0
    df_s["day"] = [_season_dap(cd) for cd in df_s["conc_date"]]
    return df_imgs, df_all, df_s


def _seq_sources(dc, cfg, fert_path, weather_path, raw_weather=True):
    """Load (fert_lookup, fert_mean, fert_std, weather_array, weather_transplant)
    for one season. Returns Nones if use_sequences is False or a file is missing
    (TCN then sees zeros).

    Leakage control: with raw_weather=True the weather array is returned UNNORMALIZED, so
    the caller can apply a single TRAIN-SEASON scaler to both seasons (see main).
    The old behavior — each season z-scored by its own full-season statistics —
    is available with raw_weather=False (`--weather-norm season`).

    The pooled `fert_mean/fert_std` returned here are only used for the TRAIN
    season; main() keeps the train values and never adopts the test season's.
    """
    fert_lookup, fert_mean, fert_std = None, 0.0, 1.0
    weather_array, weather_transplant = None, None
    if not cfg.use_sequences:
        return fert_lookup, fert_mean, fert_std, weather_array, weather_transplant
    try:
        fert_lookup, _ = dc.load_fert_lookup(fert_path)
        allv = np.concatenate([v for v in fert_lookup.values()])
        allv = allv[allv > 0]
        if allv.size:
            fert_mean, fert_std = float(allv.mean()), max(float(allv.std()), 1e-6)
    except Exception as e:
        print(f"  WARN fert seq unavailable ({e}); fert TCN uses zeros.")
    try:
        weather_array, weather_transplant = dc.load_weather_array(
            weather_path, normalize=(not raw_weather))
    except Exception as e:
        print(f"  WARN weather seq unavailable ({e}); weather TCN uses zeros.")
    return fert_lookup, fert_mean, fert_std, weather_array, weather_transplant


def _subset_metrics(dp, y_true, y_pred, stages, mask) -> Dict:
    yt = np.asarray(y_true)[mask]; yp = np.asarray(y_pred)[mask]
    st = [s for s, m in zip(stages, mask) if m]
    if len(yt) == 0:
        return {"n": 0}
    out = {"n": int(len(yt))}
    out.update(dp.regression_metrics(yt, yp))
    try:
        _, _, cls = dp.stage_aware_classification(yt, yp, st)
        out["classification"] = cls
    except Exception:
        pass
    return out


def main():
    import torch
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="M7", help="M5 or M7 (default M7).")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--code-dir", default=os.path.join(here, "..", "code_24_25_K"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--val-frac-groups", type=float, default=0.15,
                    help="Fraction of 24_25 PLOTS held out inside training for "
                         "early stopping (GroupKFold-based; default ~0.15).")
    ap.add_argument("--weather-norm", choices=("train", "season"), default="train",
                    help="Leakage control. 'train' (default): fit ONE weather scaler on the "
                         "2024-25 training rows (days <= latest training sampling "
                         "date) and apply it to BOTH seasons — matches the Methods "
                         "claim that all preprocessing was estimated from 2024-25 "
                         "only. 'season': previous behavior, each season z-scored by "
                         "its own full-season statistics (kept for comparison).")
    args = ap.parse_args()

    dc, dp = _import_dl(args.code_dir)
    model_name = args.config.upper()
    out_dir = args.output_dir or os.path.join(here, "..", "outputs_25_26", "v3",
                                              f"external_validation_{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = _make_cfg(dc, model_name, out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dc.seed_everything(dc.GLOBAL_SEED)
    print(f"[ext-DL] device={device}  config={model_name}")

    # ── 1) TRAIN-season samples + feature cols + sequence sources ──
    print("[ext-DL] scanning 24_25 (train) ...")
    df_imgs_tr, df_all_tr, df_s_tr = _build_samples(
        dc, dp, cfg, TRAIN["root_images"], TRAIN["excel_features_path"])
    groups = cfg.scalar_groups or dp.MODEL_CONFIGS[cfg.config_name].scalar_groups
    feature_cols = [c for c in dp.get_scalar_feature_columns(df_all_tr, groups)
                    if c in df_s_tr.columns]
    print(f"  train samples: {len(df_s_tr)}  scalar features: {len(feature_cols)}")

    _raw_w = (args.weather_norm == "train")
    ft_tr = _seq_sources(dc, cfg, TRAIN["fert_path"], TRAIN["weather_path"],
                         raw_weather=_raw_w)
    fert_lk_tr, fert_mu, fert_sd, wx_tr, wt_tr = ft_tr

    # class weights (inverse frequency), same as run_cv
    vc = df_all_tr[df_all_tr["K_label_idx"] >= 0]["K_label_idx"].value_counts().sort_index()
    counts = np.array([vc.get(i, 1) for i in range(dc.N_CLASSES)], dtype=np.float32)
    cls_weights = torch.from_numpy(1.0 / (counts / counts.sum())).float()
    cls_weights = cls_weights / cls_weights.sum() * dc.N_CLASSES

    # ── 2) TEST-season samples + its OWN sequence sources ──
    print("[ext-DL] scanning 25_26 (test) ...")
    df_imgs_te, df_all_te, df_s_te = _build_samples(
        dc, dp, cfg, TEST["root_images"], TEST["excel_features_path"])
    # align scalar feature columns to the TRAIN set (missing cv_* -> 0)
    for c in feature_cols:
        if c not in df_s_te.columns:
            df_s_te[c] = 0.0
    print(f"  test samples: {len(df_s_te)}")

    ft_te = _seq_sources(dc, cfg, TEST["fert_path"], TEST["weather_path"],
                         raw_weather=_raw_w)
    fert_lk_te, _, _, wx_te, wt_te = ft_te
    # NOTE: the test season's fert_mean/fert_std are deliberately discarded —
    # fertilizer normalization stays train-fit (this was already correct).

    # ── 3) crop size over the UNION of both seasons' tiles ──
    crop_hw = dc.compute_global_crop_hw(
        pd.concat([df_imgs_tr, df_imgs_te], ignore_index=True), "ortho_path")
    if cfg.crop_cap:
        crop_hw = (min(crop_hw[0], cfg.crop_cap), min(crop_hw[1], cfg.crop_cap))
    print(f"  crop_hw={crop_hw}")

    # ── 4) internal 24_25 train/val split for early stopping ──
    gkf = GroupKFold(n_splits=max(2, int(round(1.0 / max(args.val_frac_groups, 1e-3)))))
    tr_idx, va_idx = next(iter(gkf.split(np.arange(len(df_s_tr)),
                                         groups=df_s_tr["Image_ID"].values)))
    print(f"  training on {len(tr_idx)} / early-stop val {len(va_idx)} (24_25 samples)")

    # ── 4b) Leakage control: ONE weather scaler, fit on the 24_25 TRAINING rows only ──
    # Fit over days <= the latest training sampling date (no post-sampling days),
    # then apply the SAME stats to both the 24_25 and 25_26 raw arrays. This makes
    # the code match the Methods claim that every preprocessing parameter was
    # estimated from 2024-25 data only. With --weather-norm season, both arrays
    # arrive already z-scored per season (previous behavior) and we skip this.
    if args.weather_norm == "train" and wx_tr is not None:
        w_mu, w_sd = dc.weather_stats_from_rows(df_s_tr, tr_idx, wx_tr)
        wx_tr = dc.apply_weather_norm(wx_tr, w_mu, w_sd)
        if wx_te is not None:
            wx_te = dc.apply_weather_norm(wx_te, w_mu, w_sd)
        print("  [leakage] weather scaler fit on 24_25 TRAIN rows; applied to both seasons")
    else:
        print(f"  [leakage] weather-norm='{args.weather_norm}' "
              f"(each season z-scored by its own full-season stats)")

    # ── 5) train on 24_25 (reuse dl_core.train_one_fold, unmodified) ──
    model, stats, mreg, mcls = dc.train_one_fold(
        cfg, 0, df_s_tr, tr_idx, va_idx, device, crop_hw, feature_cols, "reg",
        cls_weights=cls_weights, distributed=False, rank=0, world_size=1, local_rank=0,
        fert_lookup=fert_lk_tr, fert_mean=fert_mu, fert_std=fert_sd,
        weather_array=wx_tr, weather_transplant=wt_tr)
    torch.save({"state_dict": dc.unwrap_model(model).state_dict()},
               os.path.join(out_dir, f"model_{model_name}_ext2526_reg.pt"))
    print(f"  [24_25 internal-val] reg metrics: {mreg}")

    # ── 6) inference on 25_26 with TEST-season image/seq sources swapped in,
    #        but TRAIN-fit normalization (means/stds/crop/y-scale) preserved ──
    stats_test = dict(stats)
    stats_test["fert_lookup"] = fert_lk_te if fert_lk_te is not None else stats["fert_lookup"]
    stats_test["weather_array"] = wx_te if wx_te is not None else stats["weather_array"]
    stats_test["weather_transplant"] = wt_te if wt_te is not None else stats["weather_transplant"]
    # fert_mean/fert_std kept from TRAIN (train-fit sequence normalization)

    y_true, y_pred = dc._infer_val(model, stats_test, df_s_te, cfg, device, "reg")
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    stages = df_s_te["sampling_stage"].tolist()[:len(y_true)]

    # ── 7) metrics: overall / per-stage / per-experiment ──
    overall = dict(dp.regression_metrics(y_true, y_pred))
    _, _, overall_cls = dp.stage_aware_classification(y_true, y_pred, stages)
    per_stage = {}
    for stg in sorted({s for s in stages if s is not None}):
        per_stage[stg] = _subset_metrics(dp, y_true, y_pred, stages,
                                         [s == stg for s in stages])
    exp = df_s_te["Image_ID"].astype(str).str.split("_").str[0].values[:len(y_true)]
    per_exp = {}
    for e in sorted(set(exp)):
        per_exp[e] = _subset_metrics(dp, y_true, y_pred, stages, list(exp == e))

    # ── 8) write predictions + metrics ──
    meta = df_s_te.iloc[:len(y_true)][
        [c for c in ["Image_ID", "conc_date", "day", "sampling_stage", dc.TARGET_COL]
         if c in df_s_te.columns]].copy().reset_index(drop=True)
    meta["y_true_reg"] = y_true
    meta["y_pred_reg"] = y_pred
    meta["abs_error"] = np.abs(y_true - y_pred)
    pred_path = os.path.join(out_dir, f"external_{model_name}_predictions.csv")
    meta.to_csv(pred_path, index=False)

    results = {
        "kind": "external_season_validation_DL",
        "train_season": "2024_2025", "test_season": "2025_2026",
        "config_name": model_name, "estimator": cfg.model_tag,
        "n_train_samples": int(len(df_s_tr)), "n_test_samples": int(len(y_true)),
        "internal_val_metrics_24_25": mreg,
        "overall_regression": overall, "overall_classification": overall_cls,
        "per_stage": per_stage, "per_experiment": per_exp,
        "feature_columns": feature_cols, "crop_hw": list(crop_hw),
        "weather_norm": args.weather_norm,   # Leakage control: 'train' | 'season'
    }
    metrics_path = os.path.join(out_dir, f"external_{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print(f"EXTERNAL VALIDATION (DL) | train 24_25 -> test 25_26 | {model_name}")
    print("=" * 64)
    print(f"Overall: R2={overall['R2']:.3f}  RMSE={overall['RMSE']:.3f}  "
          f"MAE={overall['MAE']:.3f}  sMAPE={overall['sMAPE']:.2f}  (n={len(y_true)})")
    if overall_cls:
        print(f"Stage-aware cls: Acc={overall_cls.get('Acc', float('nan')):.3f}  "
              f"BalAcc={overall_cls.get('BalAcc', float('nan')):.3f}  "
              f"F1m={overall_cls.get('F1_macro', float('nan')):.3f}")
    for stg, m in per_stage.items():
        if m.get("n", 0):
            print(f"  {stg:16s} n={m['n']:3d}  R2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}")
    for e, m in per_exp.items():
        if m.get("n", 0):
            print(f"  {e:6s} n={m['n']:3d}  R2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}")
    print(f"\nPredictions -> {pred_path}\nMetrics     -> {metrics_path}")


if __name__ == "__main__":
    main()
