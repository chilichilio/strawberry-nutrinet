#!/usr/bin/env python3
# ============================================================
# validate_external_25_26.py
# ------------------------------------------------------------
# INDEPENDENT-SEASON (external) validation of a 24_25 leaf-K (%) model
# on the 2025-2026 season.
#
# WHY THIS SCRIPT EXISTS
#   The manuscript currently reports only within-season 5-fold GroupKFold
#   cross-validation on the 2024-2025 data (Methods: "Training, validation,
#   and evaluation"). This script adds a stronger, complementary test:
#   train the chosen model on ALL of 2024-2025 and evaluate it, untouched,
#   on the 2025-2026 season as a held-out year. This measures temporal /
#   cross-season generalization, not just within-season CV.
#
# WHAT IT DOES NOT DO
#   It changes NOTHING in code_24_25_K/ or anywhere else. It imports the
#   existing data_prep.py + ml_runner.py as the single source of truth for
#   the target, feature groups, sample construction, stage thresholds, and
#   metrics, so the external-validation numbers are produced with exactly the
#   same definitions as the in-season results.
#
# SCOPE
#   Classical-ML configs only (M1 PLSR, M2 RF, M3 PLSR+agg, M4 RF+agg). These
#   are fully reproducible on any machine. M5/M7 (deep learning on image
#   tiles) and M6 (Eseq sequence-embedding RF) are NOT handled here because
#   they need the DL image/sequence pipeline (dl_core.py, GPU) and a separate
#   train-24_25 / infer-25_26 driver; the script errors with a clear message
#   if you request one of those.
#
# FEATURE / COLUMN PARITY
#   The feature column list is fixed by the TRAINING season (24_25). The test
#   season (25_26) is reindexed onto that exact column list so the model sees
#   the identical inputs it was trained on:
#     * missing one-hot cultivar columns (cv_*) -> filled with 0 (that
#       cultivar simply did not occur that plot-row);
#     * any other missing / NaN numeric feature -> imputed with the TRAIN
#       column mean, and the columns imputed are reported loudly, because a
#       heavily-imputed feature weakens the external-validation claim.
#
# USAGE
#   python validate_external_25_26.py --config M4
#   python validate_external_25_26.py --config M2 \
#       --train-excel ../data/24_25_K/trait_K_weather_yield_combined_conc.xlsx \
#       --test-excel  ../data/25_26_K/trait_K_weather_yield_combined_conc.xlsx \
#       --output-dir  ../outputs_25_26/external_validation_M2
#
# Requires: numpy, pandas, scikit-learn, openpyxl (same as the ML models).
# ============================================================
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Locate and import the EXISTING 24_25 model code (unmodified).
# By default we look for a sibling code_24_25_K/ folder; override with
# --code-dir if your layout differs.
# ─────────────────────────────────────────────────────────────
def _import_existing_modules(code_dir: str):
    code_dir = os.path.abspath(code_dir)
    if not os.path.isdir(code_dir):
        raise SystemExit(
            f"[FATAL] --code-dir not found: {code_dir}\n"
            f"        Point it at the folder containing data_prep.py + "
            f"ml_runner.py (normally code_24_25_K)."
        )
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import data_prep as dp        # noqa: E402
    import ml_runner as mlr       # noqa: E402
    return dp, mlr


def _clean_target_table(dp, src_path: str, out_dir: str, tag: str) -> str:
    """Return a path to a copy of `src_path` in which the target column (and
    the other nutrient concentration columns) are coerced to numeric, so stray
    non-numeric markers (e.g. a '.' cell) become NaN and are dropped by
    data_prep's require_target filter instead of crashing float() conversion.

    Non-invasive: reads the workbook, fixes only the concentration columns,
    writes a temp copy; the original file is never modified.
    """
    df = pd.read_excel(src_path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    conc_like = [dp.TARGET_COL] + [
        c for c in df.columns
        if c.endswith("(%)") or c.endswith("(ppm)")
    ]
    changed = 0
    for c in dict.fromkeys(conc_like):
        if c in df.columns:
            before = df[c].copy()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            changed += int((before.notna() & df[c].isna()).sum())
    if changed:
        print(f"  [clean] {tag}: coerced {changed} non-numeric concentration "
              f"cell(s) to NaN before loading.")
    cleaned = os.path.join(out_dir, f"_cleaned_{tag}.xlsx")
    df.to_excel(cleaned, index=False)
    return cleaned


# ─────────────────────────────────────────────────────────────
# Feature-matrix builder with train-fixed columns.
# ─────────────────────────────────────────────────────────────
def build_aligned_matrices(
    dp,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list]]:
    """Return (X_train, X_test) as float arrays over the SAME `feat_cols`,
    with a report of which test columns were missing / imputed.

    Rules:
      * cv_* one-hot columns absent from a season -> 0
      * any remaining NaN (train or test) -> TRAIN column mean
    """
    report = {"missing_in_test": [], "cv_filled_zero": [], "mean_imputed": []}

    # start from a numeric copy over the fixed training column set
    Xtr = train_df.reindex(columns=feat_cols)
    Xte = test_df.reindex(columns=feat_cols)

    for c in feat_cols:
        if c not in test_df.columns:
            report["missing_in_test"].append(c)

    Xtr = Xtr.apply(pd.to_numeric, errors="coerce")
    Xte = Xte.apply(pd.to_numeric, errors="coerce")

    # cultivar one-hots: NaN means "not this cultivar" -> 0
    cv_cols = [c for c in feat_cols if c.startswith("cv_")]
    for c in cv_cols:
        if Xte[c].isna().any():
            report["cv_filled_zero"].append(c)
        Xte[c] = Xte[c].fillna(0.0)
        Xtr[c] = Xtr[c].fillna(0.0)

    # everything else: impute with TRAIN mean (fit on train only)
    train_means = Xtr.mean(numeric_only=True)
    for c in feat_cols:
        if c in cv_cols:
            continue
        mu = train_means.get(c, np.nan)
        if pd.isna(mu):
            mu = 0.0  # column entirely NaN in train (degenerate) -> 0
        if Xte[c].isna().any():
            report["mean_imputed"].append(c)
        Xte[c] = Xte[c].fillna(mu)
        Xtr[c] = Xtr[c].fillna(mu)

    return (
        Xtr.astype(np.float64).values,
        Xte.astype(np.float64).values,
        report,
    )


# ─────────────────────────────────────────────────────────────
# Metrics helpers (delegates to data_prep for identical definitions).
# ─────────────────────────────────────────────────────────────
def _subset_metrics(dp, y_true, y_pred, stages, mask) -> Dict:
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]
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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="M4",
                    help="Which classical model config to validate: M1, M2, M3, or M4 "
                         "(default M4 = RF, V+S+C+P+T+Fagg+Wagg).")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--code-dir", default=os.path.join(here, "..", "code_24_25_K"),
                    help="Folder containing the unmodified data_prep.py + ml_runner.py.")
    ap.add_argument("--train-excel",
                    default=os.path.join(here, "..", "data", "24_25_K",
                                         "trait_K_weather_yield_combined_conc.xlsx"),
                    help="24_25 combined modeling table (training season).")
    ap.add_argument("--test-excel",
                    default=os.path.join(here, "..", "data", "25_26_K",
                                         "trait_K_weather_yield_combined_conc.xlsx"),
                    help="25_26 combined modeling table (held-out season).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write predictions + metrics (default: "
                         "../25_26/external_validation_<CONFIG>).")
    args = ap.parse_args()

    dp, mlr = _import_existing_modules(args.code_dir)

    cfg_name = args.config.upper()
    if cfg_name not in dp.MODEL_CONFIGS:
        raise SystemExit(f"Unknown config '{cfg_name}'. Choose from "
                         f"{sorted(dp.MODEL_CONFIGS)}.")
    cfg = dp.MODEL_CONFIGS[cfg_name]
    if cfg.kind != "ml" or cfg.image or cfg.sequence:
        raise SystemExit(
            f"[UNSUPPORTED] {cfg_name} is a '{cfg.model}' model "
            f"(image={cfg.image}, sequence={cfg.sequence}). This harness "
            f"handles the fixed-length classical configs only (M1..M4). "
            f"For M5/M7 (image tiles) or M6 (Eseq) you need a DL "
            f"train-24_25/infer-25_26 driver built on dl_core.py."
        )

    out_dir = args.output_dir or os.path.join(
        here, "..", "outputs_25_26", f"external_validation_{cfg_name}")
    os.makedirs(out_dir, exist_ok=True)

    for label, path in [("train (24_25)", args.train_excel),
                        ("test  (25_26)", args.test_excel)]:
        if not os.path.exists(path):
            raise SystemExit(f"[FATAL] {label} table not found:\n    {path}\n"
                             f"Build it first (see README_25_26_validation.md).")

    # ── load samples with the SAME construction rules as the in-season runs
    #    (target column pre-cleaned to numeric so a stray '.' cell can't crash
    #    the shared loader; originals are never modified).
    print(f"[load] train: {args.train_excel}")
    train_clean = _clean_target_table(dp, args.train_excel, out_dir, "train")
    train_df = dp.load_samples(train_clean)
    print(f"[load] test : {args.test_excel}")
    test_clean = _clean_target_table(dp, args.test_excel, out_dir, "test")
    test_df = dp.load_samples(test_clean)

    # feature columns are fixed by the TRAINING season
    feat_cols = dp.get_scalar_feature_columns(train_df, cfg.scalar_groups)
    if not feat_cols:
        raise SystemExit(f"No feature columns resolved for {cfg_name} groups "
                         f"{cfg.scalar_groups} in the training table.")

    X_train, X_test, colreport = build_aligned_matrices(dp, train_df, test_df, feat_cols)
    y_train = train_df[dp.TARGET_COL].astype(np.float64).values
    y_test = test_df[dp.TARGET_COL].astype(np.float64).values
    groups_train = train_df[dp.GROUP_COL].astype(str).values
    test_stages = test_df["sampling_stage"].tolist()

    print(f"\n[{cfg_name}] config = {'+'.join(cfg.scalar_groups)}  ({cfg.model})")
    print(f"  train: {X_train.shape[0]} samples, "
          f"{pd.Series(groups_train).nunique()} plots, {X_train.shape[1]} features")
    print(f"  test : {X_test.shape[0]} samples, "
          f"{test_df[dp.GROUP_COL].nunique()} plots")
    if colreport["missing_in_test"]:
        print(f"  [warn] {len(colreport['missing_in_test'])} feature column(s) "
              f"absent from 25_26 table: {colreport['missing_in_test'][:8]}"
              f"{' ...' if len(colreport['missing_in_test']) > 8 else ''}")
    if colreport["mean_imputed"]:
        print(f"  [warn] {len(colreport['mean_imputed'])} feature column(s) "
              f"had NaNs in 25_26 -> train-mean imputed: "
              f"{colreport['mean_imputed'][:8]}"
              f"{' ...' if len(colreport['mean_imputed']) > 8 else ''}")

    # ── estimator: reuse the EXACT factories the in-season models use ──
    if cfg.model == "PLSR":
        factory = mlr.make_plsr_factory()
        use_scaler = True                 # manuscript: PLSR uses standardized X
    elif cfg.model == "RF":
        factory = mlr.make_rf_factory()
        use_scaler = False                # RF is scale-invariant
    else:
        raise SystemExit(f"Unhandled estimator '{cfg.model}' for {cfg_name}.")

    # standardize on TRAIN only (parity with in-season protocol)
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        X_train_f = scaler.transform(X_train)
        X_test_f = scaler.transform(X_test)
    else:
        X_train_f, X_test_f = X_train, X_test

    # inner-fold hyper-parameter search + fit on the FULL training season
    print("  fitting on all of 24_25 (inner GroupKFold hyper-param search)...")
    model, params = factory(X_train_f, y_train, groups_train)
    print(f"  chosen params: {params}")

    # ── predict the held-out season ──
    y_pred = np.asarray(model.predict(X_test_f)).ravel()

    # ── metrics: overall, per stage, per experiment (SB2/SB3/SB4) ──
    overall = dict(dp.regression_metrics(y_test, y_pred))
    true_cls, pred_cls, overall_cls = dp.stage_aware_classification(
        y_test, y_pred, test_stages)

    # per-stage
    per_stage = {}
    for stg in sorted({s for s in test_stages if s is not None}):
        mask = [s == stg for s in test_stages]
        per_stage[stg] = _subset_metrics(dp, y_test, y_pred, test_stages, mask)

    # per-experiment (prefix of Image_ID, e.g. SB2/SB3/SB4)
    exp = test_df[dp.GROUP_COL].astype(str).str.split("_").str[0].values
    per_exp = {}
    for e in sorted(set(exp)):
        mask = list(exp == e)
        per_exp[e] = _subset_metrics(dp, y_test, y_pred, test_stages, mask)

    # ── write per-sample predictions ──
    keep = [c for c in [dp.GROUP_COL, dp.DATE_COL, "day", "sampling_stage",
                        dp.TARGET_COL] if c in test_df.columns]
    out_pred = test_df[keep].copy()
    out_pred["y_true_reg"] = y_test
    out_pred["y_pred_reg"] = y_pred
    out_pred["abs_error"] = np.abs(y_test - y_pred)
    out_pred["y_true_cls"] = true_cls
    out_pred["y_pred_cls"] = pred_cls
    out_pred["y_true_cls_name"] = [dp.class_idx_to_name(v) for v in true_cls]
    out_pred["y_pred_cls_name"] = [dp.class_idx_to_name(v) for v in pred_cls]
    pred_path = os.path.join(out_dir, f"external_{cfg_name}_predictions.csv")
    out_pred.to_csv(pred_path, index=False)

    # ── write metrics JSON ──
    results = {
        "kind": "external_season_validation",
        "train_season": "2024_2025",
        "test_season": "2025_2026",
        "config_name": cfg_name,
        "config": "+".join(cfg.scalar_groups),
        "estimator": cfg.model,
        "chosen_params": params,
        "n_train": int(X_train.shape[0]),
        "n_train_plots": int(pd.Series(groups_train).nunique()),
        "n_test": int(X_test.shape[0]),
        "n_test_plots": int(test_df[dp.GROUP_COL].nunique()),
        "n_features": int(X_train.shape[1]),
        "used_scaler": bool(use_scaler),
        "overall_regression": overall,
        "overall_classification": overall_cls,
        "per_stage": per_stage,
        "per_experiment": per_exp,
        "column_parity_report": colreport,
        "feature_columns": feat_cols,
    }
    metrics_path = os.path.join(out_dir, f"external_{cfg_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── console summary ──
    print("\n" + "=" * 64)
    print(f"EXTERNAL VALIDATION  |  train 24_25 -> test 25_26  |  {cfg_name} "
          f"({cfg.model})")
    print("=" * 64)
    print("Overall regression on held-out season:")
    print(f"  R2={overall['R2']:.3f}  RMSE={overall['RMSE']:.3f}  "
          f"MAE={overall['MAE']:.3f}  sMAPE={overall['sMAPE']:.2f}  "
          f"(n={len(y_test)})")
    if overall_cls:
        print("Overall stage-aware classification:")
        print(f"  Acc={overall_cls.get('Acc', float('nan')):.3f}  "
              f"BalAcc={overall_cls.get('BalAcc', float('nan')):.3f}  "
              f"F1m={overall_cls.get('F1_macro', float('nan')):.3f}")
    print("\nPer phenological stage:")
    for stg, m in per_stage.items():
        if m.get("n", 0):
            print(f"  {stg:16s} n={m['n']:3d}  R2={m['R2']:.3f}  "
                  f"RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    print("\nPer experiment:")
    for e, m in per_exp.items():
        if m.get("n", 0):
            print(f"  {e:6s} n={m['n']:3d}  R2={m['R2']:.3f}  "
                  f"RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    print(f"\nPredictions -> {pred_path}")
    print(f"Metrics     -> {metrics_path}")


if __name__ == "__main__":
    main()
