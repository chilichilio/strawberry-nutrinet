# ============================================================
# ml_runner.py — shared leakage-safe CV engine for the classical-ML
#                configurations (M1 PLSR, M2 RF, M3 PLSR+agg,
#                M4 RF+agg, M6 RF+Eseq).
# ------------------------------------------------------------
# WHY THIS MODULE
#   The manuscript (Table 2.7) compares seven model configurations on
#   ONE canonical sample representation and ONE leakage-safe evaluation
#   protocol.  To make the classical-ML baselines head-to-head comparable
#   with the deep-learning models (M5, M7), every ML script funnels through
#   this single engine, which in turn consumes data_prep.py — the same
#   preprocessing layer the DL path uses.  That guarantees:
#
#     * identical samples / labels        (data_prep.load_samples)
#     * identical feature grouping        (data_prep.build_feature_groups)
#     * identical 5-fold GroupKFold on Image_ID (plot never split)
#     * identical regression metrics      (MAE, RMSE, R2, sMAPE)
#     * identical stage-aware 3-class scoring (deficient/adequate/high)
#     * all scaling / hyper-parameter selection fit on the TRAIN fold ONLY
#
#   Each model script only supplies:
#       - the config name (M1..M6) -> selects the feature groups
#       - an estimator factory (how to select hyper-parameters + fit)
#       - whether to standardize features (PLSR yes; RF scale-invariant)
#       - optional per-fold feature augmentation (M6 appends Eseq)
#
# Depends only on numpy / pandas / scikit-learn (+ whatever the estimator
# factory needs).  data_prep.py must be importable (same folder).
# ============================================================

from __future__ import annotations

import os
import json
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data_prep as dp


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASEFOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The combined tabular table lives in the 24_25 data folder.
DEFAULT_EXCEL = os.path.join(
    BASEFOLDER, "data", "24_25_K", "trait_K_weather_yield_combined_conc.xlsx"
)


# Type aliases for clarity
EstimatorFactory = Callable[
    [np.ndarray, np.ndarray, np.ndarray], Tuple[object, Dict]
]
# augment(ds, train_idx, val_idx) -> (X_train_extra, X_val_extra, extra_names)
AugmentFn = Callable[
    [dp.MLDataset, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, List[str]],
]


# ─────────────────────────────────────────────────────────────
# MAIN CV DRIVER
# ─────────────────────────────────────────────────────────────
def run_ml_cv(
    *,
    config_name: str,
    model_tag: str,
    estimator_factory: EstimatorFactory,
    excel_path: str = DEFAULT_EXCEL,
    output_dir: Optional[str] = None,
    use_scaler: bool = True,
    importance: bool = False,
    augment_fn: Optional[AugmentFn] = None,
    extra_config_note: str = "",
) -> Dict:
    """Run the shared leakage-safe 5-fold CV for one ML configuration.

    Parameters
    ----------
    config_name : 'M1'..'M6'  -> selects feature groups via data_prep.MODEL_CONFIGS
    model_tag   : short label used for output file names (e.g. 'M3_PLSR')
    estimator_factory : fn(X_tr, y_tr, groups_tr) -> (fitted_model, chosen_params)
                        Performs any inner-fold hyper-parameter search and returns
                        a model already fitted on the full training fold.
    use_scaler  : standardize features (fit on TRAIN fold only). PLSR=True.
                  RF is scale-invariant; pass False to skip (kept for parity).
    importance  : if True, average estimator.feature_importances_ across folds.
    augment_fn  : optional per-fold feature builder (M6 Eseq). Fit on TRAIN fold.

    Returns the results dict (also written to <output_dir>/<model_tag>_results.json).
    """
    cfg = dp.MODEL_CONFIGS[config_name]
    if output_dir is None:
        output_dir = os.path.join(BASEFOLDER, f"outputs_{model_tag}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{model_tag}] Loading {excel_path}")
    df = dp.load_samples(excel_path)
    ds = dp.build_ml_dataset(df, cfg)             # X / y / groups / folds (GroupKFold)
    dp.assert_no_group_leakage(ds.df, ds.splits)  # safety: no plot across folds

    base_feature_cols = list(ds.feature_cols)
    print(f"[{model_tag}] {ds.summary()}")
    for g in cfg.scalar_groups:
        gdef = dp.build_feature_groups(ds.df)
        print(f"    group {g:5s}: {len(gdef.get(g, []))} cols")

    y = ds.y
    groups = ds.groups
    stages = ds.stages

    fold_rows: List[Dict] = []
    oof_pred = np.full(len(ds.df), np.nan)
    chosen_params: List[Dict] = []
    importances = None
    extra_names_final: List[str] = []

    for fold_id, (tr, va) in enumerate(ds.splits, start=1):
        X_tr = ds.X[tr].copy()
        X_va = ds.X[va].copy()
        fold_feature_cols = list(base_feature_cols)

        # ── optional per-fold feature augmentation (e.g. M6 Eseq) ──
        if augment_fn is not None:
            X_tr_e, X_va_e, extra_names = augment_fn(ds, tr, va)
            if X_tr_e is not None and X_tr_e.size:
                X_tr = np.hstack([X_tr, X_tr_e])
                X_va = np.hstack([X_va, X_va_e])
                fold_feature_cols = fold_feature_cols + list(extra_names)
                extra_names_final = list(extra_names)

        # ── standardize predictors (fit on TRAIN fold only) ──
        if use_scaler:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_va = scaler.transform(X_va)

        # ── select hyper-parameters + fit on the training fold only ──
        model, params = estimator_factory(X_tr, y[tr], groups[tr])
        chosen_params.append(params)

        pred_va = np.asarray(model.predict(X_va)).ravel()
        oof_pred[va] = pred_va

        if importance and hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=np.float64)
            if importances is None:
                importances = np.zeros(len(fold_feature_cols), dtype=np.float64)
            # guard against length drift between folds (augment is stable here)
            n = min(len(importances), len(imp))
            importances[:n] += imp[:n]

        m = dp.regression_metrics(y[va], pred_va)
        m.update({"fold": fold_id, "n_train": int(len(tr)), "n_val": int(len(va)),
                  "params": json.dumps(params)})
        fold_rows.append(m)
        print(f"  Fold {fold_id}: R2={m['R2']:.3f} RMSE={m['RMSE']:.3f} "
              f"MAE={m['MAE']:.3f} sMAPE={m['sMAPE']:.2f} | {params}")

    # ── per-fold mean ± std ──
    fold_df = pd.DataFrame(fold_rows)
    summary = {}
    for k in ["R2", "RMSE", "MAE", "sMAPE"]:
        summary[f"{k}_mean"] = float(fold_df[k].mean())
        summary[f"{k}_std"] = float(fold_df[k].std(ddof=0))

    # ── pooled out-of-fold metrics (manuscript Table style) ──
    pooled_reg = dp.regression_metrics(y, oof_pred)

    # ── stage-aware classification from pooled OOF predictions ──
    true_cls, pred_cls, pooled_cls = dp.stage_aware_classification(y, oof_pred, stages)

    # ── save per-sample OOF predictions ──
    keep = [c for c in [dp.GROUP_COL, dp.DATE_COL, "day", "sampling_stage", dp.TARGET_COL]
            if c in ds.df.columns]
    out_pred = ds.df[keep].copy()
    out_pred["y_true_reg"] = y
    out_pred["y_pred_reg"] = oof_pred
    out_pred["y_true_cls"] = true_cls
    out_pred["y_pred_cls"] = pred_cls
    out_pred["y_true_cls_name"] = [dp.class_idx_to_name(v) for v in true_cls]
    out_pred["y_pred_cls_name"] = [dp.class_idx_to_name(v) for v in pred_cls]
    out_pred.to_csv(os.path.join(output_dir, f"{model_tag}_oof_predictions.csv"), index=False)

    fold_df.to_csv(os.path.join(output_dir, f"{model_tag}_fold_metrics.csv"), index=False)

    # ── feature importance (RF) ──
    top_features = []
    if importance and importances is not None:
        names = base_feature_cols + extra_names_final
        names = names[:len(importances)]
        imp_df = (pd.DataFrame({"feature": names,
                                "importance": importances / len(ds.splits)})
                  .sort_values("importance", ascending=False))
        imp_df.to_csv(os.path.join(output_dir, f"{model_tag}_feature_importance.csv"),
                      index=False)
        top_features = imp_df.head(15).to_dict(orient="records")

    config_str = "+".join(cfg.scalar_groups) + (
        "+Eseq" if cfg.sequence else "")
    results = {
        "model": model_tag,
        "config_name": config_name,
        "config": config_str,
        "estimator": cfg.model,
        "note": extra_config_note or cfg.note,
        "n_samples": int(len(ds.df)),
        "n_plots": int(pd.Series(groups).nunique()),
        "n_features": int(len(base_feature_cols) + len(extra_names_final)),
        "folds": len(ds.splits),
        "chosen_params": chosen_params,
        "per_fold_mean_std": summary,
        "pooled_oof_regression": pooled_reg,
        "pooled_oof_classification": pooled_cls,
        "top15_features": top_features,
        "feature_groups": {g: dp.build_feature_groups(ds.df).get(g, [])
                           for g in cfg.scalar_groups},
        "extra_features": extra_names_final,
    }
    with open(os.path.join(output_dir, f"{model_tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(model_tag, config_str, summary, pooled_reg, pooled_cls, output_dir)
    return results


def _print_summary(model_tag, config_str, summary, pooled_reg, pooled_cls, output_dir):
    print("\n" + "=" * 62)
    print(f"{model_tag}  ({config_str})")
    print("=" * 62)
    print("Per-fold mean ± std:")
    for k in ["R2", "RMSE", "MAE", "sMAPE"]:
        print(f"  {k:6s}: {summary[f'{k}_mean']:.3f} ± {summary[f'{k}_std']:.3f}")
    print("Pooled out-of-fold regression:")
    print(f"  R2={pooled_reg['R2']:.3f} RMSE={pooled_reg['RMSE']:.3f} "
          f"MAE={pooled_reg['MAE']:.3f} sMAPE={pooled_reg['sMAPE']:.2f}")
    if pooled_cls:
        print("Pooled out-of-fold classification (stage-aware):")
        print(f"  Acc={pooled_cls.get('Acc', float('nan')):.3f} "
              f"BalAcc={pooled_cls.get('BalAcc', float('nan')):.3f} "
              f"F1w={pooled_cls.get('F1_weighted', float('nan')):.3f} "
              f"F1m={pooled_cls.get('F1_macro', float('nan')):.3f}")
    print(f"\nOutputs written to: {output_dir}")


# ─────────────────────────────────────────────────────────────
# REUSABLE ESTIMATOR FACTORIES
# ─────────────────────────────────────────────────────────────
def _iter_param_grid(grid: Dict[str, list]):
    import itertools
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def make_plsr_factory(max_components: int = 30, seed: int = dp.GLOBAL_SEED):
    """PLSR factory: selects n_components by inner GroupKFold on the train fold."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    import math

    def factory(X_tr, y_tr, groups_tr):
        n_groups = len(np.unique(groups_tr))
        inner_k = min(5, n_groups)
        max_comp = int(min(max_components, X_tr.shape[1], X_tr.shape[0] - 1))
        if inner_k < 2 or max_comp < 1:
            n_comp = max(1, min(max_comp, 2))
        else:
            gkf = GroupKFold(n_splits=inner_k)
            best_c, best_rmse = 1, np.inf
            for c in range(1, max_comp + 1):
                rmses = []
                for itr, iva in gkf.split(X_tr, groups=groups_tr):
                    pls = PLSRegression(n_components=c, scale=False)
                    pls.fit(X_tr[itr], y_tr[itr])
                    pred = pls.predict(X_tr[iva]).ravel()
                    rmses.append(math.sqrt(np.mean((y_tr[iva] - pred) ** 2)))
                mrmse = float(np.mean(rmses))
                if mrmse < best_rmse:
                    best_rmse, best_c = mrmse, c
            n_comp = best_c
        model = PLSRegression(n_components=n_comp, scale=False)
        model.fit(X_tr, y_tr)
        return model, {"n_components": int(n_comp)}

    return factory


def make_rf_factory(param_grid: Optional[Dict[str, list]] = None,
                    seed: int = dp.GLOBAL_SEED):
    """Random Forest factory: inner GroupKFold grid search on the train fold."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    import math

    if param_grid is None:
        param_grid = {
            "n_estimators":     [400],
            "max_depth":        [None, 20],
            "max_features":     ["sqrt", 0.33],
            "min_samples_leaf": [1, 2],
        }

    def factory(X_tr, y_tr, groups_tr):
        n_groups = len(np.unique(groups_tr))
        inner_k = min(5, n_groups)
        if inner_k < 2:
            best = {"n_estimators": 400, "max_depth": None,
                    "max_features": "sqrt", "min_samples_leaf": 1}
        else:
            gkf = GroupKFold(n_splits=inner_k)
            inner_splits = list(gkf.split(X_tr, groups=groups_tr))
            best, best_rmse = None, np.inf
            for params in _iter_param_grid(param_grid):
                rmses = []
                for itr, iva in inner_splits:
                    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
                    rf.fit(X_tr[itr], y_tr[itr])
                    pred = rf.predict(X_tr[iva])
                    rmses.append(math.sqrt(np.mean((y_tr[iva] - pred) ** 2)))
                mrmse = float(np.mean(rmses))
                if mrmse < best_rmse:
                    best_rmse, best = mrmse, params
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **best)
        model.fit(X_tr, y_tr)
        return model, dict(best)

    return factory
