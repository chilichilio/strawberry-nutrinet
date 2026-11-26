#!/usr/bin/env python3
"""
run_ml_v4.py: classical-model runs (M1 / M2 / M3 / M4 / M6), same
samples and same outer GroupKFold folds as base pipeline.

  M1, M3 -> PLSR through base pipeline ml_runner.run_ml_cv (standardized X, n_components
            by inner GroupKFold, max 30), as base pipeline M1_PLSR.py / M3_PLSR.py
  M2, M4 -> RF through base pipeline ml_runner.run_ml_cv; only the input groups differ
            (--noXY drops plot centroids, --noUAV drops V+S)
  M6     -> its own fold loop so the TCN sequence encoder can be:
              default : fit once on the outer-training partition (= base pipeline M6)
              --nested: ALSO refit inside every inner RF-tuning split
            --seqfix  : masked / left-padded / 253-day TCN (see v4_common)
            --noW     : weather sequence removed
            --shap    : TreeSHAP per held-out fold, interventional, with the
                        'typical plot' background
                        (100 training rows closest to the training median)
  --sampEnd : every non-image input ends at the leaf-sampling date (all models)
  --external: K only. Train on ALL 2024-25, predict 2025-26 (zero-label transfer).
              Writes outputs/K_ext2526/<model>__<tag>/external_<..>_predictions.csv
              in the format the base pipeline recalibration scripts read.

Output (CV): outputs/<N|K>/<model>__<tag>/ <model_tag>_fold_metrics.csv
(same columns as base pipeline), OOF predictions and a run-config JSON.
"""
import argparse
import dataclasses
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import v4_common as v4

RF_PARAM_GRID = {                       # identical to base pipeline M2/M4/M6
    "n_estimators":     [400],
    "max_depth":        [None, 20],
    "max_features":     ["sqrt", 0.33],
    "min_samples_leaf": [1, 2],
}
PLS_MAX_COMPONENTS = 30                 # identical to base pipeline M1/M3
ESEQ_EPOCHS, ESEQ_LR, ESEQ_BATCH = 40, 1e-3, 64     # identical to base pipeline M6
PLSR_MODELS, RF_MODELS = ("M1", "M3"), ("M2", "M4")


def _grid():
    import itertools
    keys = list(RF_PARAM_GRID)
    for vals in itertools.product(*[RF_PARAM_GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


# ─────────────────────────────────────────────────────────────
# TCN sequence encoder for M6 (same architecture/objective as base pipeline M6)
# ─────────────────────────────────────────────────────────────
class SeqEncoderTrainer:
    def __init__(self, dc, df, y, fert_lookup, weather_raw, seqfix, use_weather, seed):
        self.dc, self.df, self.y = dc, df, y.astype(np.float32)
        self.fert_lookup, self.weather_raw = fert_lookup, (weather_raw if use_weather else None)
        self.use_weather = use_weather
        self.seqfix, self.seed = seqfix, seed
        self.emb_dim = dc.TCN_FERT_OUT + dc.TCN_WEATHER_OUT

    def _build(self, df, fert_lookup, fert_mean, fert_std, wnorm):
        dc = self.dc
        F, W = [], []
        for i in range(len(df)):
            row = df.iloc[i]
            dap = dc._dap_of_row(row)                  # 'DAP' (v4 column) else 'day'
            iid = dc.normalize_image_id(row.get("Image_ID", ""))
            if fert_lookup is not None and iid in fert_lookup and dap > 0:
                a = (fert_lookup[iid][:dap].astype(np.float32) - fert_mean) / fert_std
            else:
                a = np.zeros(max(dap, 1), np.float32)
            if wnorm is not None and dap > 0:
                w = wnorm[:min(dap + 1, wnorm.shape[0])].T.astype(np.float32)
            else:
                w = np.zeros((dc.N_WEATHER_VARS, 1), np.float32)
            F.append(a); W.append(w)
        return F, W

    # sequences for every row of self.df, normalized with stats from fit rows only
    def _sequences(self, fit_idx):
        fert_mean, fert_std, wnorm = self.dc._fold_seq_norm(self.df, fit_idx, self.fert_lookup,
                                                            self.weather_raw)
        return self._build(self.df, self.fert_lookup, fert_mean, fert_std, wnorm)

    def other_season_sequences(self, fit_idx, df_other, fert_lookup_other, weather_raw_other):
        """Sequences for another season's rows, normalized with THIS (training) season's
        fit-row statistics, as the base pipeline external harness does (train-season scaler)."""
        dc = self.dc
        fert_mean, fert_std, _ = dc._fold_seq_norm(self.df, fit_idx, self.fert_lookup, None)
        wnorm = None
        if self.use_weather and self.weather_raw is not None and weather_raw_other is not None:
            w_mu, w_sd = dc.weather_stats_from_rows(self.df, fit_idx, self.weather_raw)
            wnorm = dc.apply_weather_norm(weather_raw_other, w_mu, w_sd)
        return self._build(df_other, fert_lookup_other, fert_mean, fert_std, wnorm)

    def _batch(self, F, W, idxs):
        import torch
        import torch.nn.functional as Fn
        if self.seqfix:
            return (v4.left_pad_nan([F[i] for i in idxs]).unsqueeze(1),
                    v4.left_pad_nan([W[i] for i in idxs]))
        mf = max(F[i].shape[-1] for i in idxs); mw = max(W[i].shape[-1] for i in idxs)   # base pipeline
        f = torch.stack([Fn.pad(torch.from_numpy(F[i]), (0, mf - F[i].shape[-1])) for i in idxs]).unsqueeze(1)
        w = torch.stack([Fn.pad(torch.from_numpy(W[i]), (0, mw - W[i].shape[-1])) for i in idxs])
        return f, w

    def _model(self):
        import torch.nn as nn
        dc = self.dc

        class SeqRegressor(nn.Module):
            def __init__(s):
                super().__init__()
                s.fert_enc = dc.TCNEncoder(1, dc.TCN_HIDDEN, dc.TCN_LEVELS, dc.TCN_KSIZE, dc.TCN_FERT_OUT)
                s.weath_enc = dc.TCNEncoder(dc.N_WEATHER_VARS, dc.TCN_HIDDEN, dc.TCN_LEVELS,
                                            dc.TCN_KSIZE, dc.TCN_WEATHER_OUT)
                s.head = nn.Sequential(nn.Linear(self.emb_dim, 32), nn.ReLU(inplace=True),
                                       nn.Linear(32, 1))   # = base pipeline _SeqRegressor

            def embed(s, f, w):
                import torch
                return torch.cat([s.fert_enc(f), s.weath_enc(w)], 1)

            def forward(s, f, w):
                e = s.embed(f, w)
                return s.head(e).squeeze(-1), e

        return SeqRegressor()

    def _embed(self, model, F, W, idx):
        import torch
        idx = list(idx); e = []
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                f, w = self._batch(F, W, idx[s:s + 256])
                e.append(model.embed(f, w).numpy())
        return np.vstack(e) if e else np.zeros((0, self.emb_dim), np.float32)

    def fit_embed(self, fit_idx, embed_sets, other=None):
        """Train on fit_idx (labels standardized on fit rows), return embeddings for each
        index array in embed_sets; with other=(F, W) also the embeddings of those
        (another season's) sequences, appended last."""
        import torch
        torch.manual_seed(self.seed); np.random.seed(self.seed)   # = base pipeline M6 seeding
        F, W = self._sequences(fit_idx)
        y = self.y; ym = float(y[fit_idx].mean()); ys = max(float(y[fit_idx].std()), 1e-6)
        model = self._model()
        opt = torch.optim.Adam(model.parameters(), lr=ESEQ_LR, weight_decay=1e-4)
        loss_fn = torch.nn.SmoothL1Loss()
        order = list(fit_idx)
        model.train()
        for _ in range(ESEQ_EPOCHS):
            np.random.shuffle(order)
            for s in range(0, len(order), ESEQ_BATCH):
                idxs = order[s:s + ESEQ_BATCH]
                f, w = self._batch(F, W, idxs)
                pred, _ = model(f, w)
                loss = loss_fn(pred, torch.from_numpy((y[idxs] - ym) / ys))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        outs = [self._embed(model, F, W, idx) for idx in embed_sets]
        if other is not None:
            F2, W2 = other
            outs.append(self._embed(model, F2, W2, range(len(F2))))
        return outs


# ─────────────────────────────────────────────────────────────
def _rf(params, seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)


def _tune_rf(X, y, inner_sets, seed):
    """base pipeline selection rule: lowest mean inner-fold RMSE. inner_sets entries are
    (E_fit, E_val, itr, iva) with the embeddings to append (or None)."""
    best, best_score = None, np.inf
    for params in _grid():
        rmse = []
        for e_fit, e_val, itr, iva in inner_sets:
            Xa = X[itr] if e_fit is None else np.hstack([X[itr], e_fit])
            Xb = X[iva] if e_val is None else np.hstack([X[iva], e_val])
            m = _rf(params, seed).fit(Xa, y[itr])
            rmse.append(np.sqrt(np.mean((m.predict(Xb) - y[iva]) ** 2)))
        if np.mean(rmse) < best_score:
            best, best_score = params, float(np.mean(rmse))
    return best


def _inner_splits(groups):
    import sklearn.model_selection as skms          # treatment splitter when --treat (patched)
    return list(skms.GroupKFold(n_splits=min(5, len(np.unique(groups))))
                .split(np.arange(len(groups)), groups=groups))


def _m6_fit_predict(enc, X, y, groups, tr, te_sets, nested, seed, other=None):
    """Fit the M6 pipeline on rows tr of (X, y) and return (rf, E_tr, [E for te_sets], E_other)."""
    inner = _inner_splits(groups[tr])
    inner_sets = []
    if nested:                                       # encoder refit on each inner-fit subset
        for itr, iva in inner:
            e_fit, e_val = enc.fit_embed(tr[itr], [tr[itr], tr[iva]])
            inner_sets.append((e_fit, e_val, itr, iva))
    outs = enc.fit_embed(tr, [tr] + list(te_sets), other=other)   # outer encoder (base pipeline)
    E_tr, E_te = outs[0], outs[1:1 + len(te_sets)]
    E_other = outs[-1] if other is not None else None
    if not nested:                                   # base pipeline: inner search reuses outer embeddings
        inner_sets = [(E_tr[itr], E_tr[iva], itr, iva) for itr, iva in inner]
    best = _tune_rf(X[tr], y[tr], inner_sets, seed)
    rf = _rf(best, seed).fit(np.hstack([X[tr], E_tr]), y[tr])
    return rf, best, E_tr, E_te, E_other


def _shap_fold(rf, X_tr_full, X_va_full):
    import shap
    bi = v4.central_background(X_tr_full)
    ex = shap.TreeExplainer(rf, data=X_tr_full[bi], feature_perturbation="interventional")
    sv = np.asarray(ex.shap_values(X_va_full, check_additivity=False))
    if sv.ndim == 3:
        sv = sv[..., 0]
    base = float(np.ravel(ex.expected_value)[0])
    err = np.abs(sv.sum(1) + base - rf.predict(X_va_full))
    # interventional TreeSHAP is exact for most rows; with current shap/sklearn a few %
    # of rows are off by up to ~0.03 (target units) -> reported, not hidden
    return sv, base, {"max": float(err.max()), "median": float(np.median(err)),
                      "p95": float(np.percentile(err, 95)), "frac_gt_0.01": float((err > 0.01).mean())}, bi


def _write_shap(out, ds, dp, names, shap_parts):
    """Long table + per-feature and per-group summaries (sum and per-feature mean |SHAP|)."""
    gdef = dp.build_feature_groups(ds.df)
    group_of = {c: g for g, cols in gdef.items() for c in cols}
    for n in names:
        if n.startswith("eseq_fert"):
            group_of[n] = "Eseq_fert"
        elif n.startswith("eseq_weather"):
            group_of[n] = "Eseq_weather"
    rows = []
    for fold_id, va, sv, base in shap_parts:
        for j, i in enumerate(va):
            for k, n in enumerate(names):
                rows.append((fold_id, ds.df["Image_ID"].iloc[i], ds.df["conc_date"].iloc[i],
                             ds.stages[i], n, group_of.get(n, "?"), float(sv[j, k]), base))
    L = pd.DataFrame(rows, columns=["fold", "Image_ID", "conc_date", "sampling_stage", "feature",
                                    "group", "shap", "base_value"])
    L.to_csv(os.path.join(out, "M6_shap_values_long.csv"), index=False)
    L["abs"] = L["shap"].abs()
    feat = (L.groupby(["feature", "group"])
              .agg(mean_abs_shap=("abs", "mean"), mean_shap=("shap", "mean")).reset_index()
              .sort_values("mean_abs_shap", ascending=False))
    feat["rank"] = np.arange(1, len(feat) + 1)
    feat.to_csv(os.path.join(out, "M6_shap_feature_summary.csv"), index=False)
    grp = feat.groupby("group").agg(sum_mean_abs=("mean_abs_shap", "sum"),
                                    n_features=("feature", "size")).reset_index()
    grp["share_pct_sum"] = 100 * grp.sum_mean_abs / grp.sum_mean_abs.sum()
    grp["per_feature_mean_abs"] = grp.sum_mean_abs / grp.n_features
    grp["share_pct_per_feature"] = 100 * grp.per_feature_mean_abs / grp.per_feature_mean_abs.sum()
    grp.sort_values("sum_mean_abs", ascending=False) \
       .to_csv(os.path.join(out, "M6_shap_group_summary.csv"), index=False)
    print("[M6 v4] SHAP written: M6_shap_values_long / _feature_summary / _group_summary", flush=True)


def run_m6(dp, dc, nut, v, out, seed, do_shap):
    cfg_name = v4.register_ml_config(dp, "M6", v)
    df = dp.load_samples(dp_excel(dp))
    ds = dp.build_ml_dataset(df, dp.MODEL_CONFIGS[cfg_name])
    dp.assert_no_group_leakage(ds.df, ds.splits)
    fert_lookup, _ = dc.load_fert_lookup(v4.fert_path(nut))
    weather_raw, _ = dc.load_weather_array(v4.weather_path(nut), normalize=False)
    enc = SeqEncoderTrainer(dc, ds.df, ds.y, fert_lookup, weather_raw, v.seqfix, not v.noW, seed)
    names = list(ds.feature_cols) + [f"eseq_fert_{i}" for i in range(dc.TCN_FERT_OUT)] + \
        [f"eseq_weather_{i}" for i in range(dc.TCN_WEATHER_OUT)]
    print(f"[M6 v4] {ds.summary()}  nested={v.nested} seqfix={v.seqfix} noW={v.noW} shap={do_shap}",
          flush=True)

    fold_rows, oof, shap_parts, bg_rows, add_rows = [], np.full(len(ds.y), np.nan), [], [], []
    for fold_id, (tr, va) in enumerate(ds.splits, start=1):
        t0 = time.time()
        tr, va = np.asarray(tr), np.asarray(va)
        rf, best, E_tr, (E_va,), _ = _m6_fit_predict(enc, ds.X, ds.y, ds.groups, tr, [va],
                                                     v.nested, seed)
        X_va_full = np.hstack([ds.X[va], E_va])
        pred = rf.predict(X_va_full)
        oof[va] = pred
        r = dp.regression_metrics(ds.y[va], pred)
        r.update(fold=fold_id, n_train=len(tr), n_val=len(va), params=json.dumps(best))
        fold_rows.append(r)
        print(f"  fold {fold_id}: R2={r.get('R2', float('nan')):.4f} params={best} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if do_shap:
            t1 = time.time()
            sv, base, err, bi = _shap_fold(rf, np.hstack([ds.X[tr], E_tr]), X_va_full)
            shap_parts.append((fold_id, va, sv, base))
            add_rows.append({"fold": fold_id, "base_value": base, **{f"additivity_err_{k}": x
                                                                     for k, x in err.items()}})
            b = ds.df.iloc[tr[bi]][["Image_ID", "conc_date"]].copy()
            b["fold"], b["y"] = fold_id, ds.y[tr[bi]]
            bg_rows.append(b)
            print(f"     SHAP: background={len(bi)} typical training plots, base={base:.3f}, "
                  f"additivity |err| median={err['median']:.1e} max={err['max']:.1e} "
                  f"(>0.01 in {100 * err['frac_gt_0.01']:.1f}% of rows) ({time.time() - t1:.0f}s)", flush=True)

    tag = "M6_RF"
    pd.DataFrame(fold_rows).to_csv(os.path.join(out, f"{tag}_fold_metrics.csv"), index=False)
    pd.DataFrame({"Image_ID": ds.df["Image_ID"].values, "conc_date": ds.df["conc_date"].values,
                  "sampling_stage": ds.stages, "y_true": ds.y, "y_pred": oof}) \
        .to_csv(os.path.join(out, f"{tag}_oof_predictions.csv"), index=False)
    if do_shap:
        pd.concat(bg_rows).to_csv(os.path.join(out, "M6_shap_background_samples.csv"), index=False)
        pd.DataFrame(add_rows).to_csv(os.path.join(out, "M6_shap_additivity_check.csv"), index=False)
        _write_shap(out, ds, dp, names, shap_parts)
    pooled = dp.regression_metrics(ds.y, oof)
    print(f"[M6 v4] pooled OOF: {pooled}", flush=True)
    print(f"[M6 v4] fold mean R2 = {np.mean([r['R2'] for r in fold_rows]):.4f}", flush=True)


# ─────────────────────────────────────────────────────────────
# external season (K): train all 2024-25 -> predict 2025-26
# ─────────────────────────────────────────────────────────────
def run_external(dp, mlr, dc, model, v, out, seed):
    sys.path.insert(0, v4.EXT_CODE_DIR)
    ext = importlib.import_module("validate_external_25_26")     # helpers only (unchanged)
    tr_x = v4.table_path("24_25")
    te_x = v4.table_path("25_26")
    train_df = dp.load_samples(ext._clean_target_table(dp, tr_x, out, "train"))
    test_df = dp.load_samples(ext._clean_target_table(dp, te_x, out, "test"))
    cfg_name = v4.register_ml_config(dp, model, v)
    cfg = dp.MODEL_CONFIGS[cfg_name]
    feat_cols = dp.get_scalar_feature_columns(train_df, cfg.scalar_groups)
    X_tr, X_te, colrep = ext.build_aligned_matrices(dp, train_df, test_df, feat_cols)
    y_tr = train_df[dp.TARGET_COL].astype(float).values
    y_te = test_df[dp.TARGET_COL].astype(float).values
    g_tr = train_df[dp.GROUP_COL].astype(str).values
    print(f"[ext v4] {cfg_name}: train {X_tr.shape}, test {X_te.shape}; "
          f"imputed in test: {colrep.get('mean_imputed', [])[:6]}", flush=True)

    if model in PLSR_MODELS:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_tr)
        est, params = mlr.make_plsr_factory(max_components=PLS_MAX_COMPONENTS)(sc.transform(X_tr), y_tr, g_tr)
        pred = np.asarray(est.predict(sc.transform(X_te))).ravel()
    elif model in RF_MODELS:
        est, params = mlr.make_rf_factory(param_grid=RF_PARAM_GRID)(X_tr, y_tr, g_tr)
        pred = est.predict(X_te)
    else:                                                           # M6
        fl_tr, _ = dc.load_fert_lookup(v4.fert_path("24_25"))
        w_tr, _ = dc.load_weather_array(v4.weather_path("24_25"), normalize=False)
        fl_te, _ = dc.load_fert_lookup(v4.fert_path("25_26"))
        w_te, _ = dc.load_weather_array(v4.weather_path("25_26"), normalize=False)
        enc = SeqEncoderTrainer(dc, train_df.reset_index(drop=True), y_tr, fl_tr, w_tr,
                                v.seqfix, not v.noW, seed)
        all_tr = np.arange(len(y_tr))
        other = enc.other_season_sequences(all_tr, test_df.reset_index(drop=True), fl_te, w_te)
        rf, params, _, _, E_te = _m6_fit_predict(enc, X_tr, y_tr, g_tr, all_tr, [], v.nested,
                                                 seed, other=other)
        pred = rf.predict(np.hstack([X_te, E_te]))

    run_name = f"{model}__{v.tag()}"
    P = test_df[[c for c in ["Image_ID", "conc_date", "day", "sampling_stage"] if c in test_df]].copy()
    P["y_true_reg"], P["y_pred_reg"] = y_te, pred
    P["abs_error"] = np.abs(y_te - pred)
    P.to_csv(os.path.join(out, f"external_{run_name}_predictions.csv"), index=False)
    m = v4.transfer_metrics(y_te, pred)
    m.update(params=params if isinstance(params, dict) else str(params), n_train=int(len(y_tr)),
             feature_columns=feat_cols, column_parity_report=colrep)
    with open(os.path.join(out, f"external_{run_name}_transfer.json"), "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(f"[ext v4] {run_name}: R2={m['R2']:.3f} r={m['pearson_r']:.3f} bias={m['bias']:.3f} "
          f"slope={m['slope_measured_on_pred']:.2f} RMSE={m['RMSE']:.3f}", flush=True)


def dp_excel(dp):
    import ml_runner as mlr
    return mlr.DEFAULT_EXCEL


def main():
    ap = argparse.ArgumentParser()
    v4.add_variant_args(ap)
    ap.add_argument("--model", required=True, choices=["M1", "M2", "M3", "M4", "M6"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shap", action="store_true", help="M6: TreeSHAP, typical-plot background")
    ap.add_argument("--external", action="store_true", help="K only: train 2024-25, predict 2025-26")
    args = ap.parse_args()
    v = v4.Variant.from_args(args)
    if args.model != "M6" and (v.seqfix or v.nested or v.noW or args.shap):
        raise SystemExit("--seqfix/--nested/--noW/--shap apply to M6 only")
    if v.single or v.noF or v.lstm or v.noTemp:
        raise SystemExit("--single/--noF/--lstm/--noTemp apply to M7 only")
    if args.model in PLSR_MODELS and v.noUAV and args.model == "M1":
        raise SystemExit("M1 without UAV inputs has only C+P+T; use M2 --noUAV instead")
    if args.external and (args.nutrient != "K" or v.treat or args.shap):
        raise SystemExit("--external is for K, without --treat/--shap")

    dp, mlr, dc = v4.setup(args.nutrient, need_dl=(args.model == "M6"))
    v4.patch_load_samples(dp, args.nutrient, v)
    if v.seqfix:
        v4.patch_seqfix(dc)
    if v.treat:
        v4.patch_treatfold(dp, dc, args.nutrient)
    out = v4.run_dir(args.nutrient, args.model, v, external=args.external)
    if os.environ.get("V4_FOLD_MANIFEST"):
        import v4_dlfolds
        v4_dlfolds.patch(dp, args.nutrient, os.environ["V4_FOLD_MANIFEST"])
        out = out.replace(v4.OUT_ROOT, v4.OUT_ROOT + "_dlfold", 1)
        os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "v4_run_config.json"), "w") as f:
        json.dump({"variant": dataclasses.asdict(v), "nutrient": args.nutrient, "model": args.model,
                   "seed": args.seed, "shap": args.shap, "external": args.external}, f, indent=2)

    if args.dry_run:                      # load the REAL data with this variant and stop
        name = v4.register_ml_config(dp, args.model, v)
        df = dp.load_samples(mlr.DEFAULT_EXCEL)
        ds = dp.build_ml_dataset(df, dp.MODEL_CONFIGS[name])
        g = dp.build_feature_groups(ds.df)
        print(f"[dry-run] {name}: {ds.summary()}", flush=True)
        for k in dp.MODEL_CONFIGS[name].scalar_groups:
            print(f"    {k:5s} {len(g.get(k, []))} cols  {g.get(k, [])[:4]}", flush=True)
        print(f"    DAP range {df['DAP'].min()}-{df['DAP'].max()}, NaN in X: {int(np.isnan(ds.X).sum())}",
              flush=True)
        if args.external:                 # also the 2025-26 test table, through the same patches
            sys.path.insert(0, v4.EXT_CODE_DIR)
            ext = importlib.import_module("validate_external_25_26")
            te = dp.load_samples(ext._clean_target_table(
                dp, v4.table_path("25_26"),
                out, "test"))
            print(f"    external test: {len(te)} samples, season {v4.season_of_df(te, dp)}, "
                  f"DAP {te['DAP'].min()}-{te['DAP'].max()}", flush=True)
            if args.model == "M6":
                dc.load_fert_lookup(v4.fert_path("25_26"))
                w, _ = dc.load_weather_array(v4.weather_path("25_26"), normalize=False)
                print(f"    25_26 daily weather {w.shape}", flush=True)
        if args.model == "M6":
            dc.load_fert_lookup(v4.fert_path(args.nutrient))
            w, _ = dc.load_weather_array(v4.weather_path(args.nutrient), normalize=False)
            print(f"    daily weather {w.shape}, all-zero rows: {int((w == 0).all(1).sum())}", flush=True)
        print("[dry-run] OK", flush=True)
        return

    if args.external:
        run_external(dp, mlr, dc, args.model, v, out, args.seed)
        return
    if args.model == "M6":
        run_m6(dp, dc, args.nutrient, v, out, args.seed, args.shap)
        return

    name = v4.register_ml_config(dp, args.model, v)
    print(f"[v4] {name}: groups {dp.MODEL_CONFIGS[name].scalar_groups}", flush=True)
    if args.model in PLSR_MODELS:
        mlr.run_ml_cv(config_name=name, model_tag=f"{args.model}_PLSR",
                      estimator_factory=mlr.make_plsr_factory(max_components=PLS_MAX_COMPONENTS),
                      excel_path=mlr.DEFAULT_EXCEL, output_dir=out,
                      use_scaler=True, importance=False)
    else:
        mlr.run_ml_cv(config_name=name, model_tag=f"{args.model}_RF",
                      estimator_factory=mlr.make_rf_factory(param_grid=RF_PARAM_GRID),
                      excel_path=mlr.DEFAULT_EXCEL, output_dir=out,
                      use_scaler=False, importance=True)


if __name__ == "__main__":
    main()
