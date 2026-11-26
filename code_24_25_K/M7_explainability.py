# ============================================================
# M7_explainability.py
# ------------------------------------------------------------
# Standalone, paper-grade explainability runner for the M7 temporal
# multimodal Transformer checkpoints saved by M7_Transformer.py (whatever
# --checkpoint_dir you point it at)
# (K% leaf-nutrient model, 24-25 season; see M7_Transformer.py + dl_core.py).
#
# Rather than re-implementing the model, dataset, and sample-building code,
# this script IMPORTS dl_core.py and data_prep.py directly -- the same modules
# M7_Transformer.py trains with -- so the rebuilt samples, GroupKFold splits,
# per-fold normalization stats, and DualTaskModel architecture are
# guaranteed identical to what produced the checkpoints. No retraining.
#
# What it produces per validation sample (held-out fold), matching + extending
# what the manuscript's "Explainability analysis" section describes:
#   * Grad-CAM on the ortho (multispectral) CNN encoder, per image date +
#     a temporal aggregate, overlaid on RGB composites
#   * Grad-CAM on the DEM CNN encoder, per image date + temporal aggregate
#   * Integrated Gradients (IG): per-band ortho saliency, DEM saliency,
#     per-date image contribution, tabular feature attribution,
#     fertilizer-TCN day-level attribution, weather-TCN variable x day
#     attribution heatmap
#   * Modality/group contribution percentages (ortho / DEM / tabular /
#     fertilizer / weather) from IG, mirroring Figure 3.8.3 in the manuscript
#   * Tabular SHAP (GradientExplainer): signed bar, beeswarm, waterfall,
#     decision plot -- images/fert/weather held fixed, mirroring Figure 3.8.4
#   * NEW -- attention-pooling weights: which image date(s) the model's
#     MaskedAttentionPooling actually weighted most heavily (a direct,
#     zero-extra-cost readout of the "temporal multimodal Transformer",
#     not present in 25c/25d)
#   * NEW -- Transformer self-attention rollout across the 2 encoder layers
#     (Abnar & Zuidema rollout), visualized as a date x date heatmap plus a
#     per-date "attention received" score, best-effort (skipped gracefully
#     if internal layer attributes differ from the assumed PyTorch layout)
#   * NEW -- occlusion / ablation sensitivity: leave-one-image-date-out and
#     leave-one-modality-out perturbation tests (cheap, gradient-free,
#     causal complement to the IG-based attribution)
#   * NEW -- validation performance figures assembled from the CSVs
#     M7_Transformer.py's training run already wrote (pred-vs-actual by
#     growth stage, residuals, per-stage RMSE/MAE/R2, cross-fold CV summary)
# Everything is aggregated per fold and then across all folds ("FINAL").
#
# Color palette is a SINGLE shared definition (SHAP_BLUE / SHAP_PINK /
# SHAP_PURPLE / neutral grey / grid grey / weather green / CMAP_HEAT), lifted
# unchanged from 25c/25d (they already agreed with each other) so every
# figure in this script -- and every figure from the earlier N% study -- uses
# the same visual language.
#
# Usage (single GPU is enough -- inference/explain only):
#   python M7_explainability.py \
#       --checkpoint_dir ../outputs_24_25/outputs_M7_Transformer \
#       --fold all --n_attr 10 --ig_steps 32
#
# See run_M7_explainability_stratified.sh for a ready sbatch script. For long
# "on_before_closest" image sequences, use --max_frames_explain to cap the
# frames actually explained (Grad-CAM/IG cost scales with sequence length)
# and --max_dates_viz to cap how many per-date PNGs are written to disk
# (per-date attribution SCORES/CSVs still cover every date either way).
# ============================================================

import os
import sys
import json
import argparse
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── import the training modules directly (single source of truth) ──
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import dl_core as dc          # model / dataset / sample-building / stats
import data_prep as dp        # target col, stage labels, MODEL_CONFIGS

BASE = os.path.dirname(THIS_DIR)   # .../Swb_AS_nutrient_prediction

# ─────────────────────────────────────────────────────────────
# SHARED VISUAL STYLE  (single definition -- reused by every figure below)
# ─────────────────────────────────────────────────────────────
SHAP_BLUE    = "#008BFB"
SHAP_PINK    = "#FF0051"
SHAP_PURPLE  = "#7B4DCC"
COLOR_NEUTRAL = "#6e6e6e"
COLOR_GRID    = "#d9d9d9"
COLOR_WEATHER = "#2CA02C"
CMAP_HEAT    = "inferno"

FIG_COLORS = {
    "neg": SHAP_BLUE, "pos": SHAP_PINK, "mid": SHAP_PURPLE,
    "neutral": COLOR_NEUTRAL, "grid": COLOR_GRID,
}
# 3-class stage-aware K status: deficient / adequate / high
CLASS_COLORS = {0: SHAP_BLUE, 1: SHAP_PURPLE, 2: SHAP_PINK}
FIG_CMAP = LinearSegmentedColormap.from_list(
    "k_status", [CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]])
# modality/group palette used for group-contribution + occlusion figures
GROUP_COLORS = {
    "ortho": SHAP_BLUE, "dem": SHAP_PURPLE, "tabular": SHAP_PINK,
    "fert": COLOR_NEUTRAL, "weather": COLOR_WEATHER,
}
GROUP_LABELS = {
    "ortho": "Ortho\n(multispectral)", "dem": "DEM",
    "tabular": "Tabular\nfeatures", "fert": "Fertilizer\nTCN",
    "weather": "Weather\nTCN",
}
STAGE_ORDER = ["transplants", "initial_flower", "initial_harvest",
               "midseason", "end_of_season"]
BAND_LABELS = ["Blue", "Green", "Red", "RedEdge", "NIR"]
CLASS_NAMES = dp.CLASS_NAMES
WEATHER_VAR_COLS = dc.WEATHER_VAR_COLS
N_WEATHER_VARS = dc.N_WEATHER_VARS

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": False, "axes.edgecolor": "#333333",
})


def stage_color(stage: Optional[str]):
    idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
    n = max(len(STAGE_ORDER) - 1, 1)
    return FIG_CMAP(idx / n)


def style_axes(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=COLOR_GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _norm01(a: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def _safe_close():
    try:
        plt.close("all")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Paper-grade explainability for M7_Transformer (K%) checkpoints.")
    # paths -- default to exactly what M7_Transformer.py used
    p.add_argument("--root_images", type=str,
                    default=os.path.join(BASE, "data", "24_25_K"))
    p.add_argument("--ortho_filter", type=str, default="ortho_by_plot")
    p.add_argument("--dem_filter", type=str, default="dem_by_plot")
    p.add_argument("--excel_features_path", type=str,
                    default=os.path.join(BASE, "data/24_25_K/trait_K_weather_yield_combined_conc.xlsx"))
    p.add_argument("--fert_path", type=str,
                    default=os.path.join(BASE, "data/24_25_K/fert_by_plot_daily_lb2kg.xlsx"))
    p.add_argument("--weather_path", type=str,
                    default=os.path.join(BASE, "data/24_25_K/FAWN_report_metric_24_25.xlsx"))
    p.add_argument("--checkpoint_dir", type=str,
                    default=os.path.join(BASE, "outputs_24_25", "outputs_M7_Transformer"),
                    help="Directory with model_M7_Transformer_{task}_fold{N}.pt "
                         "and the training CSVs (cv_results_*, val_predictions_*).")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Default: <checkpoint_dir>/explain_M7_Transformer")
    p.add_argument("--checkpoint_template", type=str,
                    default="model_{model_tag}_{task}_fold{fold}.pt")
    p.add_argument("--model_tag", type=str, default="M7_Transformer")

    p.add_argument("--fold", type=str, default="all", help="1..5 or 'all'.")
    p.add_argument("--task", type=str, default="reg", choices=["reg", "cls"],
                    help="M7 is trained with task_modes=('reg',) only -- the "
                         "cls head (trunk_cls) is never optimized, so 'cls' "
                         "explanations are not meaningful unless you retrained "
                         "with task_modes including 'cls'.")
    p.add_argument("--n_attr", type=int, default=10,
                    help="Validation samples explained per fold (only used when "
                         "--explain_select first).")
    p.add_argument("--explain_select", choices=["balanced", "enriched", "first"],
                    default="balanced",
                    help="Which validation samples to run attribution on. "
                         "'balanced' (default, paper method): pool all folds and draw an "
                         "approximately equal number of deficient/adequate/high samples "
                         "(~--explain_n_total total), subsampling larger classes. "
                         "'enriched': take ALL of any class with <= --explain_take_all_below "
                         "members (every deficiency case) then fill the rest. "
                         "'first': legacy -- first --n_attr samples per fold in fold order. "
                         "Reproducible via --seed.")
    p.add_argument("--explain_n_total", type=int, default=50,
                    help="Target total explained samples across all folds "
                         "(balanced/enriched modes).")
    p.add_argument("--explain_take_all_below", type=int, default=40,
                    help="Enriched mode only: nutrient-status classes with <= this many "
                         "samples are taken in full (e.g. all deficient observations).")
    p.add_argument("--ig_steps", type=int, default=32)
    p.add_argument("--bg_n", type=int, default=10, help="SHAP background samples.")
    p.add_argument("--seed", type=int, default=dc.GLOBAL_SEED)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--max_frames_explain", type=int, default=None,
                    help="Cap image-date sequence length actually fed to the "
                         "model here (keeps the MOST RECENT frames, same rule "
                         "as training's max_frames). None = use the full "
                         "on_before_closest sequence, matching training exactly.")
    p.add_argument("--max_dates_viz", type=int, default=6,
                    help="Cap how many per-date PNGs (Grad-CAM/IG) are saved per "
                         "sample (evenly spaced). Per-date CSV scores always "
                         "cover every date regardless of this cap.")

    p.add_argument("--skip_gradcam", action="store_true")
    p.add_argument("--skip_ig", action="store_true")
    p.add_argument("--skip_shap", action="store_true")
    p.add_argument("--skip_attention", action="store_true")
    p.add_argument("--skip_occlusion", action="store_true")
    p.add_argument("--skip_performance", action="store_true")
    p.add_argument("--skip_final_aggregate", action="store_true")
    return p.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ─────────────────────────────────────────────────────────────
# GLOBAL DATA PREP  (mirrors dl_core.run_cv steps 1-4, no training)
# ─────────────────────────────────────────────────────────────
def prepare_global_data(args) -> dict:
    print(f"[1] Scanning image folders: {args.root_images}", flush=True)
    df_imgs = dc.find_all_paired_tifs(args.root_images, args.ortho_filter, args.dem_filter)
    df_imgs["Image_ID"] = df_imgs["exp_ID"].astype(str) + "_" + df_imgs["plot_num"].astype(str)
    print(f"    image pairs: {len(df_imgs)}", flush=True)

    print(f"[2] Loading labeled table: {args.excel_features_path}", flush=True)
    df_all = dp.load_samples(args.excel_features_path, require_target=True)
    df_all["Image_ID"] = df_all["Image_ID"].astype(str)
    mcfg = dp.MODEL_CONFIGS["M7"]
    feature_cols = dp.get_scalar_feature_columns(df_all, mcfg.scalar_groups)
    print(f"    scalar features ({'+'.join(mcfg.scalar_groups)}): {len(feature_cols)}", flush=True)

    print("[3] Loading fertilizer lookup ...", flush=True)
    fert_lookup, fert_mean, fert_std = None, 0.0, 1.0
    try:
        fert_lookup, max_dap = dc.load_fert_lookup(args.fert_path)
        allv = np.concatenate([v for v in fert_lookup.values()])
        allv = allv[allv > 0]
        fert_mean, fert_std = (float(allv.mean()), max(float(allv.std()), 1e-6)) \
            if allv.size else (0.0, 1.0)
        print(f"    fert lookup: {len(fert_lookup)} plots, max_dap={max_dap}, "
              f"mean={fert_mean:.4f} std={fert_std:.4f}", flush=True)
    except Exception as e:
        print(f"    WARN fert unavailable ({e}); fert TCN will see zeros.", flush=True)

    print("[4] Loading FAWN weather ...", flush=True)
    weather_array, weather_transplant = None, None
    try:
        weather_array, weather_transplant = dc.load_weather_array(args.weather_path)
        print(f"    weather: {weather_array.shape[0]} days x {weather_array.shape[1]} vars "
              f"(season start {weather_transplant})", flush=True)
    except Exception as e:
        print(f"    WARN weather unavailable ({e}); weather TCN will see zeros.", flush=True)

    print("[5] Building M7 image samples (on_before_closest) ...", flush=True)
    df_s = dc.build_on_before_closest_samples(df_imgs, df_all, dc.MIN_IMAGES_PER_SAMPLE)
    df_s = df_s[df_s[dc.TARGET_COL].notna()].copy()
    df_s["value"] = df_s[dc.TARGET_COL].astype(float)
    if "sampling_stage" not in df_s.columns:
        df_s["sampling_stage"] = [dc.infer_sampling_stage(dc._safe_to_date(r["conc_date"]))
                                   for _, r in df_s.iterrows()]
    feature_cols = [c for c in feature_cols if c in df_s.columns]
    print(f"    labeled samples: {len(df_s)}  "
          f"(imgs/sample min={df_s['n_imgs'].min()} max={df_s['n_imgs'].max()})", flush=True)

    crop_hw = dc.compute_global_crop_hw(df_imgs, "ortho_path")
    print(f"    global pad size (H,W): {crop_hw}", flush=True)

    folds = list(GroupKFold(n_splits=dc.FOLDS).split(
        np.arange(len(df_s)), groups=df_s["Image_ID"].values))

    return {
        "df_s": df_s, "feature_cols": feature_cols, "crop_hw": crop_hw,
        "fert_lookup": fert_lookup, "fert_mean": fert_mean, "fert_std": fert_std,
        "weather_array": weather_array, "weather_transplant": weather_transplant,
        "folds": folds,
    }


def build_fold_stats(df_s, tr_idx, feature_cols, fert_lookup, fert_mean, fert_std,
                      weather_array, weather_transplant, crop_hw) -> dict:
    """Recompute the SAME train-fold normalization stats dl_core.train_one_fold
    computed (and did NOT save to disk) -- ortho/dem band stats, y stats,
    tabular feat stats, all fit on the training rows of this exact fold."""
    df_tr = df_s.iloc[tr_idx].copy().reset_index(drop=True)
    feat_mean = (df_tr[feature_cols].apply(pd.to_numeric, errors="coerce")
                 .astype(np.float64).mean(0).values.astype(np.float32)) if feature_cols else None
    feat_std = (df_tr[feature_cols].apply(pd.to_numeric, errors="coerce")
                .astype(np.float64).std(0, ddof=0).values.astype(np.float32)) if feature_cols else None
    ortho_mean, ortho_std = dc.compute_band_stats_listcol(df_tr, "ortho_paths", 5)
    dem_mean, dem_std = dc.compute_scalar_stats_listcol(df_tr, "dem_paths")
    y_mean, y_std = dc.compute_target_stats(df_tr)
    return {
        "ortho_mean": ortho_mean, "ortho_std": ortho_std,
        "dem_mean": dem_mean, "dem_std": dem_std,
        "y_mean": y_mean, "y_std": y_std,
        "feature_cols": feature_cols, "feat_mean": feat_mean, "feat_std": feat_std,
        "fert_lookup": fert_lookup, "fert_mean": fert_mean, "fert_std": fert_std,
        "weather_array": weather_array, "weather_transplant": weather_transplant,
        "crop_hw": crop_hw,
    }


def load_model_from_checkpoint(ckpt_path: str, n_features: int, device) -> dc.DualTaskModel:
    model = dc.DualTaskModel(
        ortho_backbone="resnet18", dem_backbone="convnext_tiny",
        ortho_ch=5, dem_ch=1, n_features=n_features, hidden=128,
        mlp_head_hidden=(256, 64), mlp_head_drop=0.2, n_classes=dc.N_CLASSES,
        use_image_temporal=True, use_sequences=True,
        tcn_fert_out=dc.TCN_FERT_OUT, tcn_weather_out=dc.TCN_WEATHER_OUT,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"    [WARN] missing keys ({len(missing)}): {missing[:5]}", flush=True)
    if unexpected:
        print(f"    [WARN] unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)
    model.eval()
    return model


def _set_rnn_train_only(model: nn.Module):
    """eval() everywhere except any RNN/GRU/LSTM (none in M7, kept for parity
    with 25c/25d and future-proofing); needed so cudnn backward works under
    eval() dropout/BN while still allowing gradient hooks to fire cleanly."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            m.train()


def frame_slice(paths: list, max_frames: Optional[int]) -> list:
    """Same 'keep the most recent frames' rule as OrthoDemTabularDataset."""
    if max_frames is not None and len(paths) > max_frames:
        return paths[-max_frames:]
    return paths


def evenly_spaced_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(set(idx.tolist()))


# ─────────────────────────────────────────────────────────────
# GRAD-CAM  (ortho_cnn.layer4[-1] / dem_cnn.features[-1])
# ─────────────────────────────────────────────────────────────
def _get_gradcam_layer(model: dc.DualTaskModel, branch: str):
    backbone = model.ortho_cnn if branch == "ortho" else model.dem_cnn
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]
    if hasattr(backbone, "features"):
        return backbone.features[-1]
    raise ValueError(f"Cannot find last conv layer for Grad-CAM in {branch} backbone.")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._fwd)
        self._bh = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o

    def _bwd(self, m, gi, go):
        self.gradients = go[0]

    def __call__(self, x_ortho, x_dem, x_feat, seq_len, x_time=None,
                 x_fert=None, x_weather=None, task="reg", cls_idx=None):
        _set_rnn_train_only(self.model)
        self.model.zero_grad(set_to_none=True)
        out_reg, out_cls = self.model(x_ortho, x_dem, x_feat, seq_len,
                                       x_time=x_time, x_fert=x_fert, x_weather=x_weather)
        score = out_reg[0, 0] if task == "reg" else out_cls[0, cls_idx or 0]
        score.backward()
        acts, grads = self.activations, self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        B, T = x_ortho.size(0), x_ortho.size(1)
        cam = cam.view(B, T, cam.size(-2), cam.size(-1))
        valid_t = int(seq_len[0].item())
        cam = cam[0, :valid_t]
        cmin = cam.flatten(1).min(dim=1)[0][:, None, None]
        cmax = cam.flatten(1).max(dim=1)[0][:, None, None]
        cam = (cam - cmin) / (cmax - cmin + 1e-8)
        return cam.detach()

    def close(self):
        self._fh.remove()
        self._bh.remove()


# ─────────────────────────────────────────────────────────────
# INTEGRATED GRADIENTS  (multi-input: ortho, dem, tabular, fert, weather)
# ─────────────────────────────────────────────────────────────
def integrated_gradients_multiinput(model, x_ortho, x_dem, x_feat, x_time=None,
                                     x_fert=None, x_weather=None, seq_len=None,
                                     steps=32, task="reg", cls_idx=None):
    model.eval()
    b_ortho = torch.zeros_like(x_ortho)
    b_dem = torch.zeros_like(x_dem)
    b_feat = None if x_feat is None else torch.zeros_like(x_feat)
    b_fert = None if x_fert is None else torch.zeros_like(x_fert)
    b_weather = None if x_weather is None else torch.zeros_like(x_weather)

    g_ortho = torch.zeros_like(x_ortho)
    g_dem = torch.zeros_like(x_dem)
    g_feat = None if x_feat is None else torch.zeros_like(x_feat)
    g_fert = None if x_fert is None else torch.zeros_like(x_fert)
    g_weather = None if x_weather is None else torch.zeros_like(x_weather)

    for i in range(1, steps + 1):
        a = float(i) / steps
        xo = (b_ortho + a * (x_ortho - b_ortho)).detach().requires_grad_(True)
        xd = (b_dem + a * (x_dem - b_dem)).detach().requires_grad_(True)
        xf = None if x_feat is None else (b_feat + a * (x_feat - b_feat)).detach().requires_grad_(True)
        xfert = None if x_fert is None else (b_fert + a * (x_fert - b_fert)).detach().requires_grad_(True)
        xweath = None if x_weather is None else (b_weather + a * (x_weather - b_weather)).detach().requires_grad_(True)
        _set_rnn_train_only(model)
        model.zero_grad(set_to_none=True)
        out_reg, out_cls = model(xo, xd, xf, seq_len, x_time=x_time, x_fert=xfert, x_weather=xweath)
        s = out_reg[0, 0] if task == "reg" else out_cls[0, cls_idx or 0]
        s.backward()
        g_ortho += xo.grad.detach()
        g_dem += xd.grad.detach()
        if x_feat is not None: g_feat += xf.grad.detach()
        if x_fert is not None: g_fert += xfert.grad.detach()
        if x_weather is not None: g_weather += xweath.grad.detach()

    attr_ortho = (x_ortho - b_ortho) * (g_ortho / steps)
    attr_dem = (x_dem - b_dem) * (g_dem / steps)
    attr_feat = None if x_feat is None else (x_feat - b_feat) * (g_feat / steps)
    attr_fert = None if x_fert is None else (x_fert - b_fert) * (g_fert / steps)
    attr_weather = None if x_weather is None else (x_weather - b_weather) * (g_weather / steps)
    return attr_ortho, attr_dem, attr_feat, attr_fert, attr_weather


def summarize_group_contributions(attr_o, attr_d, attr_f, attr_fert, attr_weather=None, roi_mask=None):
    if roi_mask is not None:
        m = roi_mask[0, 0].bool()
        ortho_score = float(attr_o.abs()[..., m].sum())
        dem_score = float(attr_d.abs()[..., m].sum())
    else:
        ortho_score = float(attr_o.abs().sum())
        dem_score = float(attr_d.abs().sum())
    tabular_score = float(attr_f.abs().sum()) if attr_f is not None else 0.0
    fert_score = float(attr_fert.abs().sum()) if attr_fert is not None else 0.0
    weather_score = float(attr_weather.abs().sum()) if attr_weather is not None else 0.0
    return ortho_score, dem_score, tabular_score, fert_score, weather_score


# ─────────────────────────────────────────────────────────────
# NEW -- ATTENTION CAPTURE (pooling weights + Transformer self-attention)
# ─────────────────────────────────────────────────────────────
class AttentionCapture:
    """Registers hooks around model.temporal_transformer to capture both its
    INPUT (post frame_proj, pre-Transformer) and OUTPUT (pre attn_pool) on the
    next forward pass, with no effect on the forward computation itself."""

    def __init__(self, model: dc.DualTaskModel):
        self.x_in = None
        self.z_out = None
        self._h_pre = model.temporal_transformer.register_forward_pre_hook(self._pre)
        self._h_post = model.temporal_transformer.register_forward_hook(self._post)

    def _pre(self, m, inputs):
        self.x_in = inputs[0].detach()

    def _post(self, m, inputs, output):
        self.z_out = output.detach()

    def close(self):
        self._h_pre.remove()
        self._h_post.remove()


def attention_pool_weights(model: dc.DualTaskModel, z_out: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """Exact readout of MaskedAttentionPooling's softmax weights per frame,
    reusing the trained attn_pool.score submodule -- not an approximation."""
    logits = model.attn_pool.score(z_out).squeeze(-1)
    T = z_out.size(1)
    mask = torch.arange(T, device=z_out.device)[None, :] >= seq_len[:, None]
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=1)


def _ffn_block(layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> torch.Tensor:
    return layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))


def _encoder_layer_forward_with_attn(layer: nn.TransformerEncoderLayer, x, key_padding_mask):
    norm_first = getattr(layer, "norm_first", False)
    if norm_first:
        xn = layer.norm1(x)
        attn_out, attn_w = layer.self_attn(xn, xn, xn, key_padding_mask=key_padding_mask,
                                            need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(attn_out)
        x = x + layer.dropout2(_ffn_block(layer, layer.norm2(x)))
    else:
        attn_out, attn_w = layer.self_attn(x, x, x, key_padding_mask=key_padding_mask,
                                            need_weights=True, average_attn_weights=True)
        x = layer.norm1(x + layer.dropout1(attn_out))
        x = layer.norm2(x + layer.dropout2(_ffn_block(layer, x)))
    return x, attn_w


def transformer_attention_rollout(model: dc.DualTaskModel, x_in: torch.Tensor,
                                   seq_len: torch.Tensor) -> Optional[np.ndarray]:
    """Best-effort re-play of the 2-layer temporal_transformer using each
    layer's own self_attn(..., need_weights=True) to extract per-layer
    attention, then Abnar & Zuidema-style rollout across layers. Uses only
    PUBLIC nn.TransformerEncoderLayer attributes (self_attn/linear1/linear2/
    norm1/norm2/dropout*/activation), which have been stable for years, but
    is still wrapped in try/except by the caller since it depends on the
    encoder's internal layout matching this assumed (post-norm) structure."""
    T = x_in.size(1)
    key_padding_mask = torch.arange(T, device=x_in.device)[None, :] >= seq_len[:, None]
    h = x_in
    layer_attns = []
    for layer in model.temporal_transformer.layers:
        h, attw = _encoder_layer_forward_with_attn(layer, h, key_padding_mask)
        layer_attns.append(attw.detach())
    valid_t = int(seq_len[0].item())
    eye = torch.eye(valid_t, device=x_in.device)
    roll = eye.clone()
    for attw in layer_attns:
        a = attw[0, :valid_t, :valid_t]
        a = a + eye
        a = a / a.sum(-1, keepdim=True)
        roll = a @ roll
    return roll.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────
# NEW -- OCCLUSION / ABLATION SENSITIVITY
# (gradient-free, causal complement to IG's attribution-mass estimate)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_reg(model, x_ortho, x_dem, x_feat, seq_len, x_time, x_fert, x_weather, y_mean, y_std):
    out_reg, _ = model(x_ortho, x_dem, x_feat, seq_len, x_time=x_time, x_fert=x_fert, x_weather=x_weather)
    return float(out_reg[0, 0].item()) * y_std + y_mean


@torch.no_grad()
def leave_one_date_out(model, x_ortho, x_dem, x_feat, seq_len, x_time, x_fert, x_weather,
                        y_mean, y_std, valid_t) -> np.ndarray:
    """Zero one image date at a time (ortho+dem) and record |Delta prediction|."""
    base = _predict_reg(model, x_ortho, x_dem, x_feat, seq_len, x_time, x_fert, x_weather, y_mean, y_std)
    deltas = np.zeros(valid_t, dtype=np.float32)
    for t in range(valid_t):
        xo = x_ortho.clone(); xo[0, t] = 0.0
        xd = x_dem.clone(); xd[0, t] = 0.0
        pred = _predict_reg(model, xo, xd, x_feat, seq_len, x_time, x_fert, x_weather, y_mean, y_std)
        deltas[t] = abs(pred - base)
    return deltas


@torch.no_grad()
def leave_one_group_out(model, x_ortho, x_dem, x_feat, seq_len, x_time, x_fert, x_weather,
                         y_mean, y_std) -> Dict[str, float]:
    """Zero one whole modality group at a time; |Delta prediction| per group."""
    base = _predict_reg(model, x_ortho, x_dem, x_feat, seq_len, x_time, x_fert, x_weather, y_mean, y_std)
    out = {}
    out["ortho"] = abs(_predict_reg(model, torch.zeros_like(x_ortho), x_dem, x_feat, seq_len,
                                     x_time, x_fert, x_weather, y_mean, y_std) - base)
    out["dem"] = abs(_predict_reg(model, x_ortho, torch.zeros_like(x_dem), x_feat, seq_len,
                                   x_time, x_fert, x_weather, y_mean, y_std) - base)
    if x_feat is not None and x_feat.numel() > 0:
        out["tabular"] = abs(_predict_reg(model, x_ortho, x_dem, torch.zeros_like(x_feat), seq_len,
                                           x_time, x_fert, x_weather, y_mean, y_std) - base)
    if x_fert is not None:
        out["fert"] = abs(_predict_reg(model, x_ortho, x_dem, x_feat, seq_len,
                                        x_time, torch.zeros_like(x_fert), x_weather, y_mean, y_std) - base)
    if x_weather is not None:
        out["weather"] = abs(_predict_reg(model, x_ortho, x_dem, x_feat, seq_len,
                                           x_time, x_fert, torch.zeros_like(x_weather), y_mean, y_std) - base)
    return out


# ─────────────────────────────────────────────────────────────
# SHAP (tabular branch only; images/fert/weather/time held fixed)
# ─────────────────────────────────────────────────────────────
class _TabWrapper(nn.Module):
    def __init__(self, model, xo, xd, seq_len, x_time_fixed=None, x_fert_fixed=None,
                 x_weather_fixed=None, task="reg", cls_idx=None):
        super().__init__()
        self.model = model
        self.xo, self.xd, self.seq_len = xo, xd, seq_len
        self.x_time_fixed, self.x_fert_fixed, self.x_weather_fixed = x_time_fixed, x_fert_fixed, x_weather_fixed
        self.task, self.cls_idx = task, cls_idx

    def forward(self, x_feat):
        B = x_feat.shape[0]
        xo = self.xo.expand(B, *self.xo.shape[1:]).contiguous()
        xd = self.xd.expand(B, *self.xd.shape[1:]).contiguous()
        sl = self.seq_len.expand(B)
        xfert = None if self.x_fert_fixed is None else self.x_fert_fixed.expand(B, *self.x_fert_fixed.shape[1:]).contiguous()
        xweath = None if self.x_weather_fixed is None else self.x_weather_fixed.expand(B, *self.x_weather_fixed.shape[1:]).contiguous()
        xtime = None if self.x_time_fixed is None else self.x_time_fixed.expand(B, *self.x_time_fixed.shape[1:]).contiguous()
        out_reg, out_cls = self.model(xo, xd, x_feat, sl, x_time=xtime, x_fert=xfert, x_weather=xweath)
        return out_reg if self.task == "reg" else out_cls


def run_shap_tabular(model, device, x_ortho_fixed, x_dem_fixed, seq_len, bg_feats_np, x_feat_np,
                      out_dir, feature_names, task="reg", cls_idx=None, max_display=20,
                      x_time_fixed=None, x_fert_fixed=None, x_weather_fixed=None,
                      sample_id: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    wrapped = _TabWrapper(model, x_ortho_fixed, x_dem_fixed, seq_len, x_time_fixed=x_time_fixed,
                          x_fert_fixed=x_fert_fixed, x_weather_fixed=x_weather_fixed,
                          task=task, cls_idx=cls_idx).to(device)
    _set_rnn_train_only(wrapped)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bg_full = torch.from_numpy(bg_feats_np.astype(np.float32)).to(device)
    xf = torch.from_numpy(x_feat_np.astype(np.float32)).to(device)
    if xf.ndim == 1: xf = xf.unsqueeze(0)

    # Run SHAP; on CUDA OOM (heavy image sequences), free memory and retry
    # with a progressively smaller background reference set instead of crashing.
    raw_shap = None
    bg_n_full = bg_full.shape[0]
    for bg_try in [bg_n_full, max(bg_n_full // 2, 1), 1]:
        try:
            bg = bg_full[:bg_try]
            with torch.backends.cudnn.flags(enabled=False):
                explainer = shap.GradientExplainer(wrapped, bg)
                raw_shap = explainer.shap_values(xf)
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if bg_try == 1:
                raise
            print(f"    SHAP OOM with bg={bg_try}; retrying with smaller background", flush=True)

    n_features = len(feature_names)
    if isinstance(raw_shap, list):
        shap_vals = np.array(raw_shap[cls_idx if (task == "cls" and cls_idx is not None) else 0], dtype=np.float32)
    else:
        shap_vals = np.array(raw_shap, dtype=np.float32)
    shap_vals = np.squeeze(shap_vals)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals.mean(axis=0) if shap_vals.shape[1] == n_features else shap_vals.mean(axis=1)
    shap_vals = np.asarray(shap_vals, dtype=np.float32).reshape(-1)
    if shap_vals.shape[0] != n_features:
        raise ValueError(f"SHAP length {shap_vals.shape[0]} != n_features {n_features}")

    x_feat_1d = np.asarray(x_feat_np, dtype=np.float32).reshape(-1)
    base_value = 0.0
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            idx = cls_idx if (task == "cls" and cls_idx is not None) else 0
            base_value = float(np.asarray(ev).reshape(-1)[idx])
        else:
            base_value = float(ev)
    except Exception:
        pass

    df_long = pd.DataFrame({"feature": feature_names, "feature_value": x_feat_1d,
                             "shap_value": shap_vals, "abs_shap_value": np.abs(shap_vals)})
    if sample_id is not None:
        df_long["sample_id"] = str(sample_id)
    df_long.to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)

    shap.summary_plot(shap_vals.reshape(1, -1), features=x_feat_1d.reshape(1, -1),
                       feature_names=feature_names, plot_type="bar", show=False, max_display=max_display)
    plt.savefig(os.path.join(out_dir, "shap_summary_bar.png"), dpi=300, bbox_inches="tight"); _safe_close()

    exp = shap.Explanation(values=shap_vals, base_values=base_value, data=x_feat_1d, feature_names=feature_names)
    try:
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        plt.savefig(os.path.join(out_dir, "shap_waterfall.png"), dpi=300, bbox_inches="tight")
    except Exception as e:
        print(f"    waterfall skipped: {e}")
    _safe_close()
    return shap_vals


# ─────────────────────────────────────────────────────────────
# FIGURE SAVERS -- image saliency / Grad-CAM
# ─────────────────────────────────────────────────────────────
def save_ig_maps(attr_ortho, attr_dem, out_dir: str, prefix: str, roi_mask=None):
    os.makedirs(out_dir, exist_ok=True)
    if attr_ortho.dim() == 5: attr_ortho = attr_ortho[:, -1]
    if attr_dem.dim() == 5: attr_dem = attr_dem[:, -1]
    ortho_abs = attr_ortho[0].abs()
    ortho_map = ortho_abs.sum(dim=0)
    dem_map = attr_dem[0, 0].abs()
    if roi_mask is not None:
        m = roi_mask[0, 0].bool()
        ortho_abs = ortho_abs * m.unsqueeze(0); ortho_map = ortho_map * m; dem_map = dem_map * m

    n_bands = ortho_abs.shape[0]
    band_scores = []
    for c in range(n_bands):
        arr = _norm01(ortho_abs[c].detach().cpu().numpy())
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr, cmap=CMAP_HEAT); ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{BAND_LABELS[c]}_ig.png"), dpi=300, bbox_inches="tight")
        _safe_close()
        vals = ortho_abs[c][m] if roi_mask is not None else ortho_abs[c].reshape(-1)
        band_scores.append(float(vals.mean()))

    fig, axes = plt.subplots(1, n_bands + 2, figsize=(3.2 * (n_bands + 2), 3.5))
    for c in range(n_bands):
        axes[c].imshow(_norm01(ortho_abs[c].detach().cpu().numpy()), cmap=CMAP_HEAT)
        axes[c].set_title(BAND_LABELS[c], fontsize=9); axes[c].axis("off")
    axes[n_bands].imshow(_norm01(ortho_map.detach().cpu().numpy()), cmap=CMAP_HEAT)
    axes[n_bands].set_title("Ortho (sum)", fontsize=9); axes[n_bands].axis("off")
    axes[n_bands + 1].imshow(_norm01(dem_map.detach().cpu().numpy()), cmap=CMAP_HEAT)
    axes[n_bands + 1].set_title("DEM", fontsize=9); axes[n_bands + 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ig_band_saliency_combined.png"), dpi=300, bbox_inches="tight")
    _safe_close()

    pd.DataFrame({"band": BAND_LABELS[:n_bands], "mean_abs_attr": band_scores}) \
        .sort_values("mean_abs_attr", ascending=False) \
        .to_csv(os.path.join(out_dir, f"{prefix}_band_scores.csv"), index=False)


def save_gradcam_figure(cam_np, ortho_raw, out_dir, prefix, rgb_bands=(2, 1, 0)):
    os.makedirs(out_dir, exist_ok=True)
    r, g, b = rgb_bands
    rgb = np.stack([_norm01(np.nan_to_num(ortho_raw[r], nan=0.)),
                     _norm01(np.nan_to_num(ortho_raw[g], nan=0.)),
                     _norm01(np.nan_to_num(ortho_raw[b], nan=0.))], axis=-1)
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(rgb); ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ortho_rgb.png"), dpi=300, bbox_inches="tight"); _safe_close()

    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cam_np, cmap=CMAP_HEAT)
    ax.set_title("Grad-CAM", fontsize=9); ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ortho_cam.png"), dpi=300, bbox_inches="tight"); _safe_close()

    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(rgb); ax.imshow(cam_np, cmap=CMAP_HEAT, alpha=0.5)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ortho_overlay.png"), dpi=300, bbox_inches="tight"); _safe_close()


def save_dem_gradcam_figure(cam_np, dem_raw, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    dem_gray = _norm01(np.nan_to_num(dem_raw[0], nan=0.0))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(dem_gray, cmap="gray")
    ax.set_title("DEM", fontsize=9); ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_dem_gray.png"), dpi=300, bbox_inches="tight"); _safe_close()

    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cam_np, cmap=CMAP_HEAT)
    ax.set_title("DEM Grad-CAM", fontsize=9); ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_dem_cam.png"), dpi=300, bbox_inches="tight"); _safe_close()

    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(dem_gray, cmap="gray")
    ax.imshow(cam_np, cmap=CMAP_HEAT, alpha=0.5); ax.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_dem_overlay.png"), dpi=300, bbox_inches="tight"); _safe_close()


def aggregate_band_scores(csv_paths: List[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    dfs = [pd.read_csv(p) for p in csv_paths if os.path.exists(p)]
    if not dfs: return
    df = pd.concat(dfs).groupby("band")["mean_abs_attr"].agg(["mean", "std"]).reset_index()
    df = df.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["band"], df["mean"], yerr=df["std"], color=SHAP_BLUE, capsize=4, linewidth=0.6, edgecolor="k")
    ax.set_ylabel("Mean |IG attribution|"); ax.set_xlabel("Spectral band")
    ax.set_title("Band importance (IG, held-out aggregate)"); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_agg_band_importance.png"), dpi=300, bbox_inches="tight")
    _safe_close()
    df.to_csv(os.path.join(out_dir, f"{prefix}_agg_band_scores.csv"), index=False)


def aggregate_group_contributions(rows_group: list, out_dir: str, prefix: str):
    if not rows_group: return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows_group)
    df.to_csv(os.path.join(out_dir, f"{prefix}_group_contributions.csv"), index=False)
    pct_cols = [c for c in ["ortho_pct", "dem_pct", "tabular_pct", "fert_pct", "weather_pct"] if c in df.columns]
    means = df[pct_cols].mean()
    keys = [c.replace("_pct", "") for c in pct_cols]
    labels = [GROUP_LABELS[k] for k in keys]
    colors = [GROUP_COLORS[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means.values, color=colors, edgecolor="k", linewidth=0.7)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean contribution (%)"); ax.set_title("Group-level IG contributions")
    ax.set_ylim(0, max(means.values) * 1.25); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_group_contributions.png"), dpi=300, bbox_inches="tight")
    _safe_close()


def aggregate_shap(shap_csv_paths: List[str], feature_names: List[str], out_dir: str, prefix: str,
                    max_display: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    long_dfs, shap_rows, feat_rows = [], [], []
    for p in shap_csv_paths:
        if not os.path.exists(p): continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"feature", "shap_value"}.issubset(df.columns): continue
        sample_name = Path(p).parent.name
        if "sample_id" in df.columns and df["sample_id"].notna().any():
            sample_name = str(df["sample_id"].dropna().iloc[0])
        df = df.copy(); df["sample_id"] = sample_name
        long_dfs.append(df)
        smap = dict(zip(df["feature"], df["shap_value"]))
        shap_rows.append([float(smap.get(f, 0.0)) for f in feature_names])
        if "feature_value" in df.columns:
            fmap = dict(zip(df["feature"], df["feature_value"]))
            feat_rows.append([float(fmap.get(f, 0.0)) for f in feature_names])
    if not long_dfs:
        print(f"    aggregate_shap: no valid SHAP CSVs for {prefix}")
        return
    combined = pd.concat(long_dfs, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, f"{prefix}_all_shap_long.csv"), index=False)

    shap_mat = np.asarray(shap_rows, dtype=np.float32)
    feat_mat = np.asarray(feat_rows, dtype=np.float32) if len(feat_rows) == len(shap_rows) else np.zeros_like(shap_mat)

    agg = combined.groupby("feature")["shap_value"].agg(["mean", "std"]).reset_index()
    agg["abs_mean"] = combined.groupby("feature")["shap_value"].apply(lambda s: float(np.mean(np.abs(s)))).values
    agg = agg.sort_values("abs_mean", ascending=False)
    agg.to_csv(os.path.join(out_dir, f"{prefix}_agg_shap.csv"), index=False)

    try:
        agg_bar = agg.sort_values("mean", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, max(4, len(agg_bar) * 0.28)))
        colors = [SHAP_PINK if v > 0 else SHAP_BLUE for v in agg_bar["mean"]]
        ax.barh(agg_bar["feature"], agg_bar["mean"], color=colors, edgecolor="none")
        ax.axvline(0, color="k", linewidth=0.7); ax.invert_yaxis()
        ax.set_xlabel("Mean SHAP value"); ax.set_title("Tabular SHAP -- aggregate signed effect")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_agg_shap_bar.png"), dpi=300, bbox_inches="tight")
    except Exception as e:
        print(f"    signed SHAP bar failed: {e}")
    _safe_close()

    try:
        exp_many = shap.Explanation(values=shap_mat, data=feat_mat, feature_names=feature_names)
        shap.plots.beeswarm(exp_many, max_display=max_display, show=False)
        plt.savefig(os.path.join(out_dir, f"{prefix}_agg_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
    except Exception as e:
        print(f"    beeswarm failed: {e}")
    _safe_close()


# ─────────────────────────────────────────────────────────────
# FIGURE SAVERS -- NEW: attention (pooling + rollout)
# ─────────────────────────────────────────────────────────────
def save_attention_pool_figure(weights: np.ndarray, image_dates: List[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"image_date": image_dates, "attn_pool_weight": weights}) \
        .to_csv(os.path.join(out_dir, f"{prefix}_attn_pool_weights.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(image_dates, weights, color=SHAP_PURPLE, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("Attention-pool weight"); ax.set_xlabel("UAV image date")
    ax.set_title("Which image dates the temporal Transformer weighted most")
    ax.tick_params(axis="x", rotation=45); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_attn_pool_weights.png"), dpi=300, bbox_inches="tight")
    _safe_close()


def save_attention_rollout_figure(rollout: np.ndarray, image_dates: List[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    received = rollout.mean(axis=0)  # average column = attention received from all dates
    pd.DataFrame({"image_date": image_dates, "attn_rollout_received": received}) \
        .to_csv(os.path.join(out_dir, f"{prefix}_attn_rollout_received.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(rollout, cmap=CMAP_HEAT, aspect="auto")
    ax.set_xticks(range(len(image_dates))); ax.set_yticks(range(len(image_dates)))
    ax.set_xticklabels(image_dates, rotation=90, fontsize=6)
    ax.set_yticklabels(image_dates, fontsize=6)
    ax.set_xlabel("attended-to date"); ax.set_ylabel("attending-from date")
    ax.set_title("Transformer self-attention rollout (2 layers)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_attn_rollout_heatmap.png"), dpi=300, bbox_inches="tight")
    _safe_close()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(image_dates, received, color=SHAP_PURPLE, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("Mean attention received (rollout)"); ax.set_xlabel("UAV image date")
    ax.set_title("Attention rollout -- per-date attention received")
    ax.tick_params(axis="x", rotation=45); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_attn_rollout_received.png"), dpi=300, bbox_inches="tight")
    _safe_close()


# ─────────────────────────────────────────────────────────────
# FIGURE SAVERS -- NEW: occlusion sensitivity
# ─────────────────────────────────────────────────────────────
def save_occlusion_date_figure(deltas: np.ndarray, image_dates: List[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"image_date": image_dates, "abs_delta_pred": deltas}) \
        .to_csv(os.path.join(out_dir, f"{prefix}_occlusion_date_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(image_dates, deltas, color=COLOR_NEUTRAL, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("|change in prediction| when date occluded")
    ax.set_xlabel("UAV image date"); ax.set_title("Leave-one-date-out sensitivity")
    ax.tick_params(axis="x", rotation=45); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_occlusion_date_scores.png"), dpi=300, bbox_inches="tight")
    _safe_close()


def save_occlusion_group_figure(group_deltas: Dict[str, float], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    keys = list(group_deltas.keys())
    pd.DataFrame({"group": keys, "abs_delta_pred": [group_deltas[k] for k in keys]}) \
        .to_csv(os.path.join(out_dir, f"{prefix}_occlusion_group_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([GROUP_LABELS[k] for k in keys], [group_deltas[k] for k in keys],
           color=[GROUP_COLORS[k] for k in keys], edgecolor="k", linewidth=0.7)
    ax.set_ylabel("|change in prediction| when group occluded")
    ax.set_title("Leave-one-modality-out sensitivity"); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_occlusion_group_scores.png"), dpi=300, bbox_inches="tight")
    _safe_close()


# ─────────────────────────────────────────────────────────────
# FIGURE SAVERS -- fertilizer / weather TCN attribution (per sample)
# ─────────────────────────────────────────────────────────────
def save_weather_ig_figures(attr_weather, out_dir_s: str):
    weather_var_importance = attr_weather[0].abs().mean(dim=-1).detach().cpu().numpy()
    df_v = pd.DataFrame({"weather_variable": WEATHER_VAR_COLS, "mean_abs_attr": weather_var_importance}) \
        .sort_values("mean_abs_attr", ascending=False)
    df_v.to_csv(os.path.join(out_dir_s, "weather_ig_variable_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_v["weather_variable"], df_v["mean_abs_attr"], color=COLOR_WEATHER, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("Mean |IG attribution|", fontsize=9); ax.set_title("Weather variable importance (IG)", fontsize=10)
    ax.set_xticklabels(df_v["weather_variable"], rotation=40, ha="right", fontsize=7); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_s, "weather_ig_variable_importance.png"), dpi=300, bbox_inches="tight")
    _safe_close()

    mat = attr_weather[0].abs().detach().cpu().numpy()  # (N_WEATHER_VARS, T_weather)
    T_w = mat.shape[1]
    rows = [{"weather_variable": WEATHER_VAR_COLS[v], "DAP_index": t, "abs_ig_attr": mat[v, t]}
            for v in range(len(WEATHER_VAR_COLS)) for t in range(T_w)]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir_s, "weather_ig_temporal_scores.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect="auto", cmap=CMAP_HEAT)
    ax.set_yticks(np.arange(len(WEATHER_VAR_COLS))); ax.set_yticklabels(WEATHER_VAR_COLS, fontsize=7)
    ax.set_xlabel("DAP", fontsize=9); ax.set_ylabel("Weather variable", fontsize=9)
    ax.set_title("Weather TCN temporal attribution (IG)", fontsize=10)
    tick_idx = np.linspace(0, T_w - 1, min(8, T_w)).astype(int) if T_w > 10 else np.arange(T_w)
    ax.set_xticks(tick_idx); ax.set_xticklabels(tick_idx, fontsize=8)
    plt.colorbar(im, ax=ax).set_label("|IG attribution|", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_s, "weather_ig_temporal_heatmap.png"), dpi=300, bbox_inches="tight")
    _safe_close()

    day_scores = mat.mean(axis=0)
    pd.DataFrame({"DAP_index": np.arange(T_w), "mean_abs_attr": day_scores}) \
        .to_csv(os.path.join(out_dir_s, "weather_ig_day_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(np.arange(T_w), day_scores, color=COLOR_WEATHER, linewidth=1.2)
    ax.fill_between(np.arange(T_w), day_scores, color=COLOR_WEATHER, alpha=0.2)
    ax.set_xlabel("DAP", fontsize=9); ax.set_ylabel("Mean |IG attribution|", fontsize=9)
    ax.set_title("Weather TCN day-level attribution (IG)", fontsize=10); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_s, "weather_ig_day_scores.png"), dpi=300, bbox_inches="tight")
    _safe_close()


def save_fert_ig_figure(attr_fert, out_dir_s: str):
    fert_day_abs = attr_fert[0, 0].abs().detach().cpu().numpy()
    df_f = pd.DataFrame({"DAP_index": np.arange(1, len(fert_day_abs) + 1), "mean_abs_attr": fert_day_abs})
    df_f.to_csv(os.path.join(out_dir_s, "fert_ig_day_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df_f["DAP_index"], df_f["mean_abs_attr"], color=SHAP_BLUE, linewidth=1.2)
    ax.fill_between(df_f["DAP_index"], df_f["mean_abs_attr"], color=SHAP_BLUE, alpha=0.25)
    ax.set_xlabel("DAP (days after transplant)", fontsize=9); ax.set_ylabel("Mean |IG attribution|", fontsize=9)
    ax.set_title("Fertilizer TCN day-level attribution (IG)", fontsize=10); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_s, "fert_ig_day_importance.png"), dpi=300, bbox_inches="tight")
    _safe_close()


def _agg_scored_csvs(csv_paths: List[str], group_cols: List[str], value_col: str) -> Optional[pd.DataFrame]:
    dfs = [pd.read_csv(p) for p in csv_paths if os.path.exists(p)]
    if not dfs: return None
    df = pd.concat(dfs)
    return df.groupby(group_cols)[value_col].agg(["mean", "std"]).reset_index()


def aggregate_weather_ig(sample_dirs: List[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    var_paths = [os.path.join(d, "weather_ig_variable_scores.csv") for d in sample_dirs]
    agg_var = _agg_scored_csvs(var_paths, ["weather_variable"], "mean_abs_attr")
    if agg_var is not None:
        agg_var = agg_var.sort_values("mean", ascending=False)
        agg_var.to_csv(os.path.join(out_dir, f"{prefix}_agg_weather_variable_scores.csv"), index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(agg_var["weather_variable"], agg_var["mean"], yerr=agg_var["std"],
               color=COLOR_WEATHER, edgecolor="k", linewidth=0.6, capsize=3)
        ax.set_ylabel("Mean |IG attribution|"); ax.set_title("Weather variable importance -- aggregate")
        ax.set_xticklabels(agg_var["weather_variable"], rotation=40, ha="right", fontsize=7); style_axes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_agg_weather_variable_importance.png"), dpi=300, bbox_inches="tight")
        _safe_close()

    day_paths = [os.path.join(d, "weather_ig_day_scores.csv") for d in sample_dirs]
    agg_day = _agg_scored_csvs(day_paths, ["DAP_index"], "mean_abs_attr")
    if agg_day is not None:
        agg_day = agg_day.sort_values("DAP_index")
        agg_day.to_csv(os.path.join(out_dir, f"{prefix}_agg_weather_day_scores.csv"), index=False)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(agg_day["DAP_index"], agg_day["mean"], color=COLOR_WEATHER, linewidth=1.2)
        ax.fill_between(agg_day["DAP_index"], agg_day["mean"] - agg_day["std"].fillna(0),
                         agg_day["mean"] + agg_day["std"].fillna(0), color=COLOR_WEATHER, alpha=0.2)
        ax.set_xlabel("DAP"); ax.set_ylabel("Mean |IG attribution|")
        ax.set_title("Weather TCN day-level attribution -- aggregate"); style_axes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_agg_weather_day_scores.png"), dpi=300, bbox_inches="tight")
        _safe_close()


def aggregate_fert_ig(sample_dirs: List[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    day_paths = [os.path.join(d, "fert_ig_day_scores.csv") for d in sample_dirs]
    agg_day = _agg_scored_csvs(day_paths, ["DAP_index"], "mean_abs_attr")
    if agg_day is None: return
    agg_day = agg_day.sort_values("DAP_index")
    agg_day.to_csv(os.path.join(out_dir, f"{prefix}_agg_fert_day_scores.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(agg_day["DAP_index"], agg_day["mean"], color=SHAP_BLUE, linewidth=1.2)
    ax.fill_between(agg_day["DAP_index"], agg_day["mean"] - agg_day["std"].fillna(0),
                     agg_day["mean"] + agg_day["std"].fillna(0), color=SHAP_BLUE, alpha=0.2)
    ax.set_xlabel("DAP (days after transplant)"); ax.set_ylabel("Mean |IG attribution|")
    ax.set_title("Fertilizer TCN day-level attribution -- aggregate"); style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_agg_fert_day_importance.png"), dpi=300, bbox_inches="tight")
    _safe_close()


# ─────────────────────────────────────────────────────────────
# NEW -- VALIDATION PERFORMANCE FIGURES
# (read straight from the CSVs M7_Transformer.py's training run already
#  wrote to checkpoint_dir -- no recomputation, always matches the reported
#  numbers exactly)
# ─────────────────────────────────────────────────────────────
def make_performance_figures(checkpoint_dir: str, out_dir: str, model_tag: str, task: str):
    os.makedirs(out_dir, exist_ok=True)
    all_csv = os.path.join(checkpoint_dir, f"val_predictions_{model_tag}_{task}_ALL.csv")
    if not os.path.exists(all_csv):
        print(f"    [performance] {all_csv} not found -- skipping.")
        return
    df = pd.read_csv(all_csv)
    if not {"y_true_reg", "y_pred_reg"}.issubset(df.columns):
        print("    [performance] missing y_true_reg/y_pred_reg columns -- skipping.")
        return
    yt, yp = df["y_true_reg"].values, df["y_pred_reg"].values
    stages = df["sampling_stage"].astype(str).tolist() if "sampling_stage" in df.columns else ["unknown"] * len(df)

    # pred-vs-actual, colored by growth stage
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for stage in STAGE_ORDER + ["unknown"]:
        mask = np.array([s == stage for s in stages])
        if mask.sum() == 0: continue
        ax.scatter(yt[mask], yp[mask], color=stage_color(stage if stage in STAGE_ORDER else None),
                   s=18, edgecolors="k", linewidths=0.3, alpha=0.85, label=stage)
    mn, mx = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([mn, mx], [mn, mx], "--", color=COLOR_NEUTRAL, linewidth=1)
    m = dp.regression_metrics(yt, yp)
    ax.set_xlabel("Actual K (%)"); ax.set_ylabel("Predicted K (%)")
    ax.set_title(f"{model_tag} pooled OOF | R2={m['R2']:.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f}", fontsize=9)
    ax.legend(fontsize=7, loc="best", framealpha=0.7); style_axes(ax, "both")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_{task}_pred_vs_actual.png"), dpi=300, bbox_inches="tight")
    _safe_close()

    # residuals vs actual
    resid = yp - yt
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [stage_color(s if s in STAGE_ORDER else None) for s in stages]
    ax.scatter(yt, resid, c=colors, s=16, edgecolors="k", linewidths=0.3, alpha=0.85)
    ax.axhline(0, color=COLOR_NEUTRAL, linewidth=1, linestyle="--")
    ax.set_xlabel("Actual K (%)"); ax.set_ylabel("Residual (pred - actual)")
    ax.set_title(f"{model_tag} residuals"); style_axes(ax, "both")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_{task}_residuals.png"), dpi=300, bbox_inches="tight")
    _safe_close()

    # metrics by growth stage
    rows = []
    for stage in STAGE_ORDER:
        mask = np.array([s == stage for s in stages])
        if mask.sum() == 0: continue
        mm = dp.regression_metrics(yt[mask], yp[mask]); mm["stage"] = stage; mm["n"] = int(mask.sum())
        rows.append(mm)
    if rows:
        df_stage = pd.DataFrame(rows)
        df_stage.to_csv(os.path.join(out_dir, f"{model_tag}_{task}_metrics_by_stage.csv"), index=False)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, mc in zip(axes, ["MAE", "RMSE", "R2"]):
            colors = [stage_color(s) for s in df_stage["stage"]]
            ax.bar(df_stage["stage"], df_stage[mc], color=colors, edgecolor="k", linewidth=0.6)
            ax.set_title(mc); ax.tick_params(axis="x", rotation=35); style_axes(ax)
        plt.suptitle(f"{model_tag} metrics by growth stage")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_tag}_{task}_metrics_by_stage.png"), dpi=300, bbox_inches="tight")
        _safe_close()

    # cross-fold CV summary
    cv_csv = os.path.join(checkpoint_dir, f"cv_results_{model_tag}_{task}.csv")
    if os.path.exists(cv_csv):
        df_cv = pd.read_csv(cv_csv)
        per_fold = df_cv[~df_cv["fold"].isin(["mean", "std"])].copy()
        mean_row = df_cv[df_cv["fold"] == "mean"]
        std_row = df_cv[df_cv["fold"] == "std"]
        metric_cols = [c for c in ["MAE", "RMSE", "R2", "sMAPE"] if c in per_fold.columns]
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
        if len(metric_cols) == 1: axes = [axes]
        for ax, mc in zip(axes, metric_cols):
            vals = per_fold[mc].astype(float).values
            ax.bar([f"fold{i}" for i in per_fold["fold"]], vals, color=SHAP_PURPLE, edgecolor="k", linewidth=0.6)
            if len(mean_row):
                mv = float(mean_row[mc].iloc[0])
                sv = float(std_row[mc].iloc[0]) if len(std_row) else 0.0
                ax.axhline(mv, color=SHAP_PINK, linewidth=1.2, linestyle="--", label=f"mean={mv:.3f}+-{sv:.3f}")
                ax.legend(fontsize=7)
            ax.set_title(mc); style_axes(ax)
        plt.suptitle(f"{model_tag} 5-fold CV ({task})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_tag}_{task}_cv_summary.png"), dpi=300, bbox_inches="tight")
        _safe_close()


# ─────────────────────────────────────────────────────────────
# PER-SAMPLE ORCHESTRATION
# ─────────────────────────────────────────────────────────────
def _build_val_dataset(df_va, stats, max_frames):
    return dc.OrthoDemTabularDataset(
        df_va, stats["ortho_mean"], stats["ortho_std"], stats["dem_mean"], stats["dem_std"],
        stats["y_mean"], stats["y_std"], stats["crop_hw"], stats["feature_cols"],
        stats["feat_mean"], stats["feat_std"], use_sequences=True,
        fert_lookup=stats["fert_lookup"], fert_mean=stats["fert_mean"], fert_std=stats["fert_std"],
        weather_array=stats["weather_array"], weather_transplant=stats["weather_transplant"],
        max_frames=max_frames)


def _status_classifier():
    """Return whichever stage-aware nutrient-status classifier data_prep exposes
    (classify_k_value for K seasons, classify_n_value for N), or None."""
    for name in ("classify_k_value", "classify_n_value"):
        fn = getattr(dp, name, None)
        if fn is not None:
            return fn
    return None


def build_explain_selection(df_s, folds, mode, n_total, take_all_below, seed):
    """Select which validation samples to run attribution on (paper method).

    Pools every fold's validation indices (the folds partition the data) and labels each
    sample deficient/adequate/high with the stage-aware nutrient-status thresholds, then:
      mode='balanced'  -> draw ~equal numbers per class (~n_total total), subsampling
                          larger classes; a class with fewer samples contributes all it
                          has and the shortfall is redistributed to classes with capacity.
      mode='enriched'  -> take ALL of any class with <= take_all_below members (every
                          deficiency case), then split the remaining budget across the rest.
    Deterministic given `seed`. Returns {fold_id: [positional df_s index, ...]} ordered
    deficient-first, or None if no classifier is available (caller falls back to 'first').
    """
    clsfn = _status_classifier()
    if clsfn is None:
        print("  [select] no stage-aware classifier in data_prep -- using 'first'.", flush=True)
        return None

    cls_of, by_class = {}, {}
    for _tr, va in folds:
        for gi in va:
            row = df_s.iloc[gi]
            val = row["value"]
            c = clsfn(float(val), row["sampling_stage"]) if val == val else -1  # NaN-safe
            if c in (0, 1, 2):
                cls_of[gi] = c
                by_class.setdefault(c, []).append(gi)

    rng = np.random.RandomState(seed)
    counts = {c: len(v) for c, v in by_class.items()}
    present = sorted(by_class)
    selected = set()
    alloc = {c: 0 for c in present}

    if mode == "enriched":
        rare = [c for c in present if counts[c] <= take_all_below]
        non_rare = [c for c in present if counts[c] > take_all_below]
        for c in rare:
            selected.update(by_class[c])                 # take all of rare classes
        remaining = max(n_total - len(selected), 0)
        if non_rare:
            per = remaining // len(non_rare)
            left = remaining - per * len(non_rare)
            for k, c in enumerate(non_rare):
                alloc[c] = min(per + (1 if k < left else 0), counts[c])
    else:  # 'balanced' -- approximately equal per class
        k = len(present)
        base = n_total // k
        left = n_total - base * k
        for i, c in enumerate(present):
            alloc[c] = min(base + (1 if i < left else 0), counts[c])
        # redistribute any shortfall (a class too small) to classes with capacity
        short = n_total - sum(alloc.values())
        while short > 0:
            progressed = False
            for c in present:
                if short <= 0:
                    break
                if alloc[c] < counts[c]:
                    alloc[c] += 1
                    short -= 1
                    progressed = True
            if not progressed:
                break

    for c, take in alloc.items():
        if take <= 0:
            continue
        pick = by_class[c] if take >= counts[c] else rng.choice(by_class[c], size=take, replace=False)
        selected.update(int(i) for i in pick)

    per_fold = {}
    for fold_id, (_tr, va) in enumerate(folds, start=1):
        sel = [i for i in va if i in selected]
        sel.sort(key=lambda i: (cls_of.get(i, 9), i))  # deficient(0) first
        per_fold[fold_id] = sel

    names = getattr(dp, "CLASS_NAMES", ["deficient", "adequate", "high"])
    bal = ", ".join(f"{names[c]}={sum(1 for i in selected if cls_of.get(i) == c)}/{counts.get(c, 0)}"
                    for c in present)
    print(f"  [select] {mode}: {len(selected)} samples (target~{n_total}, seed={seed}) "
          f"-> {bal}", flush=True)
    return per_fold


def run_explainability_for_fold(model, df_s, va_idx, stats, device, fold_id, task,
                                 explain_base_dir, args, explain_idx=None) -> dict:
    summary = {"fold": fold_id, "task": task, "n_val": int(len(va_idx)), "n_explained": 0, "errors": []}
    feature_names = stats["feature_cols"]
    # 'enriched' mode passes an explicit ordered index list for this fold; explain all
    # of them. Legacy 'first' mode (explain_idx is None) uses the whole val fold and
    # caps at --n_attr below, preserving the original behavior exactly.
    if explain_idx is not None:
        df_va = df_s.iloc[list(explain_idx)].copy().reset_index(drop=True)
    else:
        df_va = df_s.iloc[va_idx].copy().reset_index(drop=True)
    val_ds = _build_val_dataset(df_va, stats, args.max_frames_explain)

    # SHAP background: a handful of val-fold tabular feature vectors
    bg_n = min(args.bg_n, len(val_ds))
    bg_idx = np.random.RandomState(0).choice(len(val_ds), size=bg_n, replace=False) if bg_n > 0 else []
    bg_feats = np.stack([val_ds[i][2].numpy() for i in bg_idx], axis=0).astype(np.float32) \
        if len(bg_idx) else np.zeros((1, len(feature_names)), dtype=np.float32)

    n_explain = len(val_ds) if explain_idx is not None else min(args.n_attr, len(val_ds))
    fold_dir = os.path.join(explain_base_dir, "explanations", task, f"fold{fold_id}")

    rows_group, band_csvs, shap_csvs = [], [], []
    weather_sample_dirs, fert_sample_dirs = [], []

    for j in range(n_explain):
        sample_tag = f"fold{fold_id}/val_{j+1:02d}"
        out_dir_s = os.path.join(fold_dir, f"val_{j+1:02d}")
        os.makedirs(out_dir_s, exist_ok=True)
        try:
            row = df_va.iloc[j]
            ortho_seq, dem_seq, ft, img_time, seq_len_t, y_reg, y_cls, fert_t, weather_t = val_ds[j]
            x_ortho = ortho_seq.unsqueeze(0).to(device)
            x_dem = dem_seq.unsqueeze(0).to(device)
            x_feat = ft.unsqueeze(0).to(device) if ft.numel() else torch.empty(1, 0, device=device)
            x_time = img_time.unsqueeze(0).to(device)
            seq_len = seq_len_t.unsqueeze(0).to(device)
            x_fert = fert_t.unsqueeze(0).to(device)
            x_weather = weather_t.unsqueeze(0).to(device)
            valid_t = int(seq_len.item())

            frame_paths = frame_slice(row["ortho_paths"], args.max_frames_explain)
            image_dates = frame_slice(row.get("image_dates", frame_paths), args.max_frames_explain)
            image_dates = [str(d) for d in image_dates[:valid_t]]
            viz_idx = evenly_spaced_indices(valid_t, args.max_dates_viz)

            roi_np = np.any(np.isfinite(dc._read_multiband_with_mask(frame_paths[-1])), axis=0).astype(np.float32)
            roi_t = dc.pad_to_min_size(torch.from_numpy(roi_np)[None], *stats["crop_hw"])[None].to(device)

            model.eval()
            with torch.no_grad():
                out_reg, out_cls = model(x_ortho, x_dem, x_feat, seq_len, x_time=x_time,
                                          x_fert=x_fert, x_weather=x_weather)
                pred_reg = float(out_reg[0, 0].item()) * stats["y_std"] + stats["y_mean"]
                pred_cls = int(torch.argmax(out_cls[0]).item())
            true_reg = float(row["value"])
            cls_idx_for_attr = pred_cls if task == "cls" else None

            # ── Grad-CAM (ortho + DEM), all dates in one backward pass ──
            if not args.skip_gradcam:
                try:
                    cam_layer = _get_gradcam_layer(model, "ortho")
                    gcam = GradCAM(model, cam_layer)
                    cam = gcam(x_ortho.detach().requires_grad_(True), x_dem, x_feat, seq_len,
                               x_time=x_time, x_fert=x_fert, x_weather=x_weather,
                               task=task, cls_idx=cls_idx_for_attr)
                    cams = cam.detach().cpu().numpy()
                    for t in viz_idx:
                        cam_np = cams[t]
                        ortho_vis = x_ortho[0, t].detach().cpu().numpy()
                        if cam_np.shape != ortho_vis.shape[-2:]:
                            cam_np = F.interpolate(torch.from_numpy(cam_np)[None, None].float(),
                                                    size=ortho_vis.shape[-2:], mode="bilinear",
                                                    align_corners=False)[0, 0].numpy()
                        # NOTE: display uses the padded/normalized model-input tensor
                        # (not a fresh disk read) so it is pixel-aligned with cam_np --
                        # normalization is a monotonic affine map so the RGB composite
                        # still looks correct after the _norm01 min-max stretch.
                        save_gradcam_figure(cam_np, ortho_vis, out_dir_s, f"sample_{j+1:03d}_date{t+1:02d}")
                    cam_agg = cams[:valid_t].sum(axis=0)
                    cam_agg = (cam_agg - cam_agg.min()) / (cam_agg.max() - cam_agg.min() + 1e-8)
                    ortho_vis_last = x_ortho[0, valid_t - 1].detach().cpu().numpy()
                    if cam_agg.shape != ortho_vis_last.shape[-2:]:
                        cam_agg = F.interpolate(torch.from_numpy(cam_agg)[None, None].float(),
                                                 size=ortho_vis_last.shape[-2:], mode="bilinear",
                                                 align_corners=False)[0, 0].numpy()
                    save_gradcam_figure(cam_agg, ortho_vis_last, out_dir_s, f"sample_{j+1:03d}_temporal_aggregate")
                    gcam.close()
                except Exception as e:
                    summary["errors"].append(f"Grad-CAM ortho failed {sample_tag}: {e}")

                try:
                    cam_layer_dem = _get_gradcam_layer(model, "dem")
                    gcam_dem = GradCAM(model, cam_layer_dem)
                    cam = gcam_dem(x_ortho, x_dem.detach().requires_grad_(True), x_feat, seq_len,
                                   x_time=x_time, x_fert=x_fert, x_weather=x_weather,
                                   task=task, cls_idx=cls_idx_for_attr)
                    cams = cam.detach().cpu().numpy()
                    for t in viz_idx:
                        cam_np = cams[t]
                        dem_vis = x_dem[0, t].detach().cpu().numpy()
                        if cam_np.shape != dem_vis.shape[-2:]:
                            cam_np = F.interpolate(torch.from_numpy(cam_np)[None, None].float(),
                                                    size=dem_vis.shape[-2:], mode="bilinear",
                                                    align_corners=False)[0, 0].numpy()
                        save_dem_gradcam_figure(cam_np, dem_vis, out_dir_s, f"sample_{j+1:03d}_DEM_date{t+1:02d}")
                    gcam_dem.close()
                except Exception as e:
                    summary["errors"].append(f"Grad-CAM DEM failed {sample_tag}: {e}")

            # ── Integrated Gradients: image/tabular/fert/weather ──
            if not args.skip_ig:
                try:
                    attr_o, attr_d, attr_f, attr_fert, attr_weather = integrated_gradients_multiinput(
                        model, x_ortho, x_dem, x_feat, x_time=x_time, x_fert=x_fert, x_weather=x_weather,
                        seq_len=seq_len, steps=args.ig_steps, task=task, cls_idx=cls_idx_for_attr)

                    attr_o_agg = attr_o[:, :valid_t].abs().sum(dim=1)
                    attr_d_agg = attr_d[:, :valid_t].abs().sum(dim=1)
                    save_ig_maps(attr_o_agg, attr_d_agg, out_dir_s, "ig_temporal_aggregate", roi_mask=roi_t)
                    for t in viz_idx:
                        save_ig_maps(attr_o[:, t], attr_d[:, t], out_dir_s, f"ig_date{t+1:02d}", roi_mask=roi_t)

                    ortho_scores = attr_o[:, :valid_t].abs().sum(dim=(2, 3, 4))[0].detach().cpu().numpy()
                    dem_scores = attr_d[:, :valid_t].abs().sum(dim=(2, 3, 4))[0].detach().cpu().numpy()
                    tot = ortho_scores + dem_scores
                    pct = 100 * tot / (tot.sum() + 1e-8)
                    df_date = pd.DataFrame({"date_index": np.arange(valid_t), "image_date": image_dates,
                                             "ortho_abs_ig": ortho_scores, "dem_abs_ig": dem_scores,
                                             "image_total_abs_ig": tot, "image_pct": pct})
                    df_date.to_csv(os.path.join(out_dir_s, "image_date_ig_scores.csv"), index=False)
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.bar(df_date["image_date"], df_date["image_pct"], color=SHAP_BLUE, edgecolor="k", linewidth=0.6)
                    ax.set_ylabel("Image attribution (%)"); ax.set_xlabel("UAV image date")
                    ax.set_title("Per-date image contribution (IG)"); ax.tick_params(axis="x", rotation=45)
                    style_axes(ax); plt.tight_layout()
                    plt.savefig(os.path.join(out_dir_s, "image_date_ig_scores.png"), dpi=300, bbox_inches="tight")
                    _safe_close()

                    if attr_weather is not None:
                        save_weather_ig_figures(attr_weather, out_dir_s)
                        weather_sample_dirs.append(out_dir_s)
                    if attr_fert is not None:
                        save_fert_ig_figure(attr_fert, out_dir_s)
                        fert_sample_dirs.append(out_dir_s)

                    bcsv = os.path.join(out_dir_s, "ig_temporal_aggregate_band_scores.csv")
                    if os.path.exists(bcsv): band_csvs.append(bcsv)

                    o_s, d_s, t_s, f_s, w_s = summarize_group_contributions(
                        attr_o[:, :valid_t], attr_d[:, :valid_t], attr_f, attr_fert,
                        attr_weather=attr_weather, roi_mask=roi_t)
                    total = o_s + d_s + t_s + f_s + w_s + 1e-8
                    rows_group.append({
                        "Image_ID": str(row.get("Image_ID", f"val_{j+1}")), "pred_reg": pred_reg,
                        "ortho_pct": 100 * o_s / total, "dem_pct": 100 * d_s / total,
                        "tabular_pct": 100 * t_s / total, "fert_pct": 100 * f_s / total,
                        "weather_pct": 100 * w_s / total,
                    })
                except Exception as e:
                    summary["errors"].append(f"IG failed {sample_tag}: {e}\n{traceback.format_exc(limit=1)}")

            # ── SHAP (tabular) ──
            if not args.skip_shap and x_feat.numel() > 0:
                try:
                    shap_dir = os.path.join(out_dir_s, "shap")
                    run_shap_tabular(model, device, x_ortho, x_dem, seq_len, bg_feats,
                                      x_feat.cpu().numpy(), shap_dir, feature_names, task=task,
                                      cls_idx=cls_idx_for_attr, x_time_fixed=x_time, x_fert_fixed=x_fert,
                                      x_weather_fixed=x_weather,
                                      sample_id=dc.normalize_image_id(row.get("Image_ID", f"val_{j+1}")))
                    scsv = os.path.join(shap_dir, "shap_values.csv")
                    if os.path.exists(scsv): shap_csvs.append(scsv)
                except Exception as e:
                    summary["errors"].append(f"SHAP failed {sample_tag}: {e}")

            # ── NEW: attention pooling + Transformer rollout ──
            if not args.skip_attention:
                try:
                    cap = AttentionCapture(model)
                    with torch.no_grad():
                        model(x_ortho, x_dem, x_feat, seq_len, x_time=x_time, x_fert=x_fert, x_weather=x_weather)
                        w = attention_pool_weights(model, cap.z_out, seq_len)[0, :valid_t].detach().cpu().numpy()
                    save_attention_pool_figure(w, image_dates, out_dir_s, f"sample_{j+1:03d}")
                    try:
                        with torch.no_grad():
                            rollout = transformer_attention_rollout(model, cap.x_in, seq_len)
                        save_attention_rollout_figure(rollout, image_dates, out_dir_s, f"sample_{j+1:03d}")
                    except Exception as e:
                        summary["errors"].append(f"Attention rollout skipped {sample_tag}: {e}")
                    cap.close()
                except Exception as e:
                    summary["errors"].append(f"Attention pooling failed {sample_tag}: {e}")

            # ── NEW: occlusion sensitivity (date + modality group) ──
            if not args.skip_occlusion:
                try:
                    deltas = leave_one_date_out(model, x_ortho, x_dem, x_feat, seq_len, x_time,
                                                 x_fert, x_weather, stats["y_mean"], stats["y_std"], valid_t)
                    save_occlusion_date_figure(deltas, image_dates, out_dir_s, f"sample_{j+1:03d}")
                    group_deltas = leave_one_group_out(model, x_ortho, x_dem, x_feat, seq_len, x_time,
                                                        x_fert, x_weather, stats["y_mean"], stats["y_std"])
                    save_occlusion_group_figure(group_deltas, out_dir_s, f"sample_{j+1:03d}")
                except Exception as e:
                    summary["errors"].append(f"Occlusion failed {sample_tag}: {e}")

            # ── per-sample manifest ──
            with open(os.path.join(out_dir_s, "sample_info.json"), "w") as f:
                json.dump({"Image_ID": str(row.get("Image_ID", "")), "conc_date": str(row.get("conc_date", "")),
                           "sampling_stage": str(row.get("sampling_stage", "")), "n_images": valid_t,
                           "pred_reg": pred_reg, "true_reg": true_reg, "pred_cls": pred_cls}, f, indent=2)
            summary["n_explained"] += 1
        except Exception as e:
            summary["errors"].append(f"Sample {sample_tag} failed completely: {e}\n{traceback.format_exc(limit=2)}")
        finally:
            # Release this sample's GPU tensors before the next sample so memory
            # doesn't accumulate across the fold (prevents CUDA OOM on heavy samples).
            try:
                del x_ortho, x_dem, x_feat, x_time, seq_len, x_fert, x_weather
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── fold-level aggregation ──
    agg_dir = os.path.join(fold_dir, "aggregate")
    prefix = f"{args.model_tag}_{task}_fold{fold_id}"
    aggregate_group_contributions(rows_group, agg_dir, prefix)
    aggregate_band_scores(band_csvs, agg_dir, prefix)
    aggregate_shap(shap_csvs, feature_names, agg_dir, prefix)
    aggregate_weather_ig(weather_sample_dirs, agg_dir, prefix)
    aggregate_fert_ig(fert_sample_dirs, agg_dir, prefix)
    return summary


# ─────────────────────────────────────────────────────────────
# CROSS-FOLD FINAL AGGREGATION
# ─────────────────────────────────────────────────────────────
def run_final_aggregation(explain_base_dir: str, task: str, model_tag: str,
                           fold_ids: List[int], feature_names: List[str]) -> str:
    all_band, all_shap, all_weather_dirs, all_fert_dirs, all_group_dfs = [], [], [], [], []
    for fid in fold_ids:
        fold_dir = os.path.join(explain_base_dir, "explanations", task, f"fold{fid}")
        if not os.path.isdir(fold_dir):
            continue
        for name in sorted(os.listdir(fold_dir)):
            sdir = os.path.join(fold_dir, name)
            if not os.path.isdir(sdir) or name == "aggregate":
                continue
            b = os.path.join(sdir, "ig_temporal_aggregate_band_scores.csv")
            if os.path.exists(b): all_band.append(b)
            s = os.path.join(sdir, "shap", "shap_values.csv")
            if os.path.exists(s): all_shap.append(s)
            if os.path.exists(os.path.join(sdir, "weather_ig_variable_scores.csv")):
                all_weather_dirs.append(sdir)
            if os.path.exists(os.path.join(sdir, "fert_ig_day_scores.csv")):
                all_fert_dirs.append(sdir)
        gcsv = os.path.join(fold_dir, "aggregate", f"{model_tag}_{task}_fold{fid}_group_contributions.csv")
        if os.path.exists(gcsv):
            all_group_dfs.append(pd.read_csv(gcsv))

    final_dir = os.path.join(explain_base_dir, "explanations", task, "FINAL")
    prefix = f"{model_tag}_{task}_FINAL"
    rows_group = (pd.concat(all_group_dfs, ignore_index=True).to_dict("records")
                  if all_group_dfs else [])
    aggregate_group_contributions(rows_group, final_dir, prefix)
    aggregate_band_scores(all_band, final_dir, prefix)
    aggregate_shap(all_shap, feature_names, final_dir, prefix)
    aggregate_weather_ig(all_weather_dirs, final_dir, prefix)
    aggregate_fert_ig(all_fert_dirs, final_dir, prefix)
    return final_dir


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}", flush=True)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "explain_M7_Transformer")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    global_data = prepare_global_data(args)
    df_s, feature_cols = global_data["df_s"], global_data["feature_cols"]
    folds = global_data["folds"]

    if args.fold.lower() == "all":
        fold_ids = list(range(1, dc.FOLDS + 1))
    else:
        fold_ids = [int(args.fold)]
        assert 1 <= fold_ids[0] <= dc.FOLDS, f"--fold must be 1..{dc.FOLDS} or 'all'"

    if args.task == "cls":
        print("WARNING: M7_Transformer.py trains task_modes=('reg',) only -- "
              "trunk_cls is never optimized, so --task cls explanations reflect "
              "an UNTRAINED head unless you retrained M7 with 'cls' included.", flush=True)

    # Attribution-sample selection (paper method vs legacy). Computed once, globally,
    # so the deficiency-complete / class-enriched budget is allocated across all folds.
    explain_selection = None
    if args.explain_select in ("balanced", "enriched"):
        explain_selection = build_explain_selection(
            df_s, folds, args.explain_select, args.explain_n_total,
            args.explain_take_all_below, args.seed)

    all_summaries = []
    seq_lengths_seen = []
    for fold_id in fold_ids:
        tr_idx, va_idx = folds[fold_id - 1]
        print(f"\n{'='*70}\n FOLD {fold_id}/{dc.FOLDS}  task={args.task}\n{'='*70}", flush=True)

        stats = build_fold_stats(df_s, tr_idx, feature_cols, global_data["fert_lookup"],
                                  global_data["fert_mean"], global_data["fert_std"],
                                  global_data["weather_array"], global_data["weather_transplant"],
                                  global_data["crop_hw"])

        ckpt_name = args.checkpoint_template.format(model_tag=args.model_tag, task=args.task, fold=fold_id)
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            msg = f"Checkpoint not found: {ckpt_path}"
            print(f"  {msg}", flush=True)
            all_summaries.append({"fold": fold_id, "task": args.task, "checkpoint": ckpt_path,
                                   "checkpoint_loaded": False, "n_val": int(len(va_idx)),
                                   "n_explained": 0, "errors": [msg]})
            continue

        n_features = len(stats["feature_cols"])
        print(f"  Loading checkpoint: {ckpt_path}  (n_features={n_features})", flush=True)
        model = load_model_from_checkpoint(ckpt_path, n_features, device)

        n_dates_median = int(np.median(df_s.iloc[va_idx]["n_imgs"].values))
        seq_lengths_seen.append(n_dates_median)
        print(f"  Median image-date sequence length in this fold's val set: {n_dates_median} "
              f"(Grad-CAM/IG cost scales with this; use --max_frames_explain to cap if slow)", flush=True)

        fold_summary = run_explainability_for_fold(model, df_s, va_idx, stats, device, fold_id,
                                                     args.task, args.output_dir, args,
                                                     explain_idx=(explain_selection.get(fold_id)
                                                                  if explain_selection else None))
        fold_summary["checkpoint"], fold_summary["checkpoint_loaded"] = ckpt_path, True
        all_summaries.append(fold_summary)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = None
    if not args.skip_final_aggregate:
        print("\n[FINAL] Cross-fold aggregation ...", flush=True)
        final_dir = run_final_aggregation(args.output_dir, args.task, args.model_tag,
                                           fold_ids, feature_cols)
        print(f"  Wrote FINAL aggregate to: {final_dir}", flush=True)

    if not args.skip_performance:
        print("\n[PERFORMANCE] Building validation performance figures ...", flush=True)
        perf_dir = os.path.join(args.output_dir, "performance")
        make_performance_figures(args.checkpoint_dir, perf_dir, args.model_tag, args.task)
        print(f"  Wrote performance figures to: {perf_dir}", flush=True)

    print(f"\n{'='*70}\n EXPLAINABILITY SUMMARY\n{'='*70}", flush=True)
    summary_rows = []
    total_explained = total_errors = 0
    for s in all_summaries:
        n_err = len(s.get("errors", []))
        total_explained += s.get("n_explained", 0); total_errors += n_err
        print(f"  fold={s['fold']:<2} ckpt_loaded={s.get('checkpoint_loaded', False)} "
              f"n_val={s.get('n_val', 0):<4} n_explained={s.get('n_explained', 0):<4} errors={n_err}", flush=True)
        for e in s.get("errors", [])[:3]:
            print(f"      ! {e[:160]}", flush=True)
        summary_rows.append({"fold": s["fold"], "checkpoint": s.get("checkpoint", ""),
                              "checkpoint_loaded": s.get("checkpoint_loaded", False),
                              "n_val": s.get("n_val", 0), "n_explained": s.get("n_explained", 0),
                              "n_errors": n_err})
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, f"explain_summary_{args.task}.csv"), index=False)
    print(f"\n  Total samples explained: {total_explained}\n  Total error messages: {total_errors}", flush=True)
    print(f"  Output folder: {args.output_dir}", flush=True)
    if final_dir: print(f"  FINAL aggregate folder: {final_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
