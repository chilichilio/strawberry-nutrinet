#!/usr/bin/env python3
"""
M7_explainability_fixed.py -- NON-INVASIVE attribution fixes for M7.

Edits NOTHING. Imports M7_explainability.py and M7_explainability_stratified.py
unchanged, monkey-patches four functions, then runs. The class-stratified sample
selection (take-all rare classes) is preserved because this layers ON TOP of
M7_explainability_stratified.

Nothing about the MODEL INPUTS changes, so predictions are bit-identical to the
reported run and Tables 3/4/5 are untouched. Only the attribution maths changes.

--------------------------------------------------------------------------
FIX 1  IG baseline for the image branches            [--no-fix-baseline to skip]
--------------------------------------------------------------------------
dl_core._load_norm does  nan_to_num -> pad -> normalize , so off-plot and
padding pixels end up at (0 - band_mean)/band_std : a large negative CONSTANT,
not the mean. Stock IG uses a zeros baseline, so those dead pixels carry
near-maximal |x - baseline| and dominate the attribution. Measured effect: the
ortho IG mass splits into two disjoint populations (N: 30/50 samples at ~1.6e-3,
18/50 at ~39, only 2 samples in between; 484,000x end to end).

We do NOT touch the inputs (that would put the trained model off-distribution).
We move the BASELINE instead: baseline == input wherever the pixel is dead,
0 (= fold mean) wherever it is real. Dead pixels then contribute EXACTLY zero
by construction, per frame, with no mask file needed -- a dead pixel is exactly
equal to (0 - mean)/std, which we can recompute from stats.

This also removes the last-frame-ROI problem for free: the stock code builds one
ROI from frame_paths[-1] and applies it to every frame, so footprint mismatch
between flights leaked dead pixels into the sum. Detection here is per frame.

--------------------------------------------------------------------------
FIX 2  Grad-CAM sign                                   [--keep-relu to skip]
--------------------------------------------------------------------------
Stock: cam = relu((weights*acts).sum(1)) on a REGRESSION output. ReLU keeps only
regions pushing the prediction UP, so for a plot the model reads as deficient the
evidence is negative and the map goes blank. The K explanation sample is 68%
deficient, so most K maps are being clipped. We drop the ReLU and return a signed
CAM (use a diverging colormap when plotting).

--------------------------------------------------------------------------
FIX 3  Grad-CAM resolution                             [--cam-layer layer4 to skip]
--------------------------------------------------------------------------
Stock hooks backbone.layer4[-1] = stride 32, i.e. a ~7x7 grid bilinearly
stretched over the whole tile. Default here is layer3 (stride 16, 2x finer).

--------------------------------------------------------------------------
FIX 4  Modality shares are per-element, not summed     [--sum-shares to skip]
--------------------------------------------------------------------------
summarize_group_contributions returns sum(|attr|) per branch, and the branches
have wildly different element counts (ortho = T x 5 x H x W ; Fagg-style scalars
= ~70 ; weather = days x 15). Shares therefore track dimensionality. We report
mean |attr| per element instead. Both numbers are written out so they can be
compared.

--------------------------------------------------------------------------
DIAGNOSTIC  IG completeness
--------------------------------------------------------------------------
Stock code takes abs() before summing, which destroys IG's completeness property
and with it the only convergence check on --ig_steps. We log, per sample, the
SIGNED attribution sum against f(x) - f(baseline). If |ratio - 1| is large,
--ig_steps is too low. Written to ig_completeness_check.csv.

--------------------------------------------------------------------------
OUTPUTS -- all new names, nothing overwritten
--------------------------------------------------------------------------
  <ckpt>/explain_M7_Transformer_fixed/          (via --output_dir in the .sh)
  <output_dir>/modality_shares_fixed.csv        per-element AND summed shares
  <output_dir>/ig_completeness_check.csv        convergence diagnostic

USAGE -- same CLI as M7_explainability.py
  python M7_explainability_fixed.py --checkpoint_dir <ckpt> \
      --output_dir <ckpt>/explain_M7_Transformer_fixed \
      --fold all --task reg --ig_steps 64 --bg_n 10 --max_dates_viz 99 \
      --skip_occlusion
"""
import atexit
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import M7_explainability as mod
import M7_explainability_stratified as strat   # keeps the class-balanced budget
import dl_core as dc

# ── knobs (env or argv, so the stock argparse never sees them) ──────────────
_ARGV = list(sys.argv[1:])


def _pop_flag(name: str) -> bool:
    if name in _ARGV:
        _ARGV.remove(name)
        return True
    return False


def _pop_opt(name: str, default: str) -> str:
    if name in _ARGV:
        i = _ARGV.index(name)
        v = _ARGV[i + 1]
        del _ARGV[i:i + 2]
        return v
    return default


FIX_BASELINE = not _pop_flag("--no-fix-baseline")
FIX_CAM_SIGN = not _pop_flag("--keep-relu")
FIX_SHARES = not _pop_flag("--sum-shares")
CAM_LAYER = _pop_opt("--cam-layer", "layer3")
sys.argv = [sys.argv[0]] + _ARGV

_STATS = {}          # stashed per fold so the patched IG can rebuild dead-pixel values
_ROWS_SHARES = []
_ROWS_COMPLETE = []
_OUT_DIR = {"path": None}


# ═══════════════════════════════════════════════════════════════════════════
# stash stats + output dir (wrap whatever run_explainability_for_fold is now,
# i.e. the stratified wrapper -- so the class budget still applies)
# ═══════════════════════════════════════════════════════════════════════════
_prev_run = mod.run_explainability_for_fold


def _run_with_stats(model, df_s, va_idx, stats, device, fold_id, task,
                    explain_base_dir, args, explain_idx=None):
    _STATS.clear()
    _STATS.update({
        "ortho_mean": np.asarray(stats["ortho_mean"], dtype=np.float32),
        "ortho_std": np.asarray(stats["ortho_std"], dtype=np.float32),
        "dem_mean": float(stats["dem_mean"]),
        "dem_std": float(stats["dem_std"]),
        "fold": fold_id,
    })
    if _OUT_DIR["path"] is None:
        _OUT_DIR["path"] = getattr(args, "output_dir", None) or explain_base_dir
    return _prev_run(model, df_s, va_idx, stats, device, fold_id, task,
                     explain_base_dir, args, explain_idx=explain_idx)


mod.run_explainability_for_fold = _run_with_stats


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 -- dead-pixel-aware IG baseline  (+ completeness diagnostic)
# ═══════════════════════════════════════════════════════════════════════════
def _dead_masks(x_ortho, x_dem):
    """True where a pixel is nodata/padding, detected per frame.

    dl_core writes 0.0 into those pixels BEFORE normalising, so after
    normalisation they equal exactly (0 - mean)/std. Recompute that constant and
    match it. Requires all bands dead simultaneously, so a real pixel that
    happens to hit the constant in one band is not falsely masked.
    """
    if not _STATS:
        return None, None
    om = torch.as_tensor(_STATS["ortho_mean"], device=x_ortho.device, dtype=x_ortho.dtype)
    os_ = torch.as_tensor(_STATS["ortho_std"], device=x_ortho.device, dtype=x_ortho.dtype)
    const_o = (0.0 - om) / os_                                   # (C,)
    while const_o.dim() < x_ortho.dim():
        const_o = const_o.unsqueeze(0)
    if x_ortho.dim() == 5:                                       # (B,T,C,H,W)
        const_o = const_o.view(1, 1, -1, 1, 1)
        ch_dim = 2
    else:                                                        # (B,C,H,W)
        const_o = const_o.view(1, -1, 1, 1)
        ch_dim = 1
    dead_o = torch.isclose(x_ortho, const_o.expand_as(x_ortho), rtol=0, atol=1e-5)
    dead_o = dead_o.all(dim=ch_dim, keepdim=True).expand_as(x_ortho)

    const_d = (0.0 - _STATS["dem_mean"]) / max(_STATS["dem_std"], 1e-6)
    dead_d = torch.isclose(x_dem, torch.full_like(x_dem, const_d), rtol=0, atol=1e-5)
    return dead_o, dead_d


_orig_ig = mod.integrated_gradients_multiinput


def _ig_fixed(model, x_ortho, x_dem, x_feat, x_time=None, x_fert=None,
              x_weather=None, seq_len=None, steps=32, task="reg", cls_idx=None):
    if not FIX_BASELINE:
        return _orig_ig(model, x_ortho, x_dem, x_feat, x_time=x_time, x_fert=x_fert,
                        x_weather=x_weather, seq_len=seq_len, steps=steps,
                        task=task, cls_idx=cls_idx)

    model.eval()
    dead_o, dead_d = _dead_masks(x_ortho, x_dem)
    z_o = torch.zeros_like(x_ortho)
    z_d = torch.zeros_like(x_dem)
    # baseline == input on dead pixels -> they contribute exactly 0
    b_ortho = x_ortho if dead_o is None else torch.where(dead_o, x_ortho, z_o)
    b_dem = x_dem if dead_d is None else torch.where(dead_d, x_dem, z_d)
    if dead_o is None:
        b_ortho, b_dem = z_o, z_d
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
        xfe = None if x_fert is None else (b_fert + a * (x_fert - b_fert)).detach().requires_grad_(True)
        xw = None if x_weather is None else (b_weather + a * (x_weather - b_weather)).detach().requires_grad_(True)
        mod._set_rnn_train_only(model)
        model.zero_grad(set_to_none=True)
        out_reg, out_cls = model(xo, xd, xf, seq_len, x_time=x_time, x_fert=xfe, x_weather=xw)
        s = out_reg[0, 0] if task == "reg" else out_cls[0, cls_idx or 0]
        s.backward()
        g_ortho += xo.grad.detach()
        g_dem += xd.grad.detach()
        if x_feat is not None: g_feat += xf.grad.detach()
        if x_fert is not None: g_fert += xfe.grad.detach()
        if x_weather is not None: g_weather += xw.grad.detach()

    attr_o = (x_ortho - b_ortho) * (g_ortho / steps)
    attr_d = (x_dem - b_dem) * (g_dem / steps)
    attr_f = None if x_feat is None else (x_feat - b_feat) * (g_feat / steps)
    attr_fe = None if x_fert is None else (x_fert - b_fert) * (g_fert / steps)
    attr_w = None if x_weather is None else (x_weather - b_weather) * (g_weather / steps)

    # ── completeness: signed sum vs f(x) - f(baseline) ──
    try:
        with torch.no_grad():
            fx = model(x_ortho, x_dem, x_feat, seq_len, x_time=x_time,
                       x_fert=x_fert, x_weather=x_weather)[0][0, 0].item()
            fb = model(b_ortho, b_dem, b_feat, seq_len, x_time=x_time,
                       x_fert=b_fert, x_weather=b_weather)[0][0, 0].item()
        signed = sum(float(a.sum()) for a in
                     (attr_o, attr_d, attr_f, attr_fe, attr_w) if a is not None)
        gap = fx - fb
        _ROWS_COMPLETE.append({
            "fold": _STATS.get("fold"), "ig_steps": steps,
            "f_x": fx, "f_baseline": fb, "gap": gap,
            "signed_attr_sum": signed,
            "ratio": signed / gap if abs(gap) > 1e-9 else np.nan,
        })
    except Exception as e:                                    # never kill a run
        _ROWS_COMPLETE.append({"fold": _STATS.get("fold"), "error": str(e)})

    return attr_o, attr_d, attr_f, attr_fe, attr_w


mod.integrated_gradients_multiinput = _ig_fixed


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 -- signed Grad-CAM   /   FIX 3 -- finer layer
# ═══════════════════════════════════════════════════════════════════════════
_orig_cam_call = mod.GradCAM.__call__


def _cam_signed(self, x_ortho, x_dem, x_feat, seq_len, x_time=None, x_fert=None,
                x_weather=None, task="reg", cls_idx=None):
    if FIX_CAM_SIGN:
        _relu, F.relu = F.relu, lambda t, *a, **k: t      # identity for this call
        try:
            return _orig_cam_call(self, x_ortho, x_dem, x_feat, seq_len, x_time=x_time,
                                  x_fert=x_fert, x_weather=x_weather,
                                  task=task, cls_idx=cls_idx)
        finally:
            F.relu = _relu
    return _orig_cam_call(self, x_ortho, x_dem, x_feat, seq_len, x_time=x_time,
                          x_fert=x_fert, x_weather=x_weather, task=task, cls_idx=cls_idx)


mod.GradCAM.__call__ = _cam_signed

_orig_layer = mod._get_gradcam_layer


def _layer_finer(model, branch: str):
    if CAM_LAYER == "layer4":
        return _orig_layer(model, branch)
    backbone = model.ortho_cnn if branch == "ortho" else model.dem_cnn
    tgt = getattr(backbone, CAM_LAYER, None)
    if tgt is not None:
        return tgt[-1] if hasattr(tgt, "__getitem__") else tgt
    feats = getattr(backbone, "features", None)          # ConvNeXt-style
    if feats is not None and len(feats) >= 3:
        return feats[-3]
    return _orig_layer(model, branch)


mod._get_gradcam_layer = _layer_finer


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4 -- per-element modality shares (both variants recorded)
# ═══════════════════════════════════════════════════════════════════════════
_orig_summ = mod.summarize_group_contributions


def _summ_fixed(attr_o, attr_d, attr_f, attr_fert, attr_weather=None, roi_mask=None):
    sums = _orig_summ(attr_o, attr_d, attr_f, attr_fert,
                      attr_weather=attr_weather, roi_mask=roi_mask)

    def _mean(a, mask=None):
        if a is None:
            return 0.0
        v = a.abs()[..., mask] if mask is not None else a.abs()
        return float(v.mean()) if v.numel() else 0.0

    m = None if roi_mask is None else roi_mask[0, 0].bool()
    means = (_mean(attr_o, m), _mean(attr_d, m), _mean(attr_f),
             _mean(attr_fert), _mean(attr_weather))

    tot_s, tot_m = sum(sums) + 1e-12, sum(means) + 1e-12
    names = ["ortho", "dem", "tabular", "fert", "weather"]
    row = {"fold": _STATS.get("fold")}
    row.update({f"{n}_pct_perelement": 100 * v / tot_m for n, v in zip(names, means)})
    row.update({f"{n}_pct_summed": 100 * v / tot_s for n, v in zip(names, sums)})
    row.update({f"{n}_abs_sum": v for n, v in zip(names, sums)})
    _ROWS_SHARES.append(row)

    return means if FIX_SHARES else sums


mod.summarize_group_contributions = _summ_fixed


# ═══════════════════════════════════════════════════════════════════════════
@atexit.register
def _dump():
    out = _OUT_DIR["path"] or "."
    try:
        os.makedirs(out, exist_ok=True)
        if _ROWS_SHARES:
            p = os.path.join(out, "modality_shares_fixed.csv")
            pd.DataFrame(_ROWS_SHARES).to_csv(p, index=False)
            print(f"[fixed] wrote {p}  ({len(_ROWS_SHARES)} samples)", flush=True)
        if _ROWS_COMPLETE:
            p = os.path.join(out, "ig_completeness_check.csv")
            d = pd.DataFrame(_ROWS_COMPLETE)
            d.to_csv(p, index=False)
            if "ratio" in d:
                r = d["ratio"].dropna()
                if len(r):
                    print(f"[fixed] IG completeness ratio: median={r.median():.3f} "
                          f"IQR=[{r.quantile(.25):.3f}, {r.quantile(.75):.3f}] "
                          f"(1.0 = converged; far from 1 -> raise --ig_steps)", flush=True)
            print(f"[fixed] wrote {p}", flush=True)
    except Exception as e:
        print(f"[fixed] dump failed: {e}", flush=True)


if __name__ == "__main__":
    print("[fixed] patches active: "
          f"baseline={FIX_BASELINE}  signed_cam={FIX_CAM_SIGN}  "
          f"cam_layer={CAM_LAYER}  per_element_shares={FIX_SHARES}", flush=True)
    print(f"[fixed] stratified class budget: inherited from "
          f"M7_explainability_stratified (TAKE_ALL_BELOW="
          f"{os.environ.get('EXPLAIN_TAKE_ALL_BELOW', '40')}, "
          f"TARGET_TOTAL={os.environ.get('EXPLAIN_TARGET_TOTAL', '50')})", flush=True)
    mod.main()
