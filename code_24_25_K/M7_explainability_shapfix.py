#!/usr/bin/env python3
"""
SHAP memory fix for M7_explainability.py -- NON-INVASIVE.

Nothing in M7_explainability.py or M7_explainability_stratified.py is edited.
This module imports them, monkeypatches ONE function (run_shap_tabular), and
hands control straight back to the stratified driver -- the same pattern the
stratified wrapper already uses on run_explainability_for_fold.

THE PROBLEM
-----------
SHAP attributes only the 17 tabular predictors; images, fertilizer and weather
are held FIXED (see the _TabWrapper docstring in M7_explainability.py). Yet the
tabular SHAP call was the only stage that ran out of memory:

    N (23-24): 10 of 50 samples failed      K (24-25): 21 of 50 failed

and the failures were deterministic and concentrated on image-rich samples
(K survivors: median 1 frame; K failures: median 4 frames). In K this collapsed
the surviving set to 25 deficient / 2 adequate / 2 high, i.e. the SHAP panel
described deficient plots almost exclusively.

WHY IT HAPPENS
--------------
_TabWrapper.forward expands the fixed image tensors to the background batch
size and pushes them through the CNN:

    xo = self.xo.expand(B, *self.xo.shape[1:]).contiguous()

The values are constant, but the CNN's PARAMETERS have requires_grad=True, so
autograd builds a graph through the entire image branch to keep parameter
gradients reachable -- retaining activations for every frame of every
background row. Memory therefore scales with bg_n x frames x activation size.

Lowering --bg_n does NOT fix this. M7_explainability.py already retries
bg -> bg//2 -> 1 on OOM (the logs show 'SHAP OOM with bg=9 / bg=4' for every
failure), and those samples still failed at a background of ONE.

THE FIX
-------
SHAP needs only d(output)/d(x_feat). Freezing the model parameters for the
duration of the SHAP call means no leaf in the image branch requires grad, so
the branch is treated as a constant: activations are freed as they are produced
and the retained graph shrinks to the tabular MLP and the fusion head. The
original requires_grad flags are restored in a finally block, so nothing
downstream observes a modified model.

This changes ONLY peak memory. The forward values, the background set, bg_n and
the resulting SHAP numbers are unchanged, so results stay comparable with the
existing bg_n=10 run.

Integrated Gradients, Grad-CAM and attention are untouched: they set
requires_grad on their INPUTS (M7_explainability.py lines 458-462, 1358, 1389),
not on parameters, and they are separate code paths.

USAGE -- identical CLI to M7_explainability_stratified.py
    python M7_explainability_shapfix.py \
        --checkpoint_dir <ckpt> --output_dir <NEW dir> --fold all --task reg \
        --bg_n 10 --skip_gradcam --skip_ig --skip_attention --skip_occlusion \
        --skip_performance

Verify the patch engaged by looking for '[shapfix]' lines in the log.
"""
import sys

import torch

import M7_explainability as mod

_orig_run_shap_tabular = mod.run_shap_tabular
_ANNOUNCED = False


def _run_shap_tabular_frozen(model, device, *args, **kwargs):
    """run_shap_tabular with model parameters frozen for the call's duration."""
    global _ANNOUNCED

    saved = [(p, p.requires_grad) for p in model.parameters()]
    n_frozen = sum(1 for _, r in saved if r)
    if not _ANNOUNCED:
        print(f"[shapfix] freezing {n_frozen} parameter tensors during SHAP so the "
              f"image branch is not retained for backprop", flush=True)
        _ANNOUNCED = True

    for p, _ in saved:
        p.requires_grad_(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return _orig_run_shap_tabular(model, device, *args, **kwargs)
    finally:
        for p, r in saved:
            p.requires_grad_(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Patch the module attribute. M7_explainability.run_explainability_for_fold
# calls run_shap_tabular(...) as a module-global (line 1461), so the lookup
# resolves to the patched function.
mod.run_shap_tabular = _run_shap_tabular_frozen

# Importing the stratified driver applies ITS patch
# (mod.run_explainability_for_fold = _wrapped_run) at import time. It has no
# main() of its own -- it calls mod.main() under __main__ -- so we import it
# purely for the side effect and then call mod.main() ourselves. Both patches
# are therefore active: class-stratified sample selection AND the SHAP fix.
import M7_explainability_stratified as _strat  # noqa: E402,F401


if __name__ == "__main__":
    print("[shapfix] SHAP memory patch active: parameters frozen during SHAP so "
          "the image branch is not retained for backprop.", flush=True)
    print("[shapfix] SHAP values are UNCHANGED -- only peak memory differs, so "
          "results stay comparable with the existing bg_n=10 run.", flush=True)
    print(f"[shapfix] stratified selection also active "
          f"(argv: {' '.join(sys.argv[1:])})", flush=True)
    sys.exit(mod.main() or 0)
