#!/usr/bin/env python3
"""
Status-CLASS-stratified driver for M7_explainability.py -- NON-INVASIVE.

Why: the stock M7_explainability.py explains the FIRST --n_attr validation
samples of each fold, in fold order -- a deterministic, unbalanced subset.
The scientifically important cases (especially the rare "deficient" nutrient
status) are then barely represented, even though that is exactly the failure
failure mode worth checking.

What this does (imports M7_explainability unchanged, edits nothing in it):
  1. Pools all validation samples across folds (the 5 folds partition the data,
     so the union == the full out-of-fold set).
  2. Labels each sample deficient / adequate / high with the SAME stage-aware
     thresholds the paper uses (dp.classify_{k,n}_value).
  3. Selects a class-balanced explanation budget:
        - TAKE-ALL any class whose total count <= EXPLAIN_TAKE_ALL_BELOW
          (so all ~36 deficient-K samples get explained), then
        - split the REMAINING budget (EXPLAIN_TARGET_TOTAL) evenly across the
          non-rare classes.
  4. Per fold, restricts the run to the selected indices for that fold and sets
     n_attr so exactly those are explained -- deficient cases first.

Result: ~EXPLAIN_TARGET_TOTAL explained samples per season, class-balanced,
reproducible. Performance/accuracy figures (val_predictions_*_ALL.csv, the FULL
set) are unchanged. Nothing in M7_explainability.py is modified.

Placement: drop next to M7_explainability.py + dl_core.py + data_prep.py.

Usage: same CLI as M7_explainability.py; --n_attr is IGNORED (selection drives it):
    python M7_explainability_stratified.py \
        --checkpoint_dir <ckpt_dir> --fold all --task reg \
        --ig_steps 32 --bg_n 10 --max_dates_viz 99 --skip_occlusion

Env knobs:
    EXPLAIN_TARGET_TOTAL    (default 50)  approx explained samples per season
    EXPLAIN_TAKE_ALL_BELOW  (default 40)  classes with <= this many -> take all
    EXPLAIN_STRATIFY_SEED   (default 42)  reproducible within-class subsampling
"""
import os
import sys

import numpy as np

import M7_explainability as mod
import data_prep as dp

_orig_run = mod.run_explainability_for_fold
CLASS_NAMES = ["deficient", "adequate", "high"]

# module-level cache so the global selection is computed once, not per fold
_SELECTED = None          # set[int] of df_s row indices to explain
_CLASS_OF = None          # dict: df_s index -> class idx


def _classify_fn():
    """Return whichever stage-aware classifier this season's data_prep exposes."""
    for name in ("classify_k_value", "classify_n_value"):
        fn = getattr(dp, name, None)
        if fn is not None:
            return fn
    raise AttributeError("data_prep has neither classify_k_value nor classify_n_value")


def _build_global_selection(df_s):
    global _SELECTED, _CLASS_OF
    if _SELECTED is not None:
        return
    target_total = int(os.environ.get("EXPLAIN_TARGET_TOTAL", "50"))
    take_all_below = int(os.environ.get("EXPLAIN_TAKE_ALL_BELOW", "40"))
    seed = int(os.environ.get("EXPLAIN_STRATIFY_SEED", "42"))
    clsfn = _classify_fn()

    # class label for every val sample (whole df_s == union of all folds' val)
    cls_of, by_class = {}, {c: [] for c in range(len(CLASS_NAMES))}
    for idx in df_s.index:
        val = df_s.at[idx, "value"]
        stage = df_s.at[idx, "sampling_stage"]
        c = clsfn(float(val), stage) if val == val else -1   # NaN-safe
        if c in (0, 1, 2):
            cls_of[idx] = c
            by_class[c].append(idx)

    rng = np.random.RandomState(seed)
    counts = {c: len(v) for c, v in by_class.items()}
    rare = [c for c, n in counts.items() if 0 < n <= take_all_below]
    non_rare = [c for c, n in counts.items() if n > take_all_below]

    selected = set()
    for c in rare:                       # take ALL of every rare class
        selected.update(by_class[c])
    remaining = max(target_total - len(selected), 0)
    if non_rare:
        per = remaining // len(non_rare)
        leftover = remaining - per * len(non_rare)
        for k, c in enumerate(non_rare):
            take = min(per + (1 if k < leftover else 0), counts[c])
            pick = rng.choice(by_class[c], size=take, replace=False) if take else []
            selected.update(int(i) for i in pick)

    _SELECTED, _CLASS_OF = selected, cls_of
    bal = ", ".join(f"{CLASS_NAMES[c]}={sum(1 for i in selected if cls_of.get(i)==c)}"
                    f"/{counts.get(c,0)}" for c in range(len(CLASS_NAMES)))
    print(f"[stratify] global selection: {len(selected)} samples "
          f"(target~{target_total}, take-all<= {take_all_below}) -> {bal}", flush=True)


def _wrapped_run(model, df_s, va_idx, stats, device, fold_id, task,
                 explain_base_dir, args, explain_idx=None):
    # NOTE (bugfix): main() calls run_explainability_for_fold(..., explain_idx=...)
    # unconditionally, so this wrapper MUST accept that keyword or the run dies with
    # "TypeError: _wrapped_run() got an unexpected keyword argument 'explain_idx'".
    # The stratified selection built here is the whole point of this script, so it
    # takes precedence over an 'enriched'-mode explain_idx; we only warn if one was
    # supplied so the override is never silent.
    if explain_idx is not None and len(np.asarray(explain_idx)) > 0:
        print(f"[stratify] fold {fold_id}: ignoring enriched explain_idx "
              f"({len(np.asarray(explain_idx))} idx) — class-stratified selection wins.",
              flush=True)
    _build_global_selection(df_s)
    va = np.asarray(va_idx)

    # keep this fold's selected indices; deficient first, then adequate, high
    sel = [i for i in va if i in _SELECTED]
    sel.sort(key=lambda i: (_CLASS_OF.get(i, 9), i))
    new_va = np.asarray(sel, dtype=va.dtype)

    if new_va.size == 0:
        print(f"[stratify] fold {fold_id}: no selected samples -> skipping.", flush=True)
        return {"fold": fold_id, "task": task, "n_val": int(va.size),
                "n_explained": 0, "errors": []}

    args.n_attr = int(new_va.size)       # explain exactly the selected set
    bal = {CLASS_NAMES[c]: sum(1 for i in new_va if _CLASS_OF.get(i) == c)
           for c in range(len(CLASS_NAMES))}
    print(f"[stratify] fold {fold_id}: explaining {new_va.size} selected "
          f"(of {va.size} val) -> {bal}", flush=True)

    return _orig_run(model, df_s, new_va, stats, device, fold_id, task,
                     explain_base_dir, args)


mod.run_explainability_for_fold = _wrapped_run


if __name__ == "__main__":
    print(f"[stratify] class-stratified wrapper active "
          f"(argv: {' '.join(sys.argv[1:])})", flush=True)
    mod.main()
