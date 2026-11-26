#!/usr/bin/env python3
"""
run_dl_v4.py: M7 / M5 revision variants on the base pipeline deep-learning
pipeline (same samples, same GroupKFold folds, same hyperparameters).
Starts from the base pipeline driver's DLRunConfig and changes only what the flags say.

Within-season CV (torchrun, 2 GPUs; slurm/gpu_job.sh):
  python run_dl_v4.py --nutrient K --model M7 --seqfix --sampEnd              # new M7
  ... --noUAV / --single / --noXY / --noW                                     # comparisons
  ... --lstm / --noTemp                                                       # Table S3
  ... --noF [--single] [--noUAV]                                              # no fertilizer/weather records
  ... --seed 7                                                                # extra seeds -> <nut>_multiseed/
Output: outputs/<N|K>/<M7|M5>__<tag>/

External season, K only (single GPU, plain python):
  python run_dl_v4.py --nutrient K --model M7 --seqfix --sampEnd --external
  Trains on all 2024-25 and predicts 2025-26 through base pipeline
  code_25_26_external/validate_external_dl_25_26.py (unchanged; its config
  builder is swapped for the v4 one). Output: outputs/K_ext2526/<model>__<tag>/
  external_<model>__<tag>_predictions.csv (+ metrics), the format the base pipeline
  recalibration scripts read.
"""
import argparse
import dataclasses
import glob
import importlib
import json
import os
import shutil
import sys

import v4_common as v4

DRIVERS = {"M7": "M7_Transformer", "M5": "M5_CNN_fusion"}
EXT_DIR = "code_25_26_external"


def build_cfg(dp, drv, v, out, save_ckpt, model_tag=None):
    base_cfg = drv.CFG
    changes = dict(output_dir=out,
                   scalar_groups=v4.scalar_groups_for(dp, base_cfg.config_name, v),
                   save_checkpoints=bool(save_ckpt))
    if model_tag:
        changes["model_tag"] = model_tag
    if v.single or v.noUAV:
        changes["max_frames"] = 1
    if v.noW:
        changes["weather_path"] = "__v4_weather_disabled__"   # loader fails -> zeros (logged)
    if v.noF:
        changes["use_sequences"] = False                      # no Fseq / Wseq branches
    if v.lstm:
        changes["temporal_encoder"] = "lstm"                  # = base pipeline M7_LSTM.py
    if v.noTemp:
        changes["use_image_temporal"] = False                 # = base pipeline M7_..._no_imgtemporal.py
    return dataclasses.replace(base_cfg, **changes)


def main():
    v4.use_fork_workers()
    ap = argparse.ArgumentParser()
    v4.add_variant_args(ap)
    ap.add_argument("--model", default="M7", choices=list(DRIVERS))
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (default: base pipeline's 42). Non-default seeds go to "
                         "outputs/<nut>_multiseed/")
    ap.add_argument("--external", action="store_true",
                    help="K only: train on all 2024-25, predict 2025-26")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="keep fold .pt files (needed for explain_v4.py)")
    args = ap.parse_args()
    v = v4.Variant.from_args(args)
    if args.model == "M5" and (v.seqfix or v.single or v.noW or v.noUAV or v.noF or v.lstm or v.noTemp):
        raise SystemExit("M5 has no sequence branch and is single-date; only --noXY/--sampEnd apply")
    if v.nested:
        raise SystemExit("--nested is an M6 option")
    if args.external and (args.nutrient != "K" or v.treat or args.seed is not None):
        raise SystemExit("--external is for K (2024-25 -> 2025-26), without --treat/--seed")
    if v.lstm and v.noTemp:
        raise SystemExit("--lstm and --noTemp are alternative S3 ablations; pick one")

    dp, _, dc = v4.setup(args.nutrient, need_dl=True)
    v4.patch_load_samples(dp, args.nutrient, v)
    v4.patch_dl_samples(dc, dp, args.nutrient, v.single)
    if v.seqfix:
        v4.patch_seqfix(dc)
    if v.noUAV:
        v4.patch_no_uav(dc)
    if v.treat:
        v4.patch_treatfold(dp, dc, args.nutrient)
    if args.seed is not None:
        dc.GLOBAL_SEED = int(args.seed)   # read at call time by run_cv / train_one_fold (as base pipeline run_multiseed.py)

    drv = importlib.import_module(DRIVERS[args.model])     # module-level CFG only (main is guarded)
    out = v4.run_dir(args.nutrient, args.model, v,
                     seed=args.seed if args.seed not in (None, 42) else None,
                     external=args.external)
    run_name = f"{args.model}__{v.tag()}"
    cfg = build_cfg(dp, drv, v, out, args.save_checkpoints or args.external,
                    model_tag=(f"{DRIVERS[args.model]}_ext2526" if args.external else None))

    rec = {"variant": dataclasses.asdict(v), "nutrient": args.nutrient, "model": args.model,
           "seed": args.seed if args.seed is not None else 42, "external": args.external,
           "base pipeline_driver": drv.__file__,
           "config": {k: (list(x) if isinstance(x, tuple) else x)
                      for k, x in dataclasses.asdict(cfg).items()}}
    if int(os.environ.get("RANK", "0")) == 0:
        with open(os.path.join(out, "v4_run_config.json"), "w") as f:
            json.dump(rec, f, indent=2, default=str)
        print(json.dumps(rec, indent=2, default=str), flush=True)
    if args.dry_run:
        return

    if not args.external:
        dc.run_cv(cfg)
        return

    # ── external season: reuse the base pipeline harness with the v4 config ──
    ext_dir = v4.EXT_CODE_DIR
    sys.path.insert(0, ext_dir)
    ext = importlib.import_module("validate_external_dl_25_26")
    ext._make_cfg = lambda _dc, _model, _out: cfg
    sys.argv = [ext.__file__, "--config", args.model,
                "--code-dir", os.path.join(v4.BASE, v4.NUTRIENTS["K"]["code_dir"]),
                "--output-dir", out]
    ext.main()
    # rename to a per-run name so the recalibration / learning-curve scripts can tell runs apart
    for f in glob.glob(os.path.join(out, f"external_{args.model}_*")):
        new = f.replace(f"external_{args.model}_", f"external_{run_name}_")
        shutil.copyfile(f, new)
    _write_transfer_metrics(out, run_name)


def _write_transfer_metrics(out, run_name):
    import pandas as pd
    p = os.path.join(out, f"external_{run_name}_predictions.csv")
    if not os.path.exists(p):
        print(f"[v4] WARN no predictions file {p}", flush=True)
        return
    d = pd.read_csv(p)
    m = v4.transfer_metrics(d["y_true_reg"], d["y_pred_reg"])
    with open(os.path.join(out, f"external_{run_name}_transfer.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"[v4] external {run_name}: {m}", flush=True)


if __name__ == "__main__":
    main()
