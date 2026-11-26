#!/usr/bin/env bash
# ============================================================================
# run_all.sh: every run behind the results reported in the article, in order.
#
#   bash run_all.sh            # run everything locally (one GPU is enough; slow)
#   bash run_all.sh --print    # only print the commands (e.g. to submit as jobs)
#
# Run from final_runs/. Data must be in ../data (see ../data/README.md).
# Outputs:
#   ../outputs/<N|K>/<model>__<variant>/          within-season 5-fold CV
#   ../outputs_dlfold/N/<model>__<variant>/       N classical models on the deep-learning folds
#   ../outputs/<N|K>_multiseed/                   M7 with additional seeds
#   ../outputs/K_ext2526/<model>__<variant>/      train 2024-2025, test 2025-2026
#
# Deep-learning runs (M5, M7) need a CUDA GPU; with several GPUs, replace
# "python run_dl_v4.py" by "torchrun --standalone --nproc_per_node=<n> run_dl_v4.py".
# The classical runs are CPU-only; M6 with --nested is the slowest of them.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PRINT=0; [ "${1:-}" = "--print" ] && PRINT=1
run() { echo "+ $*"; [ $PRINT -eq 1 ] || "$@"; }

B="--seqfix --sampEnd"        # final configuration of the sequence models
DLFOLD="V4_FOLD_MANIFEST=$PWD/fold_membership_ML_vs_DL.csv"

# ---- 0. unit test of the sequence-branch fix (CPU, seconds) ----------------
run python test_seq_fix.py

# ---- 1. Table 3 and Supplementary Table S12: progressive models M1-M7 -------
# N classical models are run on the deep-learning outer folds (paired with M5/M7);
# for K the classical and deep-learning folds are already identical.
for m in M1 M2 M3 M4; do
    run env $DLFOLD python run_ml_v4.py --nutrient N --model $m --sampEnd
    run python run_ml_v4.py --nutrient K --model $m --sampEnd
done
# M6 with TreeSHAP (Figure 6, Supplementary Table S10)
run env $DLFOLD python run_ml_v4.py --nutrient N --model M6 $B --nested --shap
run python run_ml_v4.py --nutrient K --model M6 $B --nested --shap
for n in N K; do
    run python run_dl_v4.py --nutrient $n --model M5 --sampEnd
    run python run_dl_v4.py --nutrient $n --model M7 $B --save-checkpoints   # checkpoints feed step 5
done

# ---- 2. Supplementary Table S3: repeated acquisitions / temporal encoding ---
for n in N K; do
    run python run_dl_v4.py --nutrient $n --model M7 $B --single
    run python run_dl_v4.py --nutrient $n --model M7 $B --lstm
    run python run_dl_v4.py --nutrient $n --model M7 $B --noTemp
done

# ---- 3. Supplementary Table S13: input ablations ---------------------------
for n in N K; do
    E=""; [ $n = N ] && E="env $DLFOLD"
    for m in M2 M4; do
        run $E python run_ml_v4.py --nutrient $n --model $m --sampEnd --noUAV
        run $E python run_ml_v4.py --nutrient $n --model $m --sampEnd --noXY
    done
    for f in --noUAV --noXY --noW; do
        run $E python run_ml_v4.py --nutrient $n --model M6 $B --nested $f
    done
    for f in --noUAV --noXY --noW --noF "--noUAV --noF"; do
        run python run_dl_v4.py --nutrient $n --model M7 $B $f
    done
done

# ---- 4. seed-to-seed variability of M7 (Section 3.2) ------------------------
for n in N K; do
    for s in 7 13 2024 99; do run python run_dl_v4.py --nutrient $n --model M7 $B --seed $s; done
done
run python aggregate_seeds_v4.py --tag seqfix_sampEnd

# ---- 5. M7 explanations: static-branch SHAP and Integrated Gradients --------
# (Figure 7, Supplementary Figure S7, Supplementary Tables S8-S10)
# 50 held-out observations per nutrient, balanced over deficient/adequate/high.
export EXPLAIN_TARGET_TOTAL=50 EXPLAIN_TAKE_ALL_BELOW=0 EXPLAIN_STRATIFY_SEED=42
run python explain_v4.py --nutrient N $B
run python explain_v4_cudnn.py --nutrient K $B --shap-batch 5   # keeps cuDNN on for the larger K tiles

# ---- 6. independent 2025-2026 season (Table 4, Supplementary Table S14, Figure S11)
for m in M1 M2 M3 M4; do run python run_ml_v4.py --nutrient K --model $m --sampEnd --external; done
run python run_ml_v4.py --nutrient K --model M4 --sampEnd --noUAV --external
run python run_ml_v4.py --nutrient K --model M4 --sampEnd --noXY --external
for f in "" --noUAV --noXY; do
    run python run_ml_v4.py --nutrient K --model M6 $B --nested $f --external
done
run python run_dl_v4.py --nutrient K --model M5 --sampEnd --external
for f in "" --noUAV --single --noXY --noF "--noF --single"; do
    run python run_dl_v4.py --nutrient K --model M7 $B $f --external
done
EXT=../outputs/K_ext2526
run python ../code_25_26_external/recalibrate_external.py --glob "$EXT/*/external_*_predictions.csv" \
    --summary-csv "$EXT/recalibration_summary.csv"
run python learning_curve_v4.py --root "$EXT" --sizes 5 10 15 20 25 30 35 40 --reps 200 \
    --out-dir "$EXT" --runs M4__sampEnd M7__seqfix_sampEnd
run python collate_external_v4.py --root "$EXT"

# ---- 7. paired fold-level tests (Supplementary Tables S3, S12, S13) --------
run python paired_tests.py
