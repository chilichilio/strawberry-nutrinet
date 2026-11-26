# ============================================================
# M7 — Temporal multimodal Transformer (deep learning) for leaf K (%)
# ------------------------------------------------------------
# Configuration (manuscript Table 2.7):
#     M7 = I_seq + D_seq + C + P + T + Fagg + Wagg + Fseq + Wseq,
#          model = temporal multimodal Transformer
#       I_seq/D_seq  UAV image-date SEQUENCE (interval-encoded)  -> CNN+Transformer
#       Fseq         daily fertilizer series (transplant -> DAP) -> TCN branch
#       Wseq         daily FAWN weather series                   -> TCN branch
#       C+P+T+Fagg+Wagg  scalar covariates                       -> tabular MLP
#
# This is the full temporal-fusion model: per-frame CNN(ortho)+CNN(dem)
# embeddings for ALL UAV images on/before the closest flight (the same date M5
# uses as its single image) are passed through a Transformer over the image-date
# sequence, attention-pooled, and fused with TCN-encoded daily fertilizer +
# weather sequences and the scalar covariates. It is the deep counterpart that
# M5 (CNN fusion) and the classical M6 (RF + Eseq) are benchmarked against.
#
# Builds on 25c_interval_encoding_transformer_allmodes.py; for paper-grade
# explainability run M7_explainability.py (in this same folder) on the saved
# .pt checkpoints this script writes -- it imports dl_core.py directly, so
# the rebuilt samples/folds/stats/model are guaranteed to match this run
# exactly. (25c_run_explainability_from_saved_pt.py / 25d_explainability_only.py
# target the earlier N%/23-24 checkpoints, not these M7/K% ones.)
#
# Multi-GPU via torchrun (DDP). Same samples / folds / labels / metrics as
# M1..M7 (shared through dl_core -> data_prep).
#
# Run (single GPU):   python M7_Transformer.py
# Run (DDP, 2 GPUs):  torchrun --standalone --nproc_per_node=2 M7_Transformer.py
# ============================================================

import os
import dl_core as dc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dc.DLRunConfig(
    model_tag="M7_Transformer",
    config_name="M7",
    image_config="on_before_closest",  # ALL images on/before the closest flight (M5's date)
    use_image_temporal=True,         # Transformer over the image-date sequence
    use_sequences=True,              # TCN fertilizer (Fseq) + TCN weather (Wseq)
    # task_modes=("reg", "cls"),
    task_modes=("reg",),             # NOTE trailing comma -> 1-tuple, not the string "reg"

    # ── data inputs (resolved relative to the repository root) ───────────
    root_images=os.path.join(BASE, "data", "24_25_K"),
    ortho_filter="ortho_by_plot",
    dem_filter="dem_by_plot",
    # Same tabular table the ML scripts use (provides the scalar covariates +
    # K label + Image_ID/conc_date for matching to the image tiles). Point this
    # at the same file as ml_runner.DEFAULT_EXCEL so ML and DL train on the
    # identical 1,259 analyzed closest-flight targets.
    excel_features_path=os.path.join(BASE, "data/24_25_K/trait_K_weather_yield_combined_conc.xlsx"),
    fert_path=os.path.join(BASE, "data/24_25_K/fert_by_plot_daily_lb2kg.xlsx"),
    weather_path=os.path.join(BASE, "data/24_25_K/FAWN_report_metric_24_25.xlsx"),
    output_dir=os.path.join(BASE, "outputs_24_25", "outputs_M7_Transformer"),
    # ─────────────────────────────────────────────────────────────────────

    epochs=80,
    batch_size=2,                    # image sequences can be long -> small batch
    ortho_backbone="resnet18",
    dem_backbone="convnext_tiny",
    freeze_backbone_epochs=5,

    # ── memory controls (this job is host-RAM bound) ─────────────────────────
    # Use ALL frames at full image resolution (no max_frames, no crop_cap).
    # The OOM was host RAM during training, so we only pull the levers that do
    # NOT change the model or its inputs and therefore do NOT affect the
    # cross-model comparison:
    #   * use_amp           -> mixed precision (GPU-mem + speed only)
    #   * num_workers       -> fewer DataLoader prefetch copies in host RAM
    #   * persistent_workers -> off, so workers are torn down between folds
    #                           (prevents host-RAM creep across the 5 folds)
    # Combined with --mem=140gb in run_M7_Transformer.sh.
    use_amp=True,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
)


if __name__ == "__main__":
    dc.run_cv(CFG)
