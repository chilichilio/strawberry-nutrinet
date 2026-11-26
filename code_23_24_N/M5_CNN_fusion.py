# ============================================================
# M5 — CNN fusion (deep learning) for strawberry leaf N (%) prediction
# ------------------------------------------------------------
# Configuration (manuscript Table 2.7):
#     M5 = I + D + C + P + T + Fagg + Wagg,   model = CNN fusion
#       I  multispectral image tiles (ortho, 5 bands)   -> CNN encoder
#       D  DEM tiles (1 band)                           -> CNN encoder
#       (learned image representations REPLACE the handcrafted V,S of M1-M4)
#       C+P+T+Fagg+Wagg  scalar covariates              -> tabular MLP
#
# Architecture: for each unique conc date we take the SINGLE closest UAV image
# (min |image_date - conc_date|, ties -> the earlier/pre-sampling flight). Its
# CNN(ortho)+CNN(dem) embedding is fused with the scalar covariates (CNN FUSION;
# no temporal Transformer, no daily Fseq/Wseq sequences). This is the image-only
# deep model that M7 extends to the full pre-sampling image sequence + temporal
# and sequence modeling.
#
# Multi-GPU via torchrun (DDP). Same samples / folds / labels / metrics as
# M1..M7 (shared through dl_core -> data_prep).
#
# Run (single GPU):   python M5_CNN_fusion.py
# Run (DDP, 2 GPUs):  torchrun --standalone --nproc_per_node=2 M5_CNN_fusion.py
# ============================================================

import os
import dl_core as dc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dc.DLRunConfig(
    model_tag="M5_CNN_fusion",
    config_name="M5",
    image_config="closest_date",     # SINGLE closest UAV image to each conc date
    use_image_temporal=False,        # CNN fusion (single image), NO Transformer
    use_sequences=False,             # NO daily fertilizer/weather sequences
    # task_modes=("reg", "cls"),
    task_modes=("reg",),             # NOTE trailing comma -> 1-tuple, not the string "reg"

    # ── data inputs (resolved relative to the repository root) ───────────
    root_images=os.path.join(BASE, "data", "23_24_N"),
    ortho_filter="ortho_by_plot",
    dem_filter="dem_by_plot",
    # Same tabular table the ML scripts use (provides the scalar covariates +
    # N label + Image_ID/conc_date for matching to the image tiles). Point this
    # at the same file as ml_runner.DEFAULT_EXCEL so ML and DL train on the
    # identical 1,799 closest-flight targets.
    excel_features_path=os.path.join(BASE, "data/23_24_N/trait_N_weather_yield_combined_conc.xlsx"),
    output_dir=os.path.join(BASE, "outputs_23_24", "outputs_M5_CNN_fusion"),
    # ─────────────────────────────────────────────────────────────────────

    epochs=80,
    batch_size=4,                    # single closest image -> modest batch is fine
    ortho_backbone="resnet18",
    dem_backbone="convnext_tiny",
    freeze_backbone_epochs=5,
    use_amp=True,                    # mixed precision: faster, lower GPU memory
)


if __name__ == "__main__":
    dc.run_cv(CFG)
