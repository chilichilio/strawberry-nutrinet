# Base pipeline: leaf K, 2024–2025

Library modules imported by `final_runs/` (run the analyses from there).

| Module | Role |
|---|---|
| `data_prep.py` | loads the modelling table, keeps the closest flight per sampling date, stage labels and stage-specific status classes, feature groups, outlier exclusions |
| `ml_runner.py` | leakage-safe cross-validation engine for PLSR and random forest (fold-local scaling, inner-fold tuning, out-of-fold predictions) |
| `dl_core.py` | deep-learning core: tile dataset, CNN/DEM encoders, TCN sequence branches, temporal Transformer, DDP training and CV loop |
| `M5_CNN_fusion.py`, `M7_Transformer.py` | default configurations of M5 and M7 (data paths, hyperparameters) |
| `M7_explainability*.py` | Integrated Gradients, static-branch SHAP and attention for M7 checkpoints, with class-balanced sample selection |

