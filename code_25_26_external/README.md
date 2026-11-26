# Independent-season validation: train 2024–2025, test 2025–2026 (K)

Used by `final_runs/run_ml_v4.py --external` and `final_runs/run_dl_v4.py --external`.

| Module | Role |
|---|---|
| `validate_external_25_26.py` | aligns the 2025–2026 table with the training features (helpers for the classical models) |
| `validate_external_dl_25_26.py` | trains M5/M7 on all 2024–2025 observations and predicts 2025–2026 |
| `recalibrate_external.py` | label-informed recalibration of the transferred predictions (additive and linear, plot-grouped 5-fold cross-calibration) |

No 2025–2026 observation is used for training; recalibration uses only
2025–2026 labels of the calibration plots in each cross-calibration fold.
